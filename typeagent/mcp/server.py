# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fledgling MCP server on top of typeagent."""


import argparse
import time
from dataclasses import dataclass
from typing import Any

import coverage
import typechat
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from mcp.types import SamplingMessage, TextContent

# Enable coverage.py before local imports (a no-op unless COVERAGE_PROCESS_START is set).
coverage.process_startup()

from typeagent.aitools import embeddings, utils
from typeagent.knowpro import answers, query, searchlang
from typeagent.knowpro.answer_response_schema import AnswerResponse
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.knowpro.search_query_schema import SearchQuery
from typeagent.podcasts import podcast
from typeagent.storage.memory.semrefindex import TermToSemanticRefIndex
from typeagent.storage.utils import create_storage_provider


class MCPTypeChatModel(typechat.TypeChatLanguageModel):
    """TypeChat language model that uses MCP sampling API instead of direct API calls."""

    def __init__(self, session: ServerSession):
        """Initialize with MCP session for sampling.

        Args:
            session: The MCP server session that provides create_message() for sampling.
        """
        self.session = session

    async def complete(
        self, prompt: str | list[typechat.PromptSection]
    ) -> typechat.Result[str]:
        """Request completion from the MCP client's LLM."""
        try:
            # Convert prompt to MCP SamplingMessage format
            sampling_messages: list[SamplingMessage]
            if isinstance(prompt, str):
                sampling_messages = [
                    SamplingMessage(
                        role="user", content=TextContent(type="text", text=prompt)
                    )
                ]
            else:
                # PromptSection list: convert to SamplingMessage
                sampling_messages = []
                for section in prompt:
                    role = "user" if section["role"] == "user" else "assistant"
                    sampling_messages.append(
                        SamplingMessage(
                            role=role,
                            content=TextContent(type="text", text=section["content"]),
                        )
                    )

            # Use MCP sampling to request completion from client
            result = await self.session.create_message(
                messages=sampling_messages, max_tokens=4096
            )

            # Extract text content from response
            # MCP responses have a content field that can be TextContent or a list
            if isinstance(result.content, TextContent):
                return typechat.Success(result.content.text)
            elif isinstance(result.content, list):
                # Handle list of content items
                text_parts: list[str] = []
                for item in result.content:
                    if isinstance(item, TextContent):
                        text_parts.append(item.text)
                if text_parts:
                    return typechat.Success("\n".join(text_parts))
                else:
                    return typechat.Failure("No text content in MCP response")
            else:
                return typechat.Failure("No text content in MCP response")

        except Exception as e:
            return typechat.Failure(f"MCP sampling failed: {e!r}")


@dataclass
class ProcessingContext:
    lang_search_options: searchlang.LanguageSearchOptions
    answer_context_options: answers.AnswerContextOptions
    query_context: query.QueryEvalContext[
        podcast.PodcastMessage, TermToSemanticRefIndex
    ]
    embedding_model: embeddings.AsyncEmbeddingModel
    query_translator: typechat.TypeChatJsonTranslator[SearchQuery]
    answer_translator: typechat.TypeChatJsonTranslator[AnswerResponse]

    def __repr__(self) -> str:
        parts: list[str] = []
        parts.append(f"{self.lang_search_options}")
        parts.append(f"{self.answer_context_options}")
        return f"Context({', '.join(parts)})"


async def make_context(
    session: ServerSession, dbname: str | None = None
) -> ProcessingContext:
    """Create processing context using MCP-based language model.

    Args:
        session: The MCP server session that provides create_message() for sampling.
        dbname: Path to SQLite database file, or None to load from JSON file.

    Note: Embeddings still require API keys since MCP doesn't support embeddings yet.
    Make sure to set OPENAI_API_KEY or AZURE_OPENAI_API_KEY for embeddings.
    """
    settings = ConversationSettings()

    # Uses SQLite provider if dbname is specified, otherwise use memory provider
    settings.storage_provider = await create_storage_provider(
        settings.message_text_index_settings,
        settings.related_term_index_settings,
        dbname,
        podcast.PodcastMessage,
    )

    lang_search_options = searchlang.LanguageSearchOptions(
        compile_options=searchlang.LanguageQueryCompileOptions(
            exact_scope=False, verb_scope=True, term_filter=None, apply_scope=True
        ),
        exact_match=False,
        max_message_matches=25,
    )
    answer_context_options = answers.AnswerContextOptions(
        entities_top_k=50, topics_top_k=50, messages_top_k=None, chunking=None
    )

    query_context = await load_podcast_index_or_database(settings, dbname)

    # Use MCP-based model instead of one that requires API keys
    model = MCPTypeChatModel(session)
    query_translator = utils.create_translator(model, SearchQuery)
    answer_translator = utils.create_translator(model, AnswerResponse)

    context = ProcessingContext(
        lang_search_options,
        answer_context_options,
        query_context,
        settings.embedding_model,
        query_translator,
        answer_translator,
    )

    return context


async def load_podcast_index_or_database(
    settings: ConversationSettings,
    dbname: str | None = None,
) -> query.QueryEvalContext[podcast.PodcastMessage, Any]:
    if dbname is None:
        conversation = await podcast.Podcast.read_from_file(
            "testdata/Episode_53_AdrianTchaikovsky_index", settings
        )
    else:
        conversation = await podcast.Podcast.create(settings)
    return query.QueryEvalContext(conversation)


# Create an MCP server
mcp = FastMCP("typagent")

# Global variable to store database path (set via command-line argument)
_dbname: str | None = None


@dataclass
class QuestionResponse:
    success: bool
    answer: str
    time_used: int  # Milliseconds


@mcp.tool()
async def query_conversation(
    question: str, ctx: Context[ServerSession, Any, Any]
) -> QuestionResponse:
    """Send a question to the memory server and get an answer back"""
    t0 = time.time()
    question = question.strip()
    if not question:
        dt = int((time.time() - t0) * 1000)  # Convert to milliseconds
        return QuestionResponse(
            success=False, answer="No question provided", time_used=dt
        )
    context = await make_context(ctx.request_context.session, _dbname)

    # Stages 1, 2, 3 (LLM -> proto-query, compile, execute query)
    result = await searchlang.search_conversation_with_language(
        context.query_context.conversation,
        context.query_translator,
        question,
        context.lang_search_options,
    )
    if isinstance(result, typechat.Failure):
        dt = int((time.time() - t0) * 1000)  # Convert to milliseconds
        return QuestionResponse(success=False, answer=result.message, time_used=dt)

    # Stages 3a, 4 (ordinals -> messages/semrefs, LLM -> answer)
    _, combined_answer = await answers.generate_answers(
        context.answer_translator,
        result.value,
        context.query_context.conversation,
        question,
        options=context.answer_context_options,
    )
    dt = int((time.time() - t0) * 1000)  # Convert to milliseconds
    match combined_answer.type:
        case "NoAnswer":
            return QuestionResponse(
                success=False, answer=combined_answer.whyNoAnswer or "", time_used=dt
            )
        case "Answered":
            return QuestionResponse(
                success=True, answer=combined_answer.answer or "", time_used=dt
            )


# Run the MCP server
if __name__ == "__main__":
    # Load env vars
    utils.load_dotenv()

    # Set up command-line argument parsing and parse command line
    parser = argparse.ArgumentParser(description="MCP server for knowpro")
    parser.add_argument(
        "-d",
        "--database",
        type=str,
        default=None,
        help="Path to the SQLite database file (default: load from JSON file)",
    )
    args = parser.parse_args()

    # Store database path in global variable (no other straightforward way to pass to tool)
    _dbname = args.database

    # Use stdio transport for simplicity
    mcp.run(transport="stdio")
