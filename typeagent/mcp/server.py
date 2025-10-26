# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fledgling MCP server on top of knowpro."""

from dataclasses import dataclass
import time
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.shared.context import RequestContext
from mcp.types import TextContent
import typechat

from typeagent.aitools import embeddings, utils
from typeagent.aitools.embeddings import AsyncEmbeddingModel
from typeagent.knowpro import answers, query, searchlang
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.knowpro.answer_response_schema import AnswerResponse
from typeagent.knowpro.search_query_schema import SearchQuery
from typeagent.podcasts import podcast


class MCPTypeChatModel(typechat.TypeChatLanguageModel):
    """TypeChat language model that uses MCP sampling API instead of direct API calls."""

    def __init__(self, session: Any):
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
            # Convert prompt to message format
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                # PromptSection list: convert to messages
                messages = []
                for section in prompt:
                    role = "user" if section["role"] == "user" else "assistant"
                    messages.append({"role": role, "content": section["content"]})

            # Use MCP sampling to request completion from client
            result = await self.session.create_message(
                messages=messages, max_tokens=16384
            )

            # Extract text content from response
            # MCP responses have a content field that can be TextContent or a list
            if isinstance(result.content, TextContent):
                return typechat.Success(result.content.text)
            elif isinstance(result.content, list):
                # Handle list of content items
                text_parts = []
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
    query_context: query.QueryEvalContext
    embedding_model: embeddings.AsyncEmbeddingModel
    query_translator: typechat.TypeChatJsonTranslator[SearchQuery]
    answer_translator: typechat.TypeChatJsonTranslator[AnswerResponse]

    def __repr__(self) -> str:
        parts = []
        parts.append(f"{self.lang_search_options}")
        parts.append(f"{self.answer_context_options}")
        return f"Context({', '.join(parts)})"


async def make_context(session: Any) -> ProcessingContext:
    """Create processing context using MCP-based language model.

    Args:
        session: The MCP server session that provides create_message() for sampling.

    Note: Embeddings still require API keys since MCP doesn't support embeddings yet.
    Make sure to set OPENAI_API_KEY or AZURE_OPENAI_API_KEY for embeddings.
    """
    utils.load_dotenv()

    settings = ConversationSettings()
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

    query_context = await load_podcast_index(
        "testdata/Episode_53_AdrianTchaikovsky_index", settings
    )

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


async def load_podcast_index(
    podcast_file_prefix: str, settings: ConversationSettings
) -> query.QueryEvalContext:
    conversation = await podcast.Podcast.read_from_file(podcast_file_prefix, settings)
    assert (
        conversation is not None
    ), f"Failed to load podcast from {podcast_file_prefix!r}"
    return query.QueryEvalContext(conversation)


# Create an MCP server
mcp = FastMCP("knowpro")


@dataclass
class QuestionResponse:
    success: bool
    answer: str
    time_used: int  # Milliseconds


@mcp.tool()
async def query_conversation(question: str, ctx: RequestContext) -> QuestionResponse:
    """Send a question to the memory server and get an answer back"""
    t0 = time.time()
    question = question.strip()
    if not question:
        dt = int((time.time() - t0) * 1000)  # Convert to milliseconds
        return QuestionResponse(
            success=False, answer="No question provided", time_used=dt
        )
    context = await make_context(ctx.session)

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
    # Use stdio transport for simplicity
    mcp.run(transport="stdio")
