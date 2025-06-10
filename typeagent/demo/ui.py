# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import io
from pprint import pprint
import readline
import shutil
import sys
import traceback
from typing import Any

import typechat

from ..aitools.auth import load_dotenv
from ..knowpro.convknowledge import create_typechat_model
from ..knowpro.interfaces import (
    DateRange,
    Datetime,
    IConversation,
    IMessage,
    SemanticRef,
    Topic,
)
from ..knowpro.kplib import Action, ActionParam, ConcreteEntity, Quantity
from ..knowpro.query import (
    QueryEvalContext,
)
from ..knowpro.search import SearchQueryExpr, run_search_query
from ..podcasts.podcast import Podcast

from .search_query_schema import ActionTerm, SearchQuery
from .querycompiler import SearchQueryCompiler

cap = min  # More readable name for capping a value at some limit


def main() -> None:
    load_dotenv()
    translator = create_translator()
    file = "testdata/Episode_53_AdrianTchaikovsky_index"
    pod = Podcast.read_from_file(file)
    assert pod is not None, f"Failed to load podcast from {file!r}"
    # TODO: change QueryEvalContext to take [TMessage, TTermToSemanticRefIndex].
    context = QueryEvalContext(conversation=pod)  # type: ignore  # See TODO above
    print("TypeAgent demo UI 0.1 (type 'q' to exit)")
    if sys.stdin.isatty():
        try:
            readline.read_history_file(".ui_history")
        except FileNotFoundError:
            pass  # Ignore if history file does not exist.
    try:
        process_inputs(translator, context, sys.stdin)  # type: ignore  # Why is stdin not a TextIOWrapper?!
    except KeyboardInterrupt:
        print()
    finally:
        if sys.stdin.isatty():
            readline.write_history_file(".ui_history")


def create_translator() -> typechat.TypeChatJsonTranslator[SearchQuery]:
    model = create_typechat_model()  # TODO: Move out of here.
    schema = SearchQuery  # TODO: Use SearchTermGroup when ready.
    validator = typechat.TypeChatValidator[schema](schema)
    translator = typechat.TypeChatJsonTranslator[schema](model, validator, schema)
    # schema_text = translator._schema_str.rstrip()  # type: ignore  # No other way.
    # print(f"TypeScript schema for {schema.__name__}:\n{schema_text}\n")
    return translator


def process_inputs(
    translator: typechat.TypeChatJsonTranslator[SearchQuery],
    context: QueryEvalContext,
    stream: io.TextIOWrapper,
) -> None:
    conversation = context.conversation
    ps1 = "--> "
    while True:
        query_text = read_one_line(ps1, stream)
        if query_text is None:  # EOF
            break
        if not query_text:
            continue
        if query_text.lower() in ("exit", "quit", "q"):
            readline.remove_history_item(readline.get_current_history_length() - 1)
            break
        if query_text == "pdb":
            breakpoint()
            continue

        asyncio.run(wrap_process_query(query_text, context.conversation, translator))

        print()


def read_one_line(ps1: str, stream: io.TextIOWrapper) -> str | None:
    """Read a single line from the input stream. Return None for EOF."""
    if stream is sys.stdin and stream.isatty():
        try:
            return input(ps1).strip()
        except EOFError:
            print()
            return None
    else:
        if stream.isatty():
            print(ps1, end="", flush=True)
        line = stream.readline()
        if not line:
            print()
            return None
        return line.strip()


async def wrap_process_query(query_text, conversation, translator):
    try:
        await process_query(query_text, conversation, translator)
    except Exception as exc:
        traceback.print_exc()
        # traceback.print_exception(type(exc), exc, exc.__traceback__.tb_next)


async def process_query(
    query_text: str,
    conversation: IConversation[IMessage, Any],
    translator: typechat.TypeChatJsonTranslator[SearchQuery],
):
    line_width = cap(200, shutil.get_terminal_size().columns)

    # Gradually turn the query text into something we can use to search.

    # TODO: # 0. Recognize @-commands like "@search" and handle them specially.

    # 1. With LLM help, translate to SearchQuery (a tree but not yet usable to query)
    print("Search query:")
    search_query = await translate_text_to_search_query(
        conversation, translator, query_text
    )
    if search_query is None:
        print("Failed to translate command to search terms.")
        return
    pprint(search_query, width=line_width)
    print()

    # 2. Translate the search query into something directly usable as a query.
    print("Search query expressions:")
    query_exprs = translate_search_query_to_search_query_exprs(search_query)
    if not query_exprs:
        print("Failed to translate search query to query expressions.")
        return
    for i, query_expr in enumerate(query_exprs):
        print(f"---------- {i} ----------")
        pprint(query_expr, width=line_width)
    print()

    # 3. Search!
    for i, query_expr in enumerate(query_exprs):
        print(f"Searching with expression {i}:")
        results = await run_search_query(conversation, query_expr)
        if results is None:
            print(f"No results for expression {i}.")
        else:
            print(f"Results for expression {i}:")
            # pprint(results, width=line_width)
            for result in results:
                print(f"Raw query: {result.raw_query_text}")
                if result.message_matches:
                    print("Message matches:", result.message_matches)
                if result.knowledge_matches:
                    print("Knowledge matches:")
                    for key, value in sorted(result.knowledge_matches.items()):
                        print(f"Type {key}:")
                        print(f"  {value.term_matches}")
                        for scored_sem_ref_ord in value.semantic_ref_matches:
                            score = scored_sem_ref_ord.score
                            sem_ref_ord = scored_sem_ref_ord.semantic_ref_ordinal
                            if conversation.semantic_refs is None:
                                print(f"  Ord: {sem_ref_ord} (score {score})")
                            else:
                                sem_ref = conversation.semantic_refs[sem_ref_ord]
                                msg_ord = sem_ref.range.start.message_ordinal
                                chunk_ord = sem_ref.range.start.chunk_ordinal
                                msg = conversation.messages[msg_ord]
                                print(
                                    f"({score:4.1f}) {msg_ord:3d}: "
                                    f"{msg.speaker:>15.15s}: "  # type: ignore  # It's a PodcastMessage
                                    f"{repr(msg.text_chunks[chunk_ord].strip())[1:-1]:<50.50s}  "
                                    f"{summarize_knowledge(sem_ref)}"
                                )


async def translate_text_to_search_query(
    conversation: IConversation,
    translator: typechat.TypeChatJsonTranslator[SearchQuery],
    text: str,
) -> SearchQuery | None:
    prompt_preamble: list[typechat.PromptSection] = []
    time_range_preamble = get_time_range_prompt_section_for_conversation(conversation)
    if time_range_preamble is not None:
        prompt_preamble.append(time_range_preamble)
    result: typechat.Result[SearchQuery] = await translator.translate(
        text, prompt_preamble=prompt_preamble
    )
    if isinstance(result, typechat.Failure):
        print(f"Error translating {text!r}: {result.message}")
        return None
    return result.value


def translate_search_query_to_search_query_exprs(
    search_query: SearchQuery,
) -> list[SearchQueryExpr]:
    return SearchQueryCompiler().compile_query(search_query)


def summarize_knowledge(sem_ref: SemanticRef) -> str:
    """Summarize the knowledge in a SemanticRef."""
    knowledge = sem_ref.knowledge
    if knowledge is None:
        return "<No knowledge>"
    match sem_ref.knowledge_type:
        case "entity":
            entity = knowledge
            assert isinstance(entity, ConcreteEntity)
            res = [f"{entity.name} [{', '.join(entity.type)}]"]
            if entity.facets:
                for facet in entity.facets:
                    value = facet.value
                    if isinstance(value, Quantity):
                        value = f"{value.amount} {value.units}"
                    res.append(f"<{facet.name}:{value}>")
            return " ".join(res)
        case "action":
            action = knowledge
            assert isinstance(action, Action)
            res = []
            res.append("/".join(repr(verb) for verb in action.verbs))
            if action.verb_tense:
                res.append(f"[{action.verb_tense}]")
            if action.subject_entity_name != "none":
                res.append(f"subj={action.subject_entity_name!r}")
            if action.object_entity_name != "none":
                res.append(f"obj={action.object_entity_name!r}")
            if action.indirect_object_entity_name != "none":
                res.append(f"ind_obj={action.indirect_object_entity_name}")
            if action.params:
                for param in action.params:
                    if isinstance(param, ActionParam):
                        res.append(f"<{param.name}:{param.value}>")
                    else:
                        res.append(f"<{param}>")
            if action.subject_entity_facet is not None:
                res.append(f"subj_facet={action.subject_entity_facet}")
            return " ".join(res)
        case "topic":
            topic = knowledge
            assert isinstance(topic, Topic)
            return repr(topic.text)
        case "tag":
            tag = knowledge
            assert isinstance(tag, str)
            return f"#{tag}"
        case _:
            return str(sem_ref.knowledge)


# TODO: Move to conversation.py
def get_time_range_prompt_section_for_conversation(
    conversation: IConversation,
) -> typechat.PromptSection | None:
    time_range = get_time_range_for_conversation(conversation)
    if time_range is not None:
        return typechat.PromptSection(
            role="system",
            content=f"ONLY IF user request explicitly asks for time ranges, "
            f'THEN use the CONVERSATION TIME RANGE: "{time_range.start} to {time_range.end}"',
        )


# TODO: Move to conversation.py
def get_time_range_for_conversation(
    conversation: IConversation[IMessage, Any],
) -> DateRange | None:
    messages = conversation.messages
    if len(messages) > 0:
        start = messages[0].timestamp
        if start is not None:
            end = messages[-1].timestamp
            return DateRange(
                start=Datetime.fromisoformat(start),
                end=Datetime.fromisoformat(end) if end else None,
            )
    return None


if __name__ == "__main__":
    main()
