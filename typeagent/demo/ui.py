# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
from contextlib import contextmanager
import time
import io
import re
import sys
import traceback
from typing import cast

try:
    import readline
except ImportError:
    readline = None

import typechat

from ..aitools.utils import create_translator, load_dotenv, pretty_print, timelog
from ..knowpro.answer_response_schema import AnswerResponse
from ..knowpro.answers import generate_answers
from ..knowpro.convknowledge import create_typechat_model
from ..knowpro.importing import ConversationSettings
from ..knowpro.interfaces import (
    IConversation,
    IMessage,
    ITermToSemanticRefIndex,
    SemanticRef,
    Topic,
)
from ..knowpro.kplib import Action, ActionParam, ConcreteEntity, Quantity
from ..knowpro.query import QueryEvalContext
from ..knowpro.search import ConversationSearchResult
from ..knowpro.searchlang import (
    LanguageSearchDebugContext,
    search_conversation_with_language,
)
from ..knowpro.search_query_schema import SearchQuery
from ..podcasts.podcast import Podcast


def main() -> None:
    load_dotenv()
    model = create_typechat_model()
    with timelog("create typechat translator"):
        query_translator = create_translator(model, SearchQuery)
        answer_translator = create_translator(model, AnswerResponse)

    file = "testdata/Episode_53_AdrianTchaikovsky_index"
    with timelog("create conversation settings"):
        settings = ConversationSettings()
    with timelog("load podcast"):
        pod = Podcast.read_from_file(file, settings)
    assert pod is not None, f"Failed to load podcast from {file!r}"
    context = QueryEvalContext(pod)

    print("TypeAgent demo UI 0.1 (type 'q' to exit)")
    if readline and sys.stdin.isatty():
        try:
            readline.read_history_file(".ui_history")
        except FileNotFoundError:
            pass  # Ignore if history file does not exist.
    try:
        process_inputs(
            query_translator,
            answer_translator,
            context,
            cast(io.TextIOWrapper, sys.stdin),
        )
    except KeyboardInterrupt:
        print()
    finally:
        if readline and sys.stdin.isatty():
            readline.write_history_file(".ui_history")


def process_inputs[TMessage: IMessage, TIndex: ITermToSemanticRefIndex](
    query_translator: typechat.TypeChatJsonTranslator[SearchQuery],
    answer_translator: typechat.TypeChatJsonTranslator[AnswerResponse],
    context: QueryEvalContext[TMessage, TIndex],
    stream: io.TextIOWrapper,
) -> None:
    ps1 = "--> "
    while True:
        query_text = read_one_line(ps1, stream)
        if query_text is None:  # EOF
            break
        match query_text:  # Already stripped
            case "":
                continue
            case "exit" | "q" | "quit":
                if readline:
                    readline.remove_history_item(
                        readline.get_current_history_length() - 1
                    )
                break
            case "pdb":
                pretty_print(
                    asyncio.run(
                        context.conversation.secondary_indexes.term_to_related_terms_index.fuzzy_index.lookup_term(  # type: ignore
                            "novel"
                        )
                    )
                )
                print("Entering debugger; end with 'c' or 'continue'.")
                breakpoint()  # Do not remove -- 'pdb' should enter the debugger.
            case _ if re.match(r"^\d+$", query_text):
                msg_ord = int(query_text)
                messages = context.conversation.messages
                if msg_ord < 0 or msg_ord >= len(messages):
                    print(f"Message ordinal {msg_ord} out of range({len(messages)}).")
                    continue
                pretty_print(messages[msg_ord])
            case _:
                with timelog("Query processing"):
                    asyncio.run(
                        wrap_process_query(
                            query_text,
                            context.conversation,
                            query_translator,
                            answer_translator,
                        )
                    )


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


async def wrap_process_query[TMessage: IMessage, TIndex: ITermToSemanticRefIndex](
    query_text: str,
    conversation: IConversation[TMessage, TIndex],
    query_translator: typechat.TypeChatJsonTranslator[SearchQuery],
    answer_translator: typechat.TypeChatJsonTranslator[AnswerResponse],
) -> None:
    """Wrap the process_query function to handle exceptions."""
    try:
        await process_query(
            query_text, conversation, query_translator, answer_translator
        )
    except Exception as exc:
        traceback.print_exc()
        # traceback.print_exception(type(exc), exc, exc.__traceback__.tb_next)


async def process_query[TMessage: IMessage, TIndex: ITermToSemanticRefIndex](
    orig_query_text: str,
    conversation: IConversation[TMessage, TIndex],
    query_translator: typechat.TypeChatJsonTranslator[SearchQuery],
    answer_translator: typechat.TypeChatJsonTranslator[AnswerResponse],
) -> None:
    debug_context = LanguageSearchDebugContext()  # For lots of debug output.
    result = await search_conversation_with_language(
        conversation,
        query_translator,
        orig_query_text,
        debug_context=debug_context,
    )
    if debug_context:
        if debug_context.search_query:
            print("-" * 50)
            pretty_print(debug_context.search_query)
        if debug_context.search_query_expr:
            print("-" * 50)
            pretty_print(debug_context.search_query_expr)
    if not isinstance(result, typechat.Success):
        print(f"Error searching conversation: {result.message}")
        return
    search_results = result.value
    for sr in search_results:
        print("-" * 50)
        print_result(sr, conversation)
    all_answers, combined_answer = await generate_answers(
        answer_translator, search_results, conversation, orig_query_text
    )
    print("-" * 40)
    pretty_print(all_answers)
    print("-" * 40)
    if combined_answer.type == "NoAnswer":
        print(f"Failure: {combined_answer.whyNoAnswer}")
    else:
        print(combined_answer.answer)
    print("-" * 40)


def print_result[TMessage: IMessage, TIndex: ITermToSemanticRefIndex](
    result: ConversationSearchResult, conversation: IConversation[TMessage, TIndex]
) -> None:
    print(f"Raw query: {result.raw_query_text}")
    if result.message_matches:
        print("Message matches:")
        for scored_ord in sorted(
            result.message_matches, key=lambda x: x.score, reverse=True
        ):
            score = scored_ord.score
            msg_ord = scored_ord.message_ordinal
            msg = conversation.messages[msg_ord]
            text = " ".join(msg.text_chunks).strip()
            print(
                f"({score:5.1f}){msg_ord:4d}: "
                f"{msg.speaker:>15.15s}: "  # type: ignore  # It's a PodcastMessage
                f"{repr(text)[1:-1]:<150.150s}  "
            )
    if result.knowledge_matches:
        print(f"Knowledge matches ({', '.join(result.knowledge_matches.keys())}):")
        for key, value in sorted(result.knowledge_matches.items()):
            print(f"Type {key} -- {value.term_matches}:")
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
                        f"({score:5.1f}){msg_ord:4d}: "
                        f"{msg.speaker:>15.15s}: "  # type: ignore  # It's a PodcastMessage
                        f"{repr(msg.text_chunks[chunk_ord].strip())[1:-1]:<50.50s}  "
                        f"{summarize_knowledge(sem_ref)}"
                    )


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
                    elif isinstance(value, float) and value.is_integer():
                        value = int(value)
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


if __name__ == "__main__":
    main()
