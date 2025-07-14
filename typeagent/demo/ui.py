# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

__version__ = "0.1"

import argparse
import asyncio
import io
import json
import re
import sys
import traceback
from typing import cast

try:
    import readline
except ImportError:
    readline = None

import colorama
import typechat

from ..aitools.utils import create_translator, load_dotenv, pretty_print, timelog
from ..knowpro.answer_response_schema import AnswerResponse
from ..knowpro.answers import AnswerContextOptions, generate_answers
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
    LanguageQueryCompileOptions,
    LanguageQueryExpr,
    LanguageSearchDebugContext,
    LanguageSearchOptions,
    search_conversation_with_language,
)
from ..knowpro.search_query_schema import SearchQuery
from ..knowpro.serialization import deserialize_object
from ..podcasts.podcast import Podcast


def main() -> None:
    parser = argparse.ArgumentParser(description="TypeAgent demo UI.")
    parser.add_argument(
        "--podcast",
        type=str,
        default="testdata/Episode_53_AdrianTchaikovsky_index",
        help="Path to the podcast file, less _data.json suffix.",
    )
    parser.add_argument(
        "--alt-schema",
        type=str,
        default=None,
        help="Path to alternate schema file for query translator.",
    )
    parser.add_argument(
        "--search-query-index",
        type=str,
        default=None,
        help="Path to a JSON file containing a search query index mapping.",
    )

    args = parser.parse_args()

    colorama.init(autoreset=True)
    load_dotenv()

    model = create_typechat_model()
    query_translator = create_translator(model, SearchQuery)
    answer_translator = create_translator(model, AnswerResponse)

    if args.alt_schema:
        print(f"Substituting alt schema from {args.alt_schema}")
        with open(args.alt_schema) as f:
            query_translator.schema_str = f.read()

    print(colorama.Fore.YELLOW + query_translator.schema_str.rstrip())

    search_query_index: dict[str, dict[str, object]] = {}
    if args.search_query_index:
        print(f"Loading search query index from {args.search_query_index}")
        with open(args.search_query_index) as f:
            # TODO: Add type checks and annotations.
            srdata = json.load(f)
            search_query_index = {item["searchText"]: item for item in srdata}

    lang_search_options = LanguageSearchOptions(
        exact_match=False,
        max_message_matches=25,
        max_chars_in_budget=None,
        compile_options=LanguageQueryCompileOptions(
            exact_scope=False, apply_scope=True
        ),
    )
    pretty_print(lang_search_options, colorama.Fore.CYAN)
    answer_context_options = AnswerContextOptions(
        entities_top_k=50,
        topics_top_k=50,
    )
    pretty_print(answer_context_options, colorama.Fore.CYAN)

    file = args.podcast
    settings = ConversationSettings()
    with timelog(f"load podcast from {file!r}"):
        pod = Podcast.read_from_file(file, settings)
    assert pod is not None, f"Failed to load podcast from {file!r}"
    context = QueryEvalContext(pod)

    if sys.stdin.isatty():
        print(f"TypeAgent demo UI {__version__} (type 'q' to exit)")
        if readline:
            try:
                readline.read_history_file(".ui_history")
            except FileNotFoundError:
                pass  # Ignore if history file does not exist.

    try:
        process_inputs(
            query_translator,
            answer_translator,
            lang_search_options,
            answer_context_options,
            context,
            cast(io.TextIOWrapper, sys.stdin),
            search_query_index,
        )

    except KeyboardInterrupt:
        print()

    finally:
        if readline and sys.stdin.isatty():
            readline.write_history_file(".ui_history")


def process_inputs[TMessage: IMessage, TIndex: ITermToSemanticRefIndex](
    query_translator: typechat.TypeChatJsonTranslator[SearchQuery],
    answer_translator: typechat.TypeChatJsonTranslator[AnswerResponse],
    lang_search_options: LanguageSearchOptions,
    answer_context_options: AnswerContextOptions,
    context: QueryEvalContext[TMessage, TIndex],
    stream: io.TextIOWrapper,
    search_query_index: dict[str, dict[str, object]],
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
                print("-" * 50)
                with timelog("Query processing"):
                    use_search_query: SearchQuery | None = None
                    if search_query_index and query_text in search_query_index:
                        print(colorama.Fore.YELLOW + "Using pre-computed SearchQuery")
                        use_search_query = deserialize_object(
                            SearchQuery,
                            search_query_index[query_text]["searchQueryExpr"],
                        )
                    asyncio.run(
                        wrap_process_query(
                            query_text,
                            context.conversation,
                            query_translator,
                            answer_translator,
                            lang_search_options,
                            answer_context_options,
                            use_search_query,
                        )
                    )
                print("-" * 50)


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
    lang_search_options: LanguageSearchOptions,
    answer_context_options: AnswerContextOptions,
    use_search_query: SearchQuery | None = None,
) -> None:
    """Wrap the process_query function to handle exceptions."""
    try:
        await process_query(
            query_text,
            conversation,
            query_translator,
            answer_translator,
            lang_search_options,
            answer_context_options,
            use_search_query,
        )
    except Exception as exc:
        traceback.print_exc()
        # traceback.print_exception(type(exc), exc, exc.__traceback__.tb_next)


async def process_query[TMessage: IMessage, TIndex: ITermToSemanticRefIndex](
    orig_query_text: str,
    conversation: IConversation[TMessage, TIndex],
    query_translator: typechat.TypeChatJsonTranslator[SearchQuery],
    answer_translator: typechat.TypeChatJsonTranslator[AnswerResponse],
    lang_search_options: LanguageSearchOptions,
    answer_context_options: AnswerContextOptions,
    use_search_query: SearchQuery | None = None,
) -> None:
    debug_context = LanguageSearchDebugContext(use_search_query=use_search_query)
    result = await search_conversation_with_language(
        conversation,
        query_translator,
        orig_query_text,
        options=lang_search_options,
        debug_context=debug_context,
    )
    if debug_context:
        if debug_context.search_query:
            pretty_print(debug_context.search_query)
            print("-" * 50)
        if debug_context.search_query_expr:
            pretty_print(debug_context.search_query_expr)
            print("-" * 50)
    if not isinstance(result, typechat.Success):
        print(f"Error searching conversation: {result.message}")
        return
    search_results = result.value
    for sr in search_results:
        print_result(sr, conversation)
        print("-" * 50)
    all_answers, combined_answer = await generate_answers(
        answer_translator,
        search_results,
        conversation,
        orig_query_text,
        options=answer_context_options,
    )
    pretty_print(all_answers)
    print("-" * 50)
    if combined_answer.type == "NoAnswer":
        print(colorama.Fore.RED + f"Failure: {combined_answer.whyNoAnswer}")
    else:
        print(colorama.Fore.GREEN + f"{combined_answer.answer}")
    print("-" * 50)


def print_result[TMessage: IMessage, TIndex: ITermToSemanticRefIndex](
    result: ConversationSearchResult, conversation: IConversation[TMessage, TIndex]
) -> None:
    print(
        f"Raw query: {result.raw_query_text};",
        f"{len(result.message_matches)} message matches,",
        f"{len(result.knowledge_matches)} knowledge matches",
    )
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
                f"({score:5.1f}) M={msg_ord:d}: "
                f"{msg.speaker:>15.15s}: "  # type: ignore  # It's a PodcastMessage
                f"{repr(text)[1:-1]:<150.150s}  "
            )
    if result.knowledge_matches:
        print(f"Knowledge matches ({', '.join(sorted(result.knowledge_matches))}):")
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
                        f"({score:5.1f}) M={msg_ord}: "
                        # f"{msg.speaker:>15.15s}: "  # type: ignore  # It's a PodcastMessage
                        # f"{repr(msg.text_chunks[chunk_ord].strip())[1:-1]:<50.50s}  "
                        f"S={summarize_knowledge(sem_ref)}"
                    )


def summarize_knowledge(sem_ref: SemanticRef) -> str:
    """Summarize the knowledge in a SemanticRef."""
    knowledge = sem_ref.knowledge
    if knowledge is None:
        return f"{sem_ref.semantic_ref_ordinal}: <No knowledge>"
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
            return f"{sem_ref.semantic_ref_ordinal}: {' '.join(res)}"
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
            return f"{sem_ref.semantic_ref_ordinal}: {' '.join(res)}"
        case "topic":
            topic = knowledge
            assert isinstance(topic, Topic)
            return f"{sem_ref.semantic_ref_ordinal}: {topic.text!r}]"
        case "tag":
            tag = knowledge
            assert isinstance(tag, str)
            return f"{sem_ref.semantic_ref_ordinal}: #{tag!r}"
        case _:
            return f"{sem_ref.semantic_ref_ordinal}: {sem_ref.knowledge!r}"


if __name__ == "__main__":
    main()
