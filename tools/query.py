# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

__version__ = "0.3"

### Imports ###

import argparse
import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import difflib
import json
import os
import re
import shlex
import shutil
import sys
import typing

from colorama import init as colorama_init, Fore
import numpy as np

readline = None
try:
    if os.name != "nt":
        import readline
except ImportError:
    pass

import typechat

from typeagent.aitools import embeddings
from typeagent.aitools import utils

from typeagent.knowpro import answers, answer_response_schema
from typeagent.knowpro import convknowledge
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.knowpro.interfaces import (
    IConversation,
    IMessage,
    ITermToSemanticRefIndex,
    ScoredMessageOrdinal,
    ScoredSemanticRefOrdinal,
    SemanticRef,
    Tag,
    Topic,
)
from typeagent.knowpro import kplib
from typeagent.knowpro import query
from typeagent.knowpro import search, search_query_schema, searchlang
from typeagent.knowpro import serialization

from typeagent.podcasts import podcast

from typeagent.storage.memory.propindex import build_property_index
from typeagent.storage.memory.reltermsindex import build_related_terms_index
from typeagent.storage.sqlite.provider import SqliteStorageProvider
from typeagent.storage.utils import create_storage_provider


### Classes ###


class QuestionAnswerData(typing.TypedDict):
    question: str
    answer: str
    hasNoAnswer: bool
    cmd: str


class RawSearchResultData(typing.TypedDict):
    messageMatches: list[int]
    entityMatches: list[int]
    topicMatches: list[int]
    actionMatches: list[int]


class SearchResultData(typing.TypedDict):
    searchText: str
    searchQueryExpr: dict[str, typing.Any]  # Serialized search_query_schema.SearchQuery
    compiledQueryExpr: list[dict[str, typing.Any]]  # list[search.SearchQueryExpr]
    results: list[RawSearchResultData]


@dataclass
class HistoryEntry:
    """A single Q&A pair in conversation history."""

    question: str
    answer: str
    had_answer: bool


@dataclass
class ConversationHistory:
    """Tracks recent Q&A pairs for context resolution.

    This enables the query engine to resolve pronouns and references
    like "it", "she", or "the first point" by providing recent context
    to the LLM during query translation.
    """

    entries: list[HistoryEntry] = field(default_factory=list)
    max_entries: int = 5

    def add(self, question: str, answer: str, had_answer: bool) -> None:
        """Add a new Q&A pair, removing oldest if at capacity."""
        self.entries.append(HistoryEntry(question, answer, had_answer))
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)

    def clear(self) -> None:
        """Clear all history."""
        self.entries.clear()

    def to_prompt_section(self) -> typechat.PromptSection | None:
        """Format history as a prompt section for LLM.

        Returns None if there's no history to include.
        """
        if not self.entries:
            return None

        lines = [
            "Recent conversation history (use this to resolve pronouns and references like 'it', 'he', 'she', 'the first point', etc.):"
        ]
        for i, entry in enumerate(self.entries, 1):
            lines.append(f"Q{i}: {entry.question}")
            if entry.had_answer:
                answer = (
                    entry.answer[:500] + "..."
                    if len(entry.answer) > 500
                    else entry.answer
                )
                lines.append(f"A{i}: {answer}")
            else:
                lines.append(f"A{i}: [No answer found]")

        return typechat.PromptSection(role="user", content="\n".join(lines))


@dataclass
class ProcessingContext:
    query_context: query.QueryEvalContext
    ar_list: list[QuestionAnswerData]
    sr_list: list[SearchResultData]
    ar_index: dict[str, QuestionAnswerData]
    sr_index: dict[str, SearchResultData]
    debug1: typing.Literal["none", "diff", "full", "skip"]
    debug2: typing.Literal["none", "diff", "full", "skip"]
    debug3: typing.Literal["none", "diff", "full", "nice"]
    debug4: typing.Literal["none", "diff", "full", "nice"]
    embedding_model: embeddings.AsyncEmbeddingModel
    query_translator: typechat.TypeChatJsonTranslator[search_query_schema.SearchQuery]
    answer_translator: typechat.TypeChatJsonTranslator[
        answer_response_schema.AnswerResponse
    ]
    lang_search_options: searchlang.LanguageSearchOptions
    answer_context_options: answers.AnswerContextOptions
    history: ConversationHistory = field(default_factory=ConversationHistory)

    def __repr__(self) -> str:
        parts = []
        parts.append(f"ar_list={len(self.ar_list)}")
        parts.append(f"sr_list={len(self.sr_list)}")
        parts.append(f"ar_index={len(self.ar_index)}")
        parts.append(f"sr_index={len(self.sr_index)}")
        parts.append(f"debug1={self.debug1}")
        parts.append(f"debug2={self.debug2}")
        parts.append(f"debug3={self.debug3}")
        parts.append(f"debug4={self.debug4}")
        parts.append(f"lang_search_options={self.lang_search_options}")
        parts.append(f"answer_context_options={self.answer_context_options}")
        parts.append(f"history={len(self.history.entries)}/{self.history.max_entries}")
        return f"Context({', '.join(parts)})"


CommandHandler = typing.Callable[[ProcessingContext, list[str]], typing.Awaitable[None]]


def _parse_command_args(
    parser: argparse.ArgumentParser, args: list[str]
) -> argparse.Namespace | None:
    """Parse argv for a command, returning None on error instead of exiting."""

    try:
        return parser.parse_args(args)
    except SystemExit:
        return None


async def cmd_help(context: ProcessingContext, args: list[str]) -> None:
    """Show available @-commands. Usage: @help [command]

    Without arguments, lists all commands with short help.
    With one argument, shows long help for that command.
    """

    if not args:
        print("Available commands:")
        for name in sorted(commands):
            func = commands[name]
            doc = func.__doc__ or "(no documentation available)"
            summary = doc.strip().splitlines()[0]
            print(f"  @{name:<10} {summary}")
        print("Type @help <command> for details.")
        return

    cmd_name = args[0].lstrip("@")
    func = commands.get(cmd_name)
    if func is None:
        print(f"Unknown command @{cmd_name!s}. Try @help.")
        return

    doc = func.__doc__
    if not doc:
        print(f"No documentation available for @{cmd_name}.")
        return
    print(doc.strip())


async def cmd_debug(context: ProcessingContext, args: list[str]) -> None:
    """Show or update debug flags. Usage: @debug [--show] [--reset] [FLAG=VALUE ...]

    Flags: debug1-debug4, all
    Values: none, diff, full, skip, nice
    """

    parser = argparse.ArgumentParser(prog="@debug", add_help=True)
    parser.add_argument(
        "--reset", action="store_true", help="Reset to interactive defaults"
    )
    parser.add_argument("items", nargs="*", help="FLAG=VALUE items")
    ns = _parse_command_args(parser, args)
    if ns is None:
        return

    print(
        "Current debug levels: "
        f"debug1={context.debug1}, debug2={context.debug2}, "
        f"debug3={context.debug3}, debug4={context.debug4}"
    )

    if not ns.reset and not ns.items:
        return

    if ns.reset:
        debug_values = {
            "debug1": "none",
            "debug2": "none",
            "debug3": "none",
            "debug4": "nice",  # Not "none", which would suppress the answer
        }
    else:
        # Start out with existing values from context
        debug_values = {
            "debug1": context.debug1,
            "debug2": context.debug2,
            "debug3": context.debug3,
            "debug4": context.debug4,
        }

    if ns.items:
        updates: dict[str, str] = {}
        for item in ns.items:
            pair = item.split("=", maxsplit=1)
            if len(pair) != 2:
                print(f"Invalid item format: {item!r}. Expected FLAG=VALUE.")
                return
            flag = pair[0].lower()
            value = pair[1].lower()
            if flag == "all":
                updates = {
                    "debug1": value,
                    "debug2": value,
                    "debug3": value,
                    "debug4": value,
                }
            else:
                if flag not in debug_values:
                    print(f"Unknown debug flag {flag!r}. Use debug[1-4] or all.")
                    return
                updates[flag] = value

        for flag, value in updates.items():
            allowed = ProcessingContext.__annotations__[flag].__args__  # DRY
            if value not in allowed:
                allowed_list = ", ".join(sorted(allowed))
                print(f"Invalid value {value!r} for {flag}. Allowed: {allowed_list}.")
                return
            debug_values[flag] = value

        if debug_values["debug2"] == "skip":
            debug_values["debug1"] = "skip"

    context.debug1 = debug_values["debug1"]  # type: ignore[assignment]
    context.debug2 = debug_values["debug2"]  # type: ignore[assignment]
    context.debug3 = debug_values["debug3"]  # type: ignore[assignment]
    context.debug4 = debug_values["debug4"]  # type: ignore[assignment]

    print(
        "    New debug levels: "
        f"debug1={context.debug1}, debug2={context.debug2}, "
        f"debug3={context.debug3}, debug4={context.debug4}"
    )


async def cmd_stage(context: ProcessingContext, args: list[str]) -> None:
    """Run only the first N pipeline stages. Usage: @stage COUNT [--diff] QUESTION

    COUNT is 1-4 indicating how many stages to run.
    """

    parser = argparse.ArgumentParser(prog="@stage", add_help=True)
    parser.add_argument("count", type=int, choices=range(1, 5))
    parser.add_argument(
        "--diff", action="store_true", help="Compare with cached results if available"
    )
    parser.add_argument(
        "question",
        nargs=argparse.REMAINDER,
        help="Question to run through the pipeline",
    )
    ns = _parse_command_args(parser, args)
    if ns is None:
        return

    question = " ".join(ns.question).strip()
    if not question:
        print("Error: question text required.")
        return

    debug_context = searchlang.LanguageSearchDebugContext()
    record = context.sr_index.get(question)

    result = await searchlang.search_conversation_with_language(
        context.query_context.conversation,
        context.query_translator,
        question,
        context.lang_search_options,
        debug_context=debug_context,
    )
    if isinstance(result, typechat.Failure):
        print("Stages 1-3 failed:")
        print(Fore.RED + str(result) + Fore.RESET)
        return

    search_results = result.value
    last_stage = ns.count

    actual1 = debug_context.search_query
    if actual1 is None:
        print("Stage 1 produced no search query.")
    else:
        print("Stage 1 search query:")
        if ns.diff and record and "searchQueryExpr" in record:
            expected1 = serialization.deserialize_object(
                search_query_schema.SearchQuery, record["searchQueryExpr"]
            )
            if compare_and_print_diff(expected1, actual1):
                print("Stage 1 matches cached result.")
        utils.pretty_print(actual1, Fore.GREEN, Fore.RESET)
    prsep()

    if last_stage < 2:
        return

    actual2 = debug_context.search_query_expr
    if actual2 is None:
        print("Stage 2 produced no compiled query expression.")
    else:
        print("Stage 2 compiled query expression:")
        if ns.diff and record and "compiledQueryExpr" in record:
            expected2 = serialization.deserialize_object(
                list[search.SearchQueryExpr], record["compiledQueryExpr"]
            )
            if compare_and_print_diff(expected2, actual2):
                print("Stage 2 matches cached result.")
        utils.pretty_print(actual2, Fore.GREEN, Fore.RESET)
    prsep()

    if last_stage < 3:
        return

    print("Stage 3 results:")
    if ns.diff and record and "results" in record:
        expected3 = typing.cast(list[RawSearchResultData], record["results"])
        compare_results(expected3, search_results)
    for sr in search_results:
        await print_result(sr, context.query_context.conversation)
    prsep()

    if last_stage < 4:
        return

    all_answers, combined_answer = await answers.generate_answers(
        context.answer_translator,
        search_results,
        context.query_context.conversation,
        question,
        options=context.answer_context_options,
    )

    if ns.diff:
        record4 = context.ar_index.get(question)
        if record4:
            print("Stage 4 diff with cached answer:")
            expected4 = (record4["answer"], not record4["hasNoAnswer"])
            match combined_answer.type:
                case "NoAnswer":
                    actual4 = (combined_answer.whyNoAnswer or "", False)
                case "Answered":
                    actual4 = (combined_answer.answer or "", True)
            await compare_answers(context, expected4, actual4)
        else:
            print("No cached answer available for stage 4 diff.")

    print("Stage 4 answers:")
    utils.pretty_print(all_answers, Fore.GREEN, Fore.RESET)
    prsep()

    if combined_answer.type == "NoAnswer":
        print(Fore.RED + f"Failure: {combined_answer.whyNoAnswer}" + Fore.RESET)
    else:
        print(Fore.GREEN + f"{combined_answer.answer}" + Fore.RESET)
    prsep()


async def cmd_stats(context: ProcessingContext, args: list[str]) -> None:
    """Print conversation statistics. Usage: @stats"""

    if args:
        print("@stats does not take arguments. Usage: @stats")
        return
    await print_conversation_stats(context.query_context.conversation)


async def cmd_history(context: ProcessingContext, args: list[str]) -> None:
    """Show or manage conversation history. Usage: @history [--clear] [--size N]

    Without arguments, shows current history entries.
    --clear: Clears all history.
    --size N: Sets max history size (0 to disable history).

    History is used to resolve pronouns and references in follow-up questions
    like "it", "he", "she", or "the first point".
    """

    parser = argparse.ArgumentParser(prog="@history", add_help=True)
    parser.add_argument("--clear", action="store_true", help="Clear history")
    parser.add_argument("--size", type=int, help="Set max history size")
    ns = _parse_command_args(parser, args)
    if ns is None:
        return

    if ns.clear:
        context.history.clear()
        print("History cleared.")
        return

    if ns.size is not None:
        context.history.max_entries = ns.size
        while len(context.history.entries) > ns.size:
            context.history.entries.pop(0)
        print(f"History size set to {ns.size}.")
        return

    if not context.history.entries:
        print(
            f"No history yet (max {context.history.max_entries} entries). "
            "Ask some questions first."
        )
        return

    print(
        f"Conversation history "
        f"({len(context.history.entries)}/{context.history.max_entries} entries):"
    )
    for i, entry in enumerate(context.history.entries, 1):
        q_preview = (
            entry.question[:70] + "..." if len(entry.question) > 70 else entry.question
        )
        a_preview = (
            entry.answer[:70] + "..." if len(entry.answer) > 70 else entry.answer
        )
        status = (
            Fore.GREEN + "✓" + Fore.RESET
            if entry.had_answer
            else Fore.RED + "✗" + Fore.RESET
        )
        print(f"  {i}. [{status}] Q: {q_preview}")
        print(f"         A: {a_preview}")


commands: dict[str, CommandHandler] = {
    "help": cmd_help,
    "debug": cmd_debug,
    "stage": cmd_stage,
    "stats": cmd_stats,
    "history": cmd_history,
}


async def handle_at_command(context: ProcessingContext, line: str) -> None:
    """Handle @-commands.

    Input line includes leading '@'.
    """

    try:
        parts = shlex.split(line[1:])
    except ValueError as err:
        print(f"Command parse error: {err}")
        return
    if not parts:
        print("Empty command. Try @help.")
        return

    name, *args = parts
    handler = commands.get(name)
    if handler is None:
        print(f"Unknown command @{name}. Try @help.")
        return

    await handler(context, args)


### Main logic ###


async def main():
    utils.load_dotenv()
    colorama_init(autoreset=True)

    parser = make_arg_parser("TypeAgent Query Tool")
    args = parser.parse_args()
    fill_in_debug_defaults(parser, args)

    if args.logfire:
        utils.setup_logfire()

    settings = ConversationSettings()  # Has no storage provider yet
    settings.storage_provider = await create_storage_provider(
        settings.message_text_index_settings,
        settings.related_term_index_settings,
        args.database,
        podcast.PodcastMessage,
    )
    query_context = await load_podcast_index(
        args.podcast, settings, args.database, args.verbose
    )

    ar_list, ar_index = load_index_file(
        args.qafile, "question", QuestionAnswerData, args.verbose
    )
    sr_list, sr_index = load_index_file(
        args.srfile, "searchText", SearchResultData, args.verbose
    )

    model = convknowledge.create_typechat_model()
    query_translator = utils.create_translator(model, search_query_schema.SearchQuery)
    if args.alt_schema:
        if args.verbose:
            print(f"Substituting alt schema from {args.alt_schema}")
        with open(args.alt_schema) as f:
            query_translator.schema_str = f.read()
    if args.show_schema:
        print(Fore.YELLOW + query_translator.schema_str.rstrip() + Fore.RESET)

    answer_translator = utils.create_translator(
        model, answer_response_schema.AnswerResponse
    )

    context = ProcessingContext(
        query_context,
        ar_list,
        sr_list,
        ar_index,
        sr_index,
        args.debug1,
        args.debug2,
        args.debug3,
        args.debug4,
        settings.embedding_model,
        query_translator,
        answer_translator,
        searchlang.LanguageSearchOptions(
            compile_options=searchlang.LanguageQueryCompileOptions(
                exact_scope=False, verb_scope=True, term_filter=None, apply_scope=True
            ),
            exact_match=False,
            max_message_matches=25,
        ),
        answers.AnswerContextOptions(
            entities_top_k=50, topics_top_k=50, messages_top_k=None, chunking=None
        ),
        ConversationHistory(max_entries=args.history_size),
    )

    if args.verbose:
        utils.pretty_print(context, Fore.BLUE, Fore.RESET)

    if args.query is not None:
        if args.verbose:
            print(Fore.YELLOW + f"Processing single query: {args.query}" + Fore.RESET)
        await process_query(context, args.query)
    elif args.batch:
        if args.verbose:
            print(
                Fore.YELLOW
                + f"Running in batch mode [{args.offset}:{args.offset + args.limit if args.limit else ''}]."
                + Fore.RESET
            )
        await batch_loop(context, args.offset, args.limit, args.skip_counters)
    else:
        if args.verbose:
            print(Fore.YELLOW + "Running in interactive mode." + Fore.RESET)
        await interactive_loop(context)


async def print_conversation_stats(c: IConversation, verbose: bool = True) -> None:
    if not verbose:
        return
    print(f"{await c.messages.size()} messages loaded.")
    print(f"{await c.semantic_refs.size()} semantic refs loaded.")
    print(f"{await c.semantic_ref_index.size()} sem_ref index entries.")
    s = c.secondary_indexes
    if s is None:
        if verbose:
            print("NO SECONDARY INDEXES")
        return

    if s.property_to_semantic_ref_index is None:
        if verbose:
            print("NO PROPERTY TO SEMANTIC REF INDEX")
    else:
        n = await s.property_to_semantic_ref_index.size()
        if verbose:
            print(f"{n} property to semantic ref index entries.")

    if s.timestamp_index is None:
        if verbose:
            print("NO TIMESTAMP INDEX")
    else:
        if verbose:
            print(f"{await s.timestamp_index.size()} timestamp index entries.")

    if s.term_to_related_terms_index is None:
        if verbose:
            print("NO TERM TO RELATED TERMS INDEX")
    else:
        aliases = s.term_to_related_terms_index.aliases
        if verbose:
            print(f"{await aliases.size()} alias entries.")
        f = s.term_to_related_terms_index.fuzzy_index
        if f is None:
            if verbose:
                print("NO FUZZY RELATED TERMS INDEX")
        else:
            if verbose:
                print(f"{await f.size()} term entries.")

    if s.threads is None:
        if verbose:
            print("NO THREADS INDEX")
    else:
        if verbose:
            print(f"{len(s.threads.threads)} threads index entries.")

    if s.message_index is None:
        if verbose:
            print("NO MESSAGE INDEX")
    else:
        if verbose:
            print(f"{await s.message_index.size()} message index entries.")


async def batch_loop(
    context: ProcessingContext, offset: int, limit: int, skip_counters: str
) -> None:
    skips = []
    if skip_counters:
        skips = [int(x) for x in skip_counters.split(",") if x.strip().isdigit()]
    if limit == 0:
        limit = len(context.ar_list) - offset
    sublist = context.ar_list[offset : offset + limit]
    all_scores = []
    for counter, qadata in enumerate(sublist, offset + 1):
        if counter in skips:
            continue
        question = qadata["question"]
        print("-" * 20, counter, question, "-" * 20)
        score = await process_query(context, question)
        if score is not None:
            all_scores.append((score, counter))
    if not all_scores:
        return
    print("=" * 50)
    all_scores.sort(reverse=True)
    good_scores = [(score, counter) for score, counter in all_scores if score >= 0.97]
    bad_scores = [(score, counter) for score, counter in all_scores if score < 0.97]
    for label, pairs in [("Good", good_scores), ("Bad", bad_scores)]:
        print(f"{label} scores ({len(pairs)}):")
        for i in range(0, len(pairs), 10):
            print(
                ", ".join(
                    f"{score:.3f}({counter})" for score, counter in pairs[i : i + 10]
                )
            )


async def interactive_loop(context: ProcessingContext) -> None:
    if not sys.stdin.isatty():
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if line.startswith("@"):
                await handle_at_command(context, line)
            else:
                await process_query(context, line)
        return

    print(f"TypeAgent demo UI {__version__} (type 'q' to exit)")
    if readline:
        try:
            readline.read_history_file(".ui_history")
        except FileNotFoundError:
            pass  # Ignore if history file does not exist.

    try:
        while True:
            try:
                line = input("TypeAgent> ").strip()
            except EOFError:
                print()
                break
            if not line:
                continue
            if line.lower() in ("exit", "quit", "q"):
                if readline:
                    readline.remove_history_item(
                        readline.get_current_history_length() - 1
                    )
                break
            prsep()
            if line.startswith("@"):
                await handle_at_command(context, line)
            else:
                await process_query(context, line)

    finally:
        if readline:
            readline.write_history_file(".ui_history")


### Query processing logic ###


async def process_query(context: ProcessingContext, query_text: str) -> float | None:
    if not query_text.strip():
        return  # Ignore blank query (like interactive mode)
    record = context.sr_index.get(query_text)
    debug_context = searchlang.LanguageSearchDebugContext()
    if context.debug1 == "skip" or context.debug2 == "skip":
        if not record or (
            "searchQueryExpr" not in record or "compiledQueryExpr" not in record
        ):
            print("Can't skip stages 1 or 2, no precomputed outcomes found.")
        else:
            # Skipping stage 2 implies skipping stage 1, and we must supply the
            # precomputed results for both stages.
            debug_context.use_search_query = serialization.deserialize_object(
                search_query_schema.SearchQuery, record["searchQueryExpr"]
            )
            print("Skipping stage 1, substituting precomputed search query.")
            if context.debug2 == "skip":
                debug_context.use_compiled_search_query_exprs = (
                    serialization.deserialize_object(
                        list[search.SearchQueryExpr],
                        record["compiledQueryExpr"],
                    )
                )
                print(
                    "Skipping stage 2, substituting precomputed compiled query expressions."
                )
        prsep()

    history_section = context.history.to_prompt_section()
    if history_section:
        lang_search_options = replace(
            context.lang_search_options,
            model_instructions=[history_section],
        )
    else:
        lang_search_options = context.lang_search_options

    result = await searchlang.search_conversation_with_language(
        context.query_context.conversation,
        context.query_translator,
        query_text,
        lang_search_options,
        debug_context=debug_context,
    )
    if isinstance(result, typechat.Failure):
        print("Stages 1-3 failed:")
        print(Fore.RED + str(result) + Fore.RESET)
        return
    search_results = result.value

    actual1 = debug_context.search_query
    if actual1:
        if context.debug1 == "full":
            print("Stage 1 results:")
            utils.pretty_print(actual1, Fore.GREEN, Fore.RESET)
            prsep()
        elif context.debug1 == "diff":
            if record and "searchQueryExpr" in record:
                print("Stage 1 diff:")
                expected1 = serialization.deserialize_object(
                    search_query_schema.SearchQuery, record["searchQueryExpr"]
                )
                compare_and_print_diff(expected1, actual1)
            else:
                print("Stage 1 diff unavailable")
            prsep()

    actual2 = debug_context.search_query_expr
    if actual2:
        if context.debug2 == "full":
            print("Stage 2 results:")
            utils.pretty_print(actual2, Fore.GREEN, Fore.RESET)
            prsep()
        elif context.debug2 == "diff":
            if record and "compiledQueryExpr" in record:
                print("Stage 2 diff:")
                expected2 = serialization.deserialize_object(
                    list[search.SearchQueryExpr], record["compiledQueryExpr"]
                )
                compare_and_print_diff(expected2, actual2)
            else:
                print("Stage 2 diff unavailable")
            prsep()

    actual3 = search_results
    if context.debug3 == "full":
        print("Stage 3 full results:")
        utils.pretty_print(actual3, Fore.GREEN, Fore.RESET)
        prsep()
    elif context.debug3 == "nice":
        print("Stage 3 nice results:")
        for sr in search_results:
            await print_result(sr, context.query_context.conversation)
        prsep()
    elif context.debug3 == "diff":
        if record and "results" in record:
            print("Stage 3 diff:")
            expected3: list[RawSearchResultData] = record["results"]
            compare_results(expected3, actual3)
        else:
            print("Stage 3 diff unavailable")
        prsep()

    all_answers, combined_answer = await answers.generate_answers(
        context.answer_translator,
        search_results,
        context.query_context.conversation,
        query_text,
        options=context.answer_context_options,
    )

    if context.history.max_entries > 0:
        if combined_answer.type == "Answered":
            context.history.add(query_text, combined_answer.answer or "", True)
        else:
            context.history.add(query_text, combined_answer.whyNoAnswer or "", False)

    if context.debug4 == "full":
        utils.pretty_print(all_answers)
        prsep()
    if context.debug4 in ("full", "nice"):
        if combined_answer.type == "NoAnswer":
            print(Fore.RED + f"Failure: {combined_answer.whyNoAnswer}" + Fore.RESET)
        else:
            print(Fore.GREEN + f"{combined_answer.answer}" + Fore.RESET)
        prsep()
    elif context.debug4 == "diff":
        if query_text in context.ar_index:
            record = context.ar_index[query_text]
            expected4: tuple[str, bool] = (record["answer"], not record["hasNoAnswer"])
            print("Stage 4 diff:")
            match combined_answer.type:
                case "NoAnswer":
                    actual4 = (combined_answer.whyNoAnswer or "", False)
                case "Answered":
                    actual4 = (combined_answer.answer or "", True)
            score = await compare_answers(context, expected4, actual4)
            if actual4[0].startswith("TypeChat failure:"):
                print(Fore.YELLOW + "No answer received" + Fore.RESET)
            else:
                print(f"Score: {score:.3f}; Question: {query_text}")
            return score
        else:
            print("Stage 4 diff unavailable; nice answer:")
            if combined_answer.type == "NoAnswer":
                print(Fore.RED + f"Failure: {combined_answer.whyNoAnswer}" + Fore.RESET)
            else:
                print(Fore.GREEN + f"{combined_answer.answer}" + Fore.RESET)
        prsep()


def prsep():
    print("-" * 50)


### CLI processing ###


def make_arg_parser(description: str) -> argparse.ArgumentParser:
    line_width = min(144, shutil.get_terminal_size().columns)
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=lambda *a, **b: argparse.HelpFormatter(
            *a, **b, max_help_position=35 if line_width >= 100 else 28, width=line_width
        ),
    )

    default_podcast_file = "testdata/Episode_53_AdrianTchaikovsky_index"
    parser.add_argument(
        "--podcast",
        type=str,
        default=default_podcast_file,
        help="Path to the podcast index files (excluding the '_index.json' suffix)",
    )
    default_qafile = "testdata/Episode_53_Answer_results.json"
    explain_qa = "a list of questions and answers to test the full pipeline"
    parser.add_argument(
        "--qafile",
        type=str,
        default=default_qafile,
        help=f"Path to the Answer_results.json file ({explain_qa})",
    )
    default_srfile = "testdata/Episode_53_Search_results.json"
    explain_sr = "a list of intermediate results from stages 1, 2 and 3"
    parser.add_argument(
        "--srfile",
        type=str,
        default=default_srfile,
        help=f"Path to the Search_results.json file ({explain_sr})",
    )
    parser.add_argument(
        "--skip-counters",
        type=str,
        default="",
        help="List of comma-separated questions to skip",
    )
    parser.add_argument(
        "-d",
        "--database",
        type=str,
        default=None,
        help="Path to the SQLite database file (default: in-memory)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Process a single query and exit (equivalent to echo 'query' | query.py)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show verbose startup information and timing logs",
    )
    parser.add_argument(
        "--history-size",
        type=int,
        default=5,
        help="Number of recent Q&A pairs to keep for resolving pronouns/references "
        "(default: 5, 0 to disable)",
    )

    batch = parser.add_argument_group("Batch mode options")
    batch.add_argument(
        "--batch",
        action="store_true",
        help="Run in batch mode, suppressing interactive prompts.",
    )
    batch.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of initial Q/A pairs to skip (default none)",
    )
    batch.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of Q/A pairs to process (default all)",
    )
    batch.add_argument(
        "--start",
        type=int,
        default=0,
        help="Do just this question (similar to --offset START-1 --limit 1)",
    )

    debug = parser.add_argument_group("Debug options")
    debug.add_argument(
        "--debug",
        type=str,
        default=None,
        choices=["none", "diff", "full"],
        help="Default debug level: 'none' for no debug output, 'diff' for diff output, "
        "'full' for full debug output.",
    )
    arg_helper = lambda key: typing.get_args(ProcessingContext.__annotations__[key])
    debug.add_argument(
        "--debug1",
        type=str,
        default=None,
        choices=arg_helper("debug1"),
        help="Debug level override for stage 1: like --debug; or 'skip' to skip stage 1.",
    )
    debug.add_argument(
        "--debug2",
        type=str,
        default=None,
        choices=arg_helper("debug2"),
        help="Debug level override for stage 2: like --debug; or 'skip' to skip stages 1-2.",
    )
    debug.add_argument(
        "--debug3",
        type=str,
        default=None,
        choices=arg_helper("debug3"),
        help="Debug level override for stage 3: like --debug; or 'nice' to print answer only.",
    )
    debug.add_argument(
        "--debug4",
        type=str,
        default=None,
        choices=arg_helper("debug4"),
        help="Debug level override for stage 4: like --debug; or 'nice' to print answer only.",
    )
    debug.add_argument(
        "--alt-schema",
        type=str,
        default=None,
        help="Path to alternate schema file for query translator (modifies stage 1).",
    )
    debug.add_argument(
        "--show-schema",
        action="store_true",
        help="Show the TypeScript schema computed by typechat.",
    )
    debug.add_argument(
        "--logfire",
        action="store_true",
        help="Upload log events to Pydantic's Logfire server",
    )

    return parser


def fill_in_debug_defaults(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    # In batch mode, defaults are diff, diff, diff, diff.
    # In interactive mode they are none, none, none, nice.
    if args.query is not None and args.batch:
        parser.exit(2, "Error: --query cannot be combined with --batch\n")

    if not args.batch:
        if args.start or args.offset or args.limit:
            parser.exit(2, "Error: --start, --offset and --limit require --batch\n")
    else:
        if args.start:
            if args.offset != 0:
                parser.exit(2, "Error: --start and --offset can't be both set\n")
            args.offset = args.start - 1
            if args.limit == 0:
                args.limit = 1
        args.debug = args.debug or "diff"

    args.debug1 = args.debug1 or args.debug or "none"
    args.debug2 = args.debug2 or args.debug or "none"
    args.debug3 = args.debug3 or args.debug or "none"
    args.debug4 = args.debug4 or args.debug or "nice"
    if args.debug2 == "skip":
        args.debug1 = "skip"  # Skipping stage 2 implies skipping stage 1.


### Data loading ###


async def load_podcast_index(
    podcast_file_prefix: str,
    settings: ConversationSettings,
    dbname: str | None,
    verbose: bool = True,
) -> query.QueryEvalContext:
    provider = await settings.get_storage_provider()
    msgs = await provider.get_message_collection()
    if await msgs.size() > 0:  # Sqlite provider with existing non-empty database
        with utils.timelog(f"Reusing database {dbname!r}"):
            conversation = await podcast.Podcast.create(settings)
    else:
        with utils.timelog(f"Loading podcast from {podcast_file_prefix!r}"):
            conversation = await podcast.Podcast.read_from_file(
                podcast_file_prefix, settings, dbname
            )
            if isinstance(provider, SqliteStorageProvider):
                provider.db.commit()

    await print_conversation_stats(conversation, verbose)

    return query.QueryEvalContext(conversation)


def load_index_file[T: Mapping[str, typing.Any]](
    file: str, selector: str, cls: type[T], verbose: bool = True
) -> tuple[list[T], dict[str, T]]:
    # If this crashes, the file is malformed -- go figure it out.
    try:
        with open(file) as f:
            lst: list[T] = json.load(f)
    except FileNotFoundError as err:
        print(Fore.RED + str(err) + Fore.RESET)
        lst = []
    index = {item[selector]: item for item in lst}
    if len(index) != len(lst) and verbose:
        print(f"{len(lst) - len(index)} duplicate items found in {file!r}. ")
    return lst, index


### Debug output ###


async def print_result[TMessage: IMessage, TIndex: ITermToSemanticRefIndex](
    result: search.ConversationSearchResult,
    conversation: IConversation[TMessage, TIndex],
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
            msg = await conversation.messages.get_item(msg_ord)
            assert msg.metadata is not None  # For type checkers
            text = " ".join(msg.text_chunks).strip()
            print(
                f"({score:5.1f}) M={msg_ord:d}: "
                f"{msg.metadata.source!s:>15.15s}: "
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
                    sem_ref = await conversation.semantic_refs.get_item(sem_ref_ord)
                    msg_ord = sem_ref.range.start.message_ordinal
                    msg = await conversation.messages.get_item(msg_ord)
                    print(
                        f"({score:5.1f}) M={msg_ord}: "
                        f"S={summarize_knowledge(sem_ref)}"
                    )


def summarize_knowledge(sem_ref: SemanticRef) -> str:
    """Summarize the knowledge in a SemanticRef."""
    knowledge = sem_ref.knowledge
    if knowledge is None:
        return f"{sem_ref.semantic_ref_ordinal}: <No knowledge>"

    if isinstance(knowledge, kplib.ConcreteEntity):
        entity = knowledge
        res = [f"{entity.name} [{', '.join(entity.type)}]"]
        if entity.facets:
            for facet in entity.facets:
                value = facet.value
                if isinstance(value, kplib.Quantity):
                    value = f"{value.amount} {value.units}"
                elif isinstance(value, float) and value.is_integer():
                    value = int(value)
                res.append(f"<{facet.name}:{value}>")
        return f"{sem_ref.semantic_ref_ordinal}: {' '.join(res)}"
    elif isinstance(knowledge, kplib.Action):
        action = knowledge
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
                if isinstance(param, kplib.ActionParam):
                    res.append(f"<{param.name}:{param.value}>")
                else:
                    res.append(f"<{param}>")
        if action.subject_entity_facet is not None:
            res.append(f"subj_facet={action.subject_entity_facet}")
        return f"{sem_ref.semantic_ref_ordinal}: {' '.join(res)}"
    elif isinstance(knowledge, Topic):
        topic = knowledge
        return f"{sem_ref.semantic_ref_ordinal}: {topic.text!r}"
    elif isinstance(knowledge, Tag):
        tag = knowledge
        return f"{sem_ref.semantic_ref_ordinal}: #{tag.text!r}"
    else:
        return f"{sem_ref.semantic_ref_ordinal}: {sem_ref.knowledge!r}"


def compare_results(
    matches_records: list[RawSearchResultData],
    results: list[search.ConversationSearchResult],
) -> bool:
    if len(results) != len(matches_records):
        print(f"(Result sizes mismatch, {len(results)} != {len(matches_records)})")
        return False
    res = True
    for result, record in zip(results, matches_records):
        if not compare_message_ordinals(
            result.message_matches, record["messageMatches"]
        ):
            res = False
        if not compare_semantic_ref_ordinals(
            (
                []
                if "entity" not in result.knowledge_matches
                else result.knowledge_matches["entity"].semantic_ref_matches
            ),
            record.get("entityMatches", []),
            "entity",
        ):
            res = False
        if not compare_semantic_ref_ordinals(
            (
                []
                if "action" not in result.knowledge_matches
                else result.knowledge_matches["action"].semantic_ref_matches
            ),
            record.get("actionMatches", []),
            "action",
        ):
            res = False
        if not compare_semantic_ref_ordinals(
            (
                []
                if "topic" not in result.knowledge_matches
                else result.knowledge_matches["topic"].semantic_ref_matches
            ),
            record.get("topicMatches", []),
            "topic",
        ):
            res = False
    return res


# Special case: In the Podcast, these messages are all Kevin saying "Yeah",
# so if the difference is limited to these, we consider it a match.
NOISE_MESSAGES = frozenset({42, 46, 52, 68, 70})


def compare_message_ordinals(aa: list[ScoredMessageOrdinal], b: list[int]) -> bool:
    a = [aai.message_ordinal for aai in aa]
    if set(a) ^ set(b) <= NOISE_MESSAGES:
        return True
    print("Message ordinals do not match:")
    utils.list_diff("  Expected:", b, "  Actual:", a, max_items=20)
    return False


def compare_semantic_ref_ordinals(
    aa: list[ScoredSemanticRefOrdinal], b: list[int], label: str
) -> bool:
    a = [aai.semantic_ref_ordinal for aai in aa]
    if sorted(a) == sorted(b):
        return True
    print(f"{label.capitalize()} SemanticRef ordinals do not match:")
    utils.list_diff("  Expected:", b, "  Actual:", a, max_items=20)
    return False


def compare_and_print_diff(a: object, b: object) -> bool:  # True if equal
    """Diff two objects whose repr() is a valid Python expression."""
    if a == b:
        return True
    a_repr = repr(a)
    b_repr = repr(b)
    if a_repr == b_repr:
        return True
    # Shorten floats so slight differences in score etc. don't cause false positives.
    a_repr = re.sub(r"\b\d\.\d\d+", lambda m: f"{float(m.group()):.3f}", a_repr)
    b_repr = re.sub(r"\b\d\.\d\d+", lambda m: f"{float(m.group()):.3f}", b_repr)
    if a_repr == b_repr:
        return True
    a_formatted = utils.format_code(a_repr)
    b_formatted = utils.format_code(b_repr)
    print_diff(a_formatted, b_formatted, n=2)
    return False


async def compare_answers(
    context: ProcessingContext, expected: tuple[str, bool], actual: tuple[str, bool]
) -> float:
    expected_text, expected_success = expected
    actual_text, actual_success = actual

    if expected_success != actual_success:
        print(
            f"Expected success: {Fore.RED}{expected_success}{Fore.RESET}; "
            f"actual: {Fore.GREEN}{actual_success}{Fore.RESET}"
        )
        score = 0.000 if expected_success else 0.001  # 0.001 == Answer not expected

    elif not actual_success:
        print(Fore.GREEN + f"Both failed" + Fore.RESET)
        score = 1.001

    elif expected_text == actual_text:
        print(Fore.GREEN + f"Both equal" + Fore.RESET)
        score = 1.000

    else:
        score = await equality_score(context, expected_text, actual_text)

    if len(expected_text.splitlines()) <= 100 and len(actual_text.splitlines()) <= 100:
        n = 100
    else:
        n = 2
    if score == 1.0:
        print(actual_text)
    else:
        print_diff(expected_text, actual_text, n=n)

    return score


def print_diff(a: str, b: str, n: int) -> None:
    diff = difflib.unified_diff(
        a.splitlines(),
        b.splitlines(),
        fromfile="expected",
        tofile="actual",
        n=n,
    )
    for x in diff:
        if x.startswith("-"):
            print(Fore.RED + x.rstrip("\n") + Fore.RESET)
        elif x.startswith("+"):
            print(Fore.GREEN + x.rstrip("\n") + Fore.RESET)
        else:
            print(x.rstrip("\n"))


async def equality_score(context: ProcessingContext, a: str, b: str) -> float:
    if a == b:
        return 1.0
    if a.lower() == b.lower():
        return 0.999
    embeddings = await context.embedding_model.get_embeddings([a, b])
    assert embeddings.shape[0] == 2, "Expected two embeddings"
    return np.dot(embeddings[0], embeddings[1])


### Run main ###

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, BrokenPipeError):
        print()
        sys.exit(1)
