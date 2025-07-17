# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import asyncio
import builtins
from dataclasses import dataclass
import difflib
import json
import re
import sys
from typing import Any, TypedDict

import black
import colorama
import numpy as np
import typechat

from typeagent.aitools import utils
from typeagent.aitools.embeddings import AsyncEmbeddingModel
from typeagent.knowpro.answer_response_schema import AnswerResponse
from typeagent.knowpro import answers
from typeagent.knowpro.importing import ConversationSettings
from typeagent.knowpro.convknowledge import create_typechat_model
from typeagent.knowpro.interfaces import (
    IConversation,
    ScoredMessageOrdinal,
    ScoredSemanticRefOrdinal,
)
from typeagent.knowpro.search import ConversationSearchResult, SearchQueryExpr
from typeagent.knowpro.search_query_schema import SearchQuery
from typeagent.knowpro.serialization import deserialize_object
from typeagent.knowpro import searchlang
from typeagent.podcasts.podcast import Podcast


@dataclass
class Context:
    conversation: IConversation
    query_translator: typechat.TypeChatJsonTranslator[SearchQuery]
    answer_translator: typechat.TypeChatJsonTranslator[AnswerResponse]
    embedding_model: AsyncEmbeddingModel
    lang_search_options: searchlang.LanguageSearchOptions
    answer_options: answers.AnswerContextOptions
    interactive: bool
    sr_index: dict[str, dict[str, Any]]
    use_search_query: bool
    use_compiled_search_query: bool


def main():
    colorama.init()  # So timelog behaves with non-tty stdout.

    # Parse arguments.

    default_qafile = "testdata/Episode_53_Answer_results.json"
    default_srfile = "testdata/Episode_53_Search_results.json"
    default_podcast_file = "testdata/Episode_53_AdrianTchaikovsky_index"

    explanation = "a list of objects with 'question' and 'answer' keys"
    explanation_sr = (
        "a list of objects with 'searchText', 'searchQueryExpr' and 'results' keys"
    )
    parser = argparse.ArgumentParser(
        description="Run queries from` QAFILE and compare answers to expectations"
    )
    parser.add_argument(
        "--qafile",
        type=str,
        default=default_qafile,
        help=f"Path to the Answer_results.json file ({explanation})",
    )
    parser.add_argument(
        "--srfile",
        type=str,
        default=default_srfile,
        help=f"Path to the Search_results.json file ({explanation_sr})",
    )
    parser.add_argument(
        "--use-search-query",
        action="store_true",
        default=False,
        help="Use search query from SRFILE",
    )
    parser.add_argument(
        "--use-compiled-search-query",
        action="store_true",
        default=False,
        help="Use compiled search query from SRFILE",
    )
    parser.add_argument(
        "--podcast",
        type=str,
        default=default_podcast_file,
        help="Path to the podcast index files (excluding the '_index.json' suffix)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of initial Q/A pairs to skip",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of Q/A pairs to print (0 means all)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Do just this question (similar to --offset START-1 --limit 1)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        default=False,
        help="Run in interactive mode, waiting for user input before each question",
    )
    args = parser.parse_args()
    if args.start:
        if args.offset != 0:
            print("Error: --start and --offset can't be both set", file=sys.stderr)
            sys.exit(2)
        args.offset = args.start - 1
        if args.limit == 0:
            args.limit = 1

    # Read evaluation data.

    with open(args.qafile, "r") as file:
        data = json.load(file)
    with open(args.srfile, "r") as file:
        srdata = json.load(file)
        sr_index = {item["searchText"]: item for item in srdata}
    assert isinstance(data, list), "Expected a list of Q/A pairs"
    assert len(data) > 0, "Expected non-empty Q/A data"
    assert all(
        isinstance(qa_pair, dict) and "question" in qa_pair and "answer" in qa_pair
        for qa_pair in data
    ), "Expected each Q/A pair to be a dict with 'question' and 'answer' keys"

    # Read podcast data.

    utils.load_dotenv()
    settings = ConversationSettings()
    with utils.timelog("Loading podcast data"):
        conversation = Podcast.read_from_file(args.podcast, settings)
    assert conversation is not None, f"Failed to load podcast from {file!r}"

    # Create translators.

    model = create_typechat_model()
    query_translator = utils.create_translator(model, SearchQuery)
    answer_translator = utils.create_translator(model, AnswerResponse)

    # Create context.

    context = Context(
        conversation,
        query_translator,
        answer_translator,
        AsyncEmbeddingModel(),
        lang_search_options=searchlang.LanguageSearchOptions(
            compile_options=searchlang.LanguageQueryCompileOptions(
                exact_scope=False, verb_scope=True, term_filter=None, apply_scope=True
            ),
            exact_match=False,
            max_message_matches=25,
        ),
        answer_options=answers.AnswerContextOptions(
            entities_top_k=50, topics_top_k=50, messages_top_k=None, chunking=None
        ),
        interactive=args.interactive,
        sr_index=sr_index,
        use_search_query=args.use_search_query,
        use_compiled_search_query=args.use_compiled_search_query,
    )
    utils.pretty_print(context.lang_search_options)
    utils.pretty_print(context.answer_options)

    # Loop over eval data, skipping duplicate questions
    # (Those differ in 'cmd' value, which we don't support yet.)

    offset = args.offset
    limit = args.limit
    last_q = ""
    counter = 0
    all_scores: list[tuple[float, int]] = []  # [(score, counter), ...]
    for qa_pair in data:
        question = qa_pair.get("question")
        answer = qa_pair.get("answer")
        if question:
            question = question.strip()
        if answer:
            answer = answer.strip()
        if not (question and answer) or question == last_q:
            continue
        counter += 1
        last_q = question

        # Process offset if specified.
        if offset > 0:
            offset -= 1
            continue

        # Wait for user input before continuing.
        print("-" * 25, counter, "-" * 25)
        if context.interactive:
            try:
                input("Press Enter to continue... ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

        # Compare the given answer with the actual answer for the question.
        actual_answer, score = asyncio.run(compare_actual_to_expected(context, qa_pair))
        all_scores.append((score, counter))
        good_enough = score >= 0.97
        print(f"Score: {score:.3f}; Question: {question}", flush=True)
        if context.interactive or not good_enough:
            cmd = qa_pair.get("cmd")
            if cmd and cmd != f'@kpAnswer --query "{question}"':
                print(f"Command: {cmd}")
            if qa_pair.get("hasNoAnswer"):
                answer = f"Failure: {answer}"
            print(f"Expected answer:\n{answer}")
            print("-" * 20)
            print(f"Actual answer:\n{actual_answer}", flush=True)

        # Process limit if specified.
        if limit > 0:
            limit -= 1
            if limit == 0:
                break

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


class RawSearchResult(TypedDict):
    messageMatches: list[int]
    entityMatches: list[int]
    topicMatches: list[int]
    actionMatches: list[int]


async def compare_actual_to_expected(
    context: Context, qa_pair: dict[str, str | None]
) -> tuple[str | None, float]:
    the_answer: str | None = None
    score = 0.0

    question = qa_pair.get("question")
    answer = qa_pair.get("answer")
    failed = qa_pair.get("hasNoAnswer")
    cmd = qa_pair.get("cmd")
    if question:
        question = question.strip()
    if answer:
        answer = answer.strip()
    if not (question and answer):
        return None, score

    if not context.interactive:
        log = lambda *args, **kwds: None  # Disable debug output in non-interactive mode
    else:
        log = print

    log()
    log("=" * 40)
    if cmd:
        log(f"Command: {cmd}")
    log(f"Question: {question}")
    log(f"Answer: {answer}")
    log("-" * 40)

    debug_context = searchlang.LanguageSearchDebugContext()
    if context.use_search_query or context.use_compiled_search_query:
        record = context.sr_index.get(question)
        if record:
            print("Using search query from SRFILE")
            debug_context.use_search_query = deserialize_object(
                SearchQuery, record["searchQueryExpr"]
            )
            if context.use_compiled_search_query:
                print("Using compiled search query from SRFILE")
                debug_context.use_compiled_search_query_exprs = deserialize_object(
                    list[SearchQueryExpr], record["compiledQueryExpr"]
                )

    result = await searchlang.search_conversation_with_language(
        context.conversation,
        context.query_translator,
        question,
        context.lang_search_options,
        debug_context=debug_context,
    )
    log("-" * 40)
    if not isinstance(result, typechat.Success):
        print("Error:", result.message)
    else:
        record = context.sr_index.get(question)
        if record:
            qx = deserialize_object(SearchQuery, record["searchQueryExpr"])
            qc = deserialize_object(
                list[searchlang.SearchQueryExpr], record["compiledQueryExpr"]
            )
            qr: list[RawSearchResult] = record["results"]
            if debug_context.search_query:
                compare_and_print_diff(
                    qx,
                    debug_context.search_query,
                    "Search query from LLM does not match reference.",
                )
            if debug_context.search_query_expr:
                compare_and_print_diff(
                    qc,
                    debug_context.search_query_expr,
                    "Compiled search query expression from LLM does not match reference.",
                )
            compare_results(
                result.value, qr, "Search results from index do not match reference,"
            )
        all_answers, combined_answer = await answers.generate_answers(
            context.answer_translator,
            result.value,
            context.conversation,
            question,
            options=context.answer_options,
        )
        log("-" * 40)
        if combined_answer.type == "NoAnswer":
            # TODO: Compare failure messages.
            if failed:
                score = 1.001  # Magic score so we can tell both are failures.
            the_answer = f"Failure: {combined_answer.whyNoAnswer}"
            log(the_answer)
            log("All answers:")
            if context.interactive:
                utils.pretty_print(all_answers)
        else:
            assert combined_answer.answer is not None, "Expected an answer"
            the_answer = combined_answer.answer
            if failed:
                score = 0.0
            else:
                score = await equality_score(context, answer, the_answer)
            log(the_answer)
            log("Correctness score:", score)
    log("=" * 40)

    return the_answer, score


async def equality_score(context: Context, a: str, b: str) -> float:
    a = a.strip()
    b = b.strip()
    if a == b:
        return 1.0
    if a.lower() == b.lower():
        return 0.999
    embeddings = await context.embedding_model.get_embeddings([a, b])
    assert embeddings.shape[0] == 2, "Expected two embeddings"
    return np.dot(embeddings[0], embeddings[1])


def compare_results(
    results: list[ConversationSearchResult],
    matches_records: list[RawSearchResult],
    message: str,
) -> bool:
    if len(results) != len(matches_records):
        print(
            f"Warning: {message} "
            f"(Result sizes mismatch, {len(results)} != {len(matches_records)})"
        )
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


def compare_and_print_diff(
    a: object, b: object, message: str
) -> bool:  # True if unequal
    """Diff two objects whose repr() is a valid Python expression."""
    if a == b:
        return False
    a_repr = builtins.repr(a)
    b_repr = builtins.repr(b)
    if a_repr == b_repr:
        return False
    # Shorten floats so slight differences in score etc. don't cause false positives.
    a_repr = re.sub(r"\b\d\.\d\d+", lambda m: f"{float(m.group()):.3f}", a_repr)
    b_repr = re.sub(r"\b\d\.\d\d+", lambda m: f"{float(m.group()):.3f}", b_repr)
    if a_repr == b_repr:
        return False
    print("Warning:", message)
    a_formatted = black.format_str(a_repr, mode=black.FileMode())
    b_formatted = black.format_str(b_repr, mode=black.FileMode())
    diff = difflib.unified_diff(
        a_formatted.splitlines(True),
        b_formatted.splitlines(True),
        fromfile="expected",
        tofile="actual",
        n=2,
    )
    for x in diff:
        print(x, end="")
    return True


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        sys.exit(1)
