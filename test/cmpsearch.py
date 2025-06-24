# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import asyncio
import builtins
from dataclasses import dataclass
import json

import numpy as np
import typechat

from typeagent.knowpro.importing import ConversationSettings
from typeagent.aitools import auth
from typeagent.aitools.embeddings import AsyncEmbeddingModel
from typeagent.demo import ui  # TODO: Move what we import to a more appropriate place
from typeagent.knowpro.convknowledge import create_typechat_model
from typeagent.knowpro.interfaces import IConversation
from typeagent.knowpro.search_query_schema import SearchQuery
from typeagent.knowpro import searchlang
from typeagent.podcasts.podcast import Podcast


@dataclass
class Context:
    conversation: IConversation
    query_translator: typechat.TypeChatJsonTranslator[SearchQuery]
    embedding_model: AsyncEmbeddingModel
    options: None
    interactive: bool


def main():
    # Parse arguments.

    default_qafile = (
        "../../../AISystems-Archive/data/knowpro/test/Episode_53_Answer_results.json"
    )
    default_podcast_file = "testdata/Episode_53_AdrianTchaikovsky_index"

    explanation = "a list of objects with 'question' and 'answer' keys"
    parser = argparse.ArgumentParser(description="Parse Q/A data file")
    parser.add_argument(
        "--qafile",
        type=str,
        default=default_qafile,
        help=f"Path to the data file ({explanation})",
    )
    parser.add_argument(
        "--podcast",
        type=str,
        default=default_podcast_file,
        help="Path to the podcast index files (excluding the '_index.json' suffix)",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Number of initial Q/A pairs to skip (for debugging purposes)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        default=False,
        help="Run in interactive mode, waiting for user input before each question",
    )
    args = parser.parse_args()

    # Read evaluation data.

    with open(args.qafile, "r") as file:
        data = json.load(file)
    assert isinstance(data, list), "Expected a list of Q/A pairs"
    assert len(data) > 0, "Expected non-empty Q/A data"
    assert all(
        isinstance(qa_pair, dict) and "question" in qa_pair and "answer" in qa_pair
        for qa_pair in data
    ), "Expected each Q/A pair to be a dict with 'question' and 'answer' keys"

    # Read podcast data.

    auth.load_dotenv()
    settings = ConversationSettings()
    with ui.timelog("Loading podcast data"):
        conversation = Podcast.read_from_file(args.podcast, settings)
    assert conversation is not None, f"Failed to load podcast from {file!r}"

    # Create translator.

    model = create_typechat_model()
    query_translator = ui.create_translator(model, SearchQuery)

    # Create context.

    context = Context(
        conversation,
        query_translator,
        AsyncEmbeddingModel(),
        options=None,  # TODO: Set options if needed
        interactive=args.interactive,
    )

    # Loop over eval data, skipping duplicate questions
    # (Those differ in 'cmd' value, which we don't support yet.)

    skip = args.skip
    last_q = ""
    counter = 0
    for qa_pair in data:
        counter += 1
        question = qa_pair.get("question")
        answer = qa_pair.get("answer")
        if not (question and answer) or question == last_q:
            continue
        last_q = question
        if skip > 0:
            skip -= 1
            continue

        # Wait for user input before continuing.
        if context.interactive:
            try:
                input("Press Enter to continue... ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

        # Compare the given answer with the actual answer for the question.
        actual_answer, score = asyncio.run(compare(context, qa_pair))
        print("-" * 25, counter, "-" * 25)
        if not context.interactive and score < 0.97:
            print(f"Question: {question}")
            print(f"Expected answer:\n{answer}")
            print("-" * 20)
            print(f"Actual answer:\n{actual_answer}")
            print(f"Score: {score:.3f}")
        else:
            print(f"Score: {score:.3f} (question: {question})")


async def compare(
    context: Context, qa_pair: dict[str, str | None]
) -> tuple[str | None, float]:
    the_answer: str | None = None
    score = 0.0

    question = qa_pair.get("question")
    answer = qa_pair.get("answer")
    cmd = qa_pair.get("cmd")
    if not (question and answer):
        return None, score

    if not context.interactive:
        print = lambda *args, **kwds: None  # Disable printing in non-interactive mode
    else:
        print = builtins.print

    print()
    print("=" * 40)
    if cmd:
        print(f"Command: {cmd}")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print("-" * 40)

    result = await searchlang.search_conversation_with_language(
        context.conversation,
        context.query_translator,
        question,
        context.options,
    )
    print("-" * 40)
    if not isinstance(result, typechat.Success):
        print("Error:", result.message)
    else:
        all_answers, combined_answer = await ui.generate_answers(
            result.value, context.conversation, question
        )
        print("-" * 40)
        if combined_answer.type == "NoAnswer":
            print("Failure:", combined_answer.whyNoAnswer)
            print("All answers:")
            if context.interactive:
                ui.pretty_print(all_answers)
        else:
            assert combined_answer.answer is not None, "Expected an answer"
            the_answer = combined_answer.answer
            score = await equality_score(context, answer, the_answer)
            print(the_answer)
            print("Correctness score:", score)
    print("=" * 40)

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


if __name__ == "__main__":
    main()
