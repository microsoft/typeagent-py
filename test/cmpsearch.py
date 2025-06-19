# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

DEFAULT_DATA_FILE = (
    "../../../AISystems-Archive/data/knowpro/test/Episode_53_Answer_results.json"
)

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Parse Q/A data file")
    parser.add_argument(
        "JSONFILE",
        nargs="?",
        type=str,
        default=DEFAULT_DATA_FILE,
        help="Path to the data file",
    )
    args = parser.parse_args()

    # Here you would add the logic to process the data file
    print(f"Processing data from: {args.JSONFILE}")

    with open(args.JSONFILE, "r") as file:
        data = json.load(file)
    assert isinstance(data, list), "Expected a list of Q/A pairs"

    last_q = ""
    for qa_pair in data:
        question = qa_pair.get("question")
        answer = qa_pair.get("answer")
        if not (question and answer) or question == last_q:
            continue
        last_q = question
        print("-" * 40)
        print(f"Answer: {answer}")
        print(f"Question: {question}")
        try:
            result = input("Press Enter to continue: ")
            if result == "q":
                break
        except (EOFError, KeyboardInterrupt):
            print()
            break


main()
