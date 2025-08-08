# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import glob
import os
import re
import statistics
import sys

from colorama import init as colorama_init, Back, Fore, Style


def main():
    parser = argparse.ArgumentParser(
        description="Compare evaluation results from multiple files."
    )
    parser.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default="auto",
        help="Control color output. Default 'auto' uses colors if stdout is a terminal.",
    )
    parser.add_argument(
        "files",
        nargs="*",
    )
    args = parser.parse_args()

    # Initialize colorama according to --color.
    match args.color:
        case "auto":
            colorama_init(strip=not sys.stdout.isatty())
        case "always":
            colorama_init(strip=False)
        case "never":
            colorama_init(strip=True)
        case _:
            raise ValueError(f"Invalid color option: {args.color}")

    files = args.files or sorted(glob.glob("evals/eval-*.txt"))
    table = {}  # {file: {counter: score, ...}, ...}
    questions = {}  # {counter: question, ...}

    # Fill table with scoring data from eval files
    for file in files:
        with open(file, "r") as f:
            lines = f.readlines()

        scores = {}
        counter = None
        for i, line in enumerate(lines):
            if m := re.match(r"^(?:-+|\*+)\s+(\d+)\s+", line):
                counter = int(m.group(1))
            elif m := re.match(r"^Score:\s+([\d.]+); Question:\s+(.*)$", line):
                score = float(m.group(1))
                scores[counter] = score
                question = m.group(2)
                if counter not in questions:
                    questions[counter] = question
                elif questions[counter] != question:
                    print(f"File {file} has a different question for {counter}:")
                    print(f"< {questions[counter]}")
                    print(f"> {question}")

        table[file] = scores

    # Print header
    all_files = list(table.keys())
    print_header(all_files)

    # Print data
    all_counters = sorted(
        {counter for data in table.values() for counter in data.keys()},
        key=lambda x: statistics.mean(table[file].get(x, 0.0) for file in all_files),
        reverse=True,
    )
    for counter in all_counters:
        print(f"{counter:>3}:", end="")
        for file in all_files:
            score = table[file].get(counter, None)
            if score is None:
                output = Fore.YELLOW + "  N/A " + Fore.RESET
                output = Style.BRIGHT + output + Style.RESET_ALL
            else:
                output = f"{score:.3f}"
                output = f"{output:>6}"
                if score >= 0.97:
                    output = Fore.GREEN + output + Fore.RESET
                    if score >= 0.999:
                        output = Style.BRIGHT + output + Style.RESET_ALL
                elif score >= 0.9:
                    output = Fore.BLUE + output + Fore.RESET
                else:
                    output = Fore.RED + output + Fore.RESET
                    if score == 0.0:
                        output = Style.BRIGHT + output + Style.RESET_ALL
            print(output, end="")
        print(f" {questions.get(counter)}")

    # Print header again
    print_footer(all_files)


def print_header(all_files):
    print("    ", end="")
    for i, file in enumerate(all_files):
        base = os.path.basename(file)
        m = re.match(r"eval-(\d+\w*).*\.txt", base)
        if m:
            label = m.group(1)
        else:
            label = "--"
        print(f"{label:>6}", end="")
    print()


def print_footer(all_files):
    for i, file in reversed(list(enumerate(all_files))):
        print("     |" * i + "     " + os.path.basename(file))


if __name__ == "__main__":
    main()
