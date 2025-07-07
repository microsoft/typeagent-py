# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import glob
import re
import statistics
import sys

from colorama import Back, Fore, Style


def main():
    files = sys.argv[1:] or glob.glob("evals/eval-*.txt")
    table = {}  # {file: {counter: score, ...}, ...}
    questions = {}  # {counter: question, ...}

    # Fill table with scoring data from eval files
    for file in files:
        with open(file, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            m = re.match(r"^(?:-+|\*+)\s+(\d+)\s+", line)
            if m:
                counter = int(m.group(1))
                nextline = lines[i + 1]
                mm = re.match(r"^.*; Question:\s+(.*)$", nextline)
                if mm:
                    question = mm.group(1)
                    if counter not in questions:
                        questions[counter] = question
                    elif questions[counter] != question:
                        print(f"File {file} has a different question for {counter}:")
                        print(f"< {questions[counter]}")
                        print(f"> {question}")

        i = lines.index("==================================================\n")
        if i < 0:
            print(f"File {file} does not contain a separator line")
            continue
        lines = lines[i + 1 :]
        text = "".join(lines)
        matches = re.findall(r"\d\.\d\d\d\(\d+\)", text)
        if not matches:
            print(f"File {file} does not contain any scores")
            continue
        # print(len(matches), matches)
        data = {}
        for match in matches:
            m = re.match(r"(\d\.\d\d\d)\((\d+)\)", match)
            assert m
            score = float(m.group(1))
            counter = int(m.group(2))
            data[counter] = score
        assert len(data) == len(matches)
        table[file] = data

    # Print header
    all_files = sorted(table.keys())
    print_header(all_files)

    # Print data
    all_counters = sorted(
        {counter for data in table.values() for counter in data.keys()},
        key=lambda x: table[all_files[0]].get(x, 0.0),
        reverse=True,
    )
    for counter in all_counters:
        print(f"{counter:>3}:", end="")
        for file in all_files:
            score = table[file].get(counter, None)
            if score is None:
                output = "N/A"
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
    print_header(all_files)


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


if __name__ == "__main__":
    main()
