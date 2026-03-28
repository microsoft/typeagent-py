# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Print the Date header of .eml files as a hex POSIX timestamp."""

import email
import email.utils
import sys


def main() -> None:
    for path in sys.argv[1:]:
        with open(path, "rb") as f:
            msg = email.message_from_binary_file(f)
        date_str = msg["Date"] or ""
        parsed = email.utils.parsedate_tz(date_str)
        if parsed is None:
            hex_ts = "????????"
        else:
            hex_ts = f"{int(email.utils.mktime_tz(parsed)) * 1000 << 20:016x}"
        print(f"{path}\t{date_str:<40}\t{hex_ts}")


if __name__ == "__main__":
    main()
