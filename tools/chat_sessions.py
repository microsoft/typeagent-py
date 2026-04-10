#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Browse VS Code Copilot chat sessions stored on disk.

Usage:
    python tools/chat_sessions.py                  # list all sessions
    python tools/chat_sessions.py -n 5             # list 5 most recent
    python tools/chat_sessions.py --all            # include empty sessions
    python tools/chat_sessions.py <session-id>     # show full conversation
    python tools/chat_sessions.py <number>         # show session by list index
    python tools/chat_sessions.py -s <query>       # search messages for text
"""

import argparse
from collections.abc import Iterator
import contextlib
import datetime
import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import termios
import textwrap
from typing import Any

from colorama import Fore, init, Style

VSCODE_USER_DIR = Path.home() / "Library" / "Application Support" / "Code" / "User"
# Linux: Path.home() / ".config" / "Code" / "User"
# Windows: Path.home() / "AppData" / "Roaming" / "Code" / "User"

# Color settings
use_color = True


def should_use_color(args: argparse.Namespace | None = None) -> bool:
    """Determine if color should be used based on args and environment."""
    # Check explicit command-line flags first
    if args is not None:
        if hasattr(args, "color"):
            if args.color == "always":
                return True
            if args.color == "never":
                return False
    # Check environment variables
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    # Default: use color if output is a TTY
    return sys.stdout.isatty()


def find_session_dirs() -> list[Path]:
    """Find all chatSessions directories across workspaces and global storage."""
    dirs: list[Path] = []
    # Per-workspace sessions
    ws_root = VSCODE_USER_DIR / "workspaceStorage"
    if ws_root.is_dir():
        for entry in ws_root.iterdir():
            chat_dir = entry / "chatSessions"
            if chat_dir.is_dir():
                dirs.append(chat_dir)
    # Global (empty window) sessions
    global_dir = VSCODE_USER_DIR / "globalStorage" / "emptyWindowChatSessions"
    if global_dir.is_dir():
        dirs.append(global_dir)
    return dirs


def get_workspace_name(session_dir: Path) -> str:
    """Try to resolve a workspace name from workspace.json next to chatSessions."""
    ws_json = session_dir.parent / "workspace.json"
    if ws_json.is_file():
        try:
            data = json.loads(ws_json.read_text())
            folder = data.get("folder")
            if folder:
                # "file:///Users/guido/typeagent-py" -> "typeagent-py"
                return folder.rstrip("/").rsplit("/", 1)[-1]
        except Exception:
            pass
    if "emptyWindowChatSessions" in str(session_dir):
        return "(no workspace)"
    return session_dir.parent.name[:12]


type SessionInfo = dict[str, Any]


def _splice(target: list[Any], index: int, items: list[Any]) -> None:
    """Splice items into target at index, extending if needed."""
    while len(target) < index:
        target.append(None)
    target[index : index + len(items)] = items


def parse_jsonl(path: Path) -> SessionInfo | None:
    """Parse a .jsonl chat session file.

    The JSONL format is a delta/patch stream:
      kind 0: session metadata (creationDate, model, etc.)
      kind 1: property update at key-path k
      kind 2: array splice — v is the new items, i is the offset in the
              array identified by k (e.g. ["requests"] or
              ["requests", 0, "response"])
    We reconstruct the final session state by replaying all patches.
    """
    lines = path.read_text(errors="replace").strip().splitlines()
    if not lines:
        return None

    info: SessionInfo = {
        "path": str(path),
        "session_id": path.stem,
        "title": None,
        "creation_date": None,
        "requests": [],
    }

    # Accumulate raw request dicts; patches are applied in order.
    raw_requests: list[dict[str, Any]] = []

    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        kind = record.get("kind")
        k: list[Any] = record.get("k", [])
        v = record.get("v")
        i: int | None = record.get("i")

        if kind == 0 and isinstance(v, dict):
            # Session metadata
            ts = v.get("creationDate")
            if ts:
                info["creation_date"] = ts
            model_info = (
                v.get("inputState", {}).get("selectedModel", {}).get("metadata", {})
            )
            info["model"] = model_info.get("name", "")
            if v.get("customTitle"):
                info["title"] = v["customTitle"]
            # Initial snapshot may include requests already.
            for req in v.get("requests", []):
                if isinstance(req, dict):
                    raw_requests.append(req)

        elif kind == 1:
            # Scalar property update at key-path k
            if "customTitle" in k:
                info["title"] = v
            elif (
                len(k) >= 3
                and k[0] == "requests"
                and isinstance(k[1], int)
                and k[1] < len(raw_requests)
            ):
                # e.g. k: ["requests", 0, "modelState"]
                raw_requests[k[1]][k[2]] = v

        elif kind == 2:
            items = v if isinstance(v, list) else []
            if k == ["requests"]:
                # Full request objects
                if i is not None:
                    _splice(raw_requests, i, items)
                else:
                    raw_requests.extend(items)
            elif (
                len(k) >= 3
                and k[0] == "requests"
                and isinstance(k[1], int)
                and k[1] < len(raw_requests)
            ):
                # Patch a sub-array, e.g. k: ["requests", 0, "response"]
                req_idx = k[1]
                prop = k[2]
                arr = raw_requests[req_idx].get(prop)
                if not isinstance(arr, list):
                    arr = []
                if i is not None:
                    _splice(arr, i, items)
                else:
                    arr.extend(items)
                raw_requests[req_idx][prop] = arr

    # Parse the final reconstructed state of each request.
    for req in raw_requests:
        parsed = _parse_request(req)
        if parsed:
            info["requests"].append(parsed)

    return info


def parse_json(path: Path) -> SessionInfo | None:
    """Parse a .json chat session file."""
    try:
        data = json.loads(path.read_text(errors="replace"))
    except json.JSONDecodeError:
        return None

    info: SessionInfo = {
        "path": str(path),
        "session_id": data.get("sessionId", path.stem),
        "title": None,
        "creation_date": data.get("creationDate"),
        "model": (
            data.get("inputState", {})
            .get("selectedModel", {})
            .get("metadata", {})
            .get("name", "")
        ),
        "requests": [],
    }

    for req in data.get("requests", []):
        parsed = _parse_request(req)
        if parsed:
            info["requests"].append(parsed)

    return info


def _parse_request(req: dict[str, Any]) -> dict[str, Any] | None:
    """Extract user message and assistant response from a request object."""
    if not isinstance(req, dict):
        return None

    user_text = req.get("message", {}).get("text", "")
    timestamp = req.get("timestamp")
    model_id = req.get("modelId", "")

    # modelState.value: 1 = completed, 4 = cancelled
    model_state_raw = req.get("modelState", {})
    model_state = (
        model_state_raw.get("value") if isinstance(model_state_raw, dict) else None
    )

    # Collect assistant response text
    response_parts: list[str] = []
    thinking_parts: list[str] = []
    for part in req.get("response", []):
        if isinstance(part, dict):
            if part.get("kind") == "thinking" and part.get("value"):
                thinking_parts.append(part["value"])
            elif "value" in part and isinstance(part["value"], str) and part["value"]:
                if part.get("kind") not in ("thinking", "toolInvocationSerialized"):
                    response_parts.append(part["value"])

    # Collect tool calls
    tool_calls: list[str] = []
    for part in req.get("response", []):
        if isinstance(part, dict) and part.get("kind") == "toolInvocationSerialized":
            tool_id = part.get("toolId", "")
            tool_data = part.get("toolSpecificData", {})
            if isinstance(tool_data, dict):
                cmd = tool_data.get("commandLine", {})
                if isinstance(cmd, dict):
                    display = cmd.get("forDisplay", cmd.get("original", ""))
                    if display:
                        tool_calls.append(display.strip())
                        continue
            # Non-terminal tools: show a short label
            if tool_id:
                tool_calls.append(f"[{tool_id}]")

    return {
        "user": user_text,
        "assistant": "\n".join(response_parts),
        "thinking": "\n".join(thinking_parts),
        "tools": tool_calls,
        "timestamp": timestamp,
        "model": model_id,
        "model_state": model_state,
    }


def load_all_sessions() -> list[SessionInfo]:
    """Load all sessions from disk."""
    sessions: list[SessionInfo] = []
    for session_dir in find_session_dirs():
        workspace = get_workspace_name(session_dir)
        for f in session_dir.iterdir():
            info = None
            if f.suffix == ".jsonl":
                info = parse_jsonl(f)
            elif f.suffix == ".json":
                info = parse_json(f)
            if info:
                info["workspace"] = workspace
                sessions.append(info)

    # Sort by creation date (newest first)
    sessions.sort(
        key=lambda s: s.get("creation_date") or 0,
        reverse=True,
    )
    return sessions


def format_timestamp(ts: int | None) -> str:
    if not ts:
        return "?"
    # VS Code stores timestamps in milliseconds
    dt = datetime.datetime.fromtimestamp(ts / 1000)
    return dt.strftime("%Y-%m-%d %H:%M")


def get_terminal_width() -> int:
    """Get terminal character width using ioctl."""
    try:
        return struct.unpack("HHHH", fcntl.ioctl(0, termios.TIOCGWINSZ, b"\0" * 8))[1]
    except (OSError, struct.error, IOError):
        return 80  # default fallback


def list_sessions(
    sessions: list[SessionInfo],
    limit: int | None = None,
    show_all: bool = False,
    term_width: int | None = None,
) -> None:
    """Print a summary table of sessions."""
    to_show = sessions[:limit] if limit else sessions
    width = term_width if term_width is not None else 999999
    for i, s in enumerate(to_show):
        reqs = s.get("requests", [])
        n_msgs = len(reqs)
        if n_msgs == 0 and not show_all:
            continue
        title = s.get("title")
        first_msg = ""
        if reqs:
            first_msg = reqs[0].get("user", "")[:80]
        label = title or first_msg or "(empty)"
        # Remove newlines to prevent formatting issues
        label = label.replace("\n", " ").replace("\r", "")
        date_str = format_timestamp(s.get("creation_date"))
        workspace = s.get("workspace", "?")

        if use_color:
            # Colorize the session listing
            line = (
                f"  {Fore.CYAN}{i + 1:3d}{Style.RESET_ALL}. "
                f"[{Fore.YELLOW}{date_str}{Style.RESET_ALL}] "
                f"({Fore.MAGENTA}{workspace}{Style.RESET_ALL}, "
                f"{Fore.GREEN}{n_msgs}{Style.RESET_ALL} msgs) {label}"
            )
        else:
            line = f"  {i + 1:3d}. [{date_str}] ({workspace}, {n_msgs} msgs) {label}"
        # Clip to terminal width
        if len(line) > width:
            line = line[: width - 1]
        print(line)


def show_session(session: SessionInfo) -> None:
    """Print a full conversation."""
    title = session.get("title") or "(untitled)"
    date_str = format_timestamp(session.get("creation_date"))
    workspace = session.get("workspace", "?")
    model = session.get("model", "?")
    session_id = session.get("session_id", "?")

    if use_color:
        print(f"Session: {Fore.CYAN}{title}{Style.RESET_ALL}")
        print(f"  ID:        {Fore.YELLOW}{session_id}{Style.RESET_ALL}")
        print(f"  Date:      {Fore.YELLOW}{date_str}{Style.RESET_ALL}")
        print(f"  Workspace: {Fore.MAGENTA}{workspace}{Style.RESET_ALL}")
        print(f"  Model:     {Fore.GREEN}{model}{Style.RESET_ALL}")
        print(
            f"  Messages:  {Fore.CYAN}{len(session.get('requests', []))}{Style.RESET_ALL}"
        )
    else:
        print(f"Session: {title}")
        print(f"  ID:        {session_id}")
        print(f"  Date:      {date_str}")
        print(f"  Workspace: {workspace}")
        print(f"  Model:     {model}")
        print(f"  Messages:  {len(session.get('requests', []))}")
    print("=" * 72)

    for req in session.get("requests", []):
        ts = format_timestamp(req.get("timestamp"))
        model_id = req.get("model", "")
        model_short = model_id.split("/")[-1] if "/" in model_id else model_id
        model_state = req.get("model_state")

        user_text = req.get("user", "")
        assistant_text = req.get("assistant", "")
        thinking = req.get("thinking", "")
        tools = req.get("tools", [])

        cancelled = model_state == 4
        status = " (cancelled)" if cancelled else ""

        if use_color:
            print(
                f"\n--- [{Fore.YELLOW}{ts}{Style.RESET_ALL}]{Fore.YELLOW}{status}{Style.RESET_ALL} ---"
            )
            print(f"\n{Fore.CYAN}YOU{Style.RESET_ALL}: {user_text}")

            if thinking:
                wrapped = textwrap.fill(
                    thinking, width=70, initial_indent="  ", subsequent_indent="  "
                )
                print(
                    f"\n{Fore.MAGENTA}<thinking>{Style.RESET_ALL}\n{wrapped}\n{Fore.MAGENTA}</thinking>{Style.RESET_ALL}"
                )

            if tools:
                for tool_cmd in tools:
                    if tool_cmd.startswith("["):
                        print(f"\n  {Fore.GREEN}{tool_cmd}{Style.RESET_ALL}")
                    else:
                        print(f"\n  {Fore.GREEN}${Style.RESET_ALL} {tool_cmd}")

            if assistant_text:
                print(
                    f"\n{Fore.CYAN}COPILOT{Style.RESET_ALL} ({Fore.GREEN}{model_short}{Style.RESET_ALL}):\n{assistant_text}"
                )
            elif tools and not cancelled:
                print(
                    f"\n{Fore.CYAN}COPILOT{Style.RESET_ALL} ({Fore.GREEN}{model_short}{Style.RESET_ALL}): ({len(tools)} tool call(s), no text response)"
                )
        else:
            print(f"\n--- [{ts}]{status} ---")
            print(f"\nYOU: {user_text}")

            if thinking:
                wrapped = textwrap.fill(
                    thinking, width=70, initial_indent="  ", subsequent_indent="  "
                )
                print(f"\n<thinking>\n{wrapped}\n</thinking>")

            if tools:
                for tool_cmd in tools:
                    if tool_cmd.startswith("["):
                        print(f"\n  {tool_cmd}")
                    else:
                        print(f"\n  $ {tool_cmd}")

            if assistant_text:
                print(f"\nCOPILOT ({model_short}):\n{assistant_text}")
            elif tools and not cancelled:
                print(
                    f"\nCOPILOT ({model_short}): ({len(tools)} tool call(s), no text response)"
                )

        print()


def search_sessions(
    sessions: list[SessionInfo], query: str, term_width: int | None = None
) -> None:
    """Search all sessions for messages containing query text."""
    query_lower = query.lower()
    hits = 0
    width = term_width if term_width is not None else 999999
    for i, s in enumerate(sessions):
        for req in s.get("requests", []):
            user = req.get("user", "")
            assistant = req.get("assistant", "")
            if query_lower in user.lower() or query_lower in assistant.lower():
                title = s.get("title") or "(untitled)"
                date_str = format_timestamp(s.get("creation_date"))
                workspace = s.get("workspace", "?")
                line1 = f"\n  [{date_str}] ({workspace}) {title}"
                if len(line1) > width:
                    line1 = line1[: width - 1]
                print(line1)
                print(f"  Session #{i + 1}")
                # Show the matching message snippet
                for text, label in [(user, "YOU"), (assistant, "COPILOT")]:
                    idx = text.lower().find(query_lower)
                    if idx >= 0:
                        start = max(0, idx - 40)
                        end = min(len(text), idx + len(query) + 40)
                        snippet = text[start:end].replace("\n", " ")
                        has_start_ellipsis = start > 0
                        has_end_ellipsis = end < len(text)
                        if has_start_ellipsis:
                            snippet = "..." + snippet
                        if has_end_ellipsis:
                            snippet = snippet + "..."

                        if use_color:
                            prefix = f"    {Fore.CYAN}{label}{Style.RESET_ALL}: "
                        else:
                            prefix = f"    {label}: "

                        line2 = prefix + snippet
                        # If line is too long, clip but ensure "..." stays at the end
                        if len(line2) > width:
                            available = width - len(prefix)
                            if available >= 3 and (
                                has_start_ellipsis or has_end_ellipsis
                            ):
                                # Clip content but keep "..." at the end
                                line2 = prefix + snippet[: available - 3] + "..."
                            else:
                                # No room for ellipsis or no ellipsis needed, just clip
                                line2 = line2[: width - 1]
                        print(line2)
                hits += 1
    if hits == 0:
        print(f"No messages found matching '{query}'.")
    else:
        print(f"\n{hits} match(es) found.")


def get_default_pager() -> str:
    """Determine the pager, using the same fallback chain as git."""
    # 1. git config core.pager
    try:
        result = subprocess.run(
            ["git", "config", "--get", "core.pager"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    # 2. GIT_PAGER env
    if pager := os.environ.get("GIT_PAGER"):
        return pager
    # 3. PAGER env
    if pager := os.environ.get("PAGER"):
        return pager
    # 4. less
    return "less"


@contextlib.contextmanager
def smart_pager(pager_cmd: str) -> Iterator[None]:
    """Capture output, then pipe through pager only if it exceeds terminal height."""
    if not sys.stdout.isatty():
        yield
        return

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        yield
    finally:
        sys.stdout = old_stdout

    output = buf.getvalue()
    term_lines = shutil.get_terminal_size().lines
    n_lines = output.count("\n")

    if n_lines < term_lines:
        old_stdout.write(output)
    else:
        env = os.environ.copy()
        # less: quit-if-one-screen, raw-control-chars, no-init
        env.setdefault("LESS", "FRX")
        try:
            proc = subprocess.Popen(
                pager_cmd,
                shell=True,
                stdin=subprocess.PIPE,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            proc.communicate(input=output)
        except (OSError, BrokenPipeError):
            old_stdout.write(output)


def main() -> None:
    global VSCODE_USER_DIR
    if sys.platform == "linux":
        VSCODE_USER_DIR = Path.home() / ".config" / "Code" / "User"
    elif sys.platform == "win32":
        VSCODE_USER_DIR = Path.home() / "AppData" / "Roaming" / "Code" / "User"

    parser = argparse.ArgumentParser(description="Browse VS Code Copilot chat sessions")
    parser.add_argument(
        "session",
        nargs="?",
        help="Session ID or list index to view in full",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=None,
        help="Number of recent sessions to list",
    )
    parser.add_argument(
        "-s",
        "--search",
        type=str,
        default=None,
        help="Search messages for text",
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        default=False,
        help="Include empty sessions in the listing",
    )
    parser.add_argument(
        "--pager",
        type=str,
        default=None,
        help="Pager command (default: from git config, then $GIT_PAGER, $PAGER, less)",
    )
    parser.add_argument(
        "--no-pager",
        action="store_true",
        default=False,
        help="Disable pager",
    )
    parser.add_argument(
        "--color",
        type=str,
        choices=["always", "never", "auto"],
        default="auto",
        help="When to use color (always, never, auto)",
    )
    args = parser.parse_args()

    # Initialize colorama with autoreset disabled so we can use explicit Style.RESET_ALL
    # Use strip=False to preserve ANSI codes even when piped (caller can strip if needed)
    init(autoreset=False, strip=False)
    global use_color
    use_color = should_use_color(args)

    pager_cmd = args.pager if args.pager is not None else get_default_pager()

    sessions = load_all_sessions()
    if not sessions:
        if use_color:
            print(f"{Fore.RED}No chat sessions found.{Style.RESET_ALL}")
        else:
            print("No chat sessions found.")
        return

    use_pager = not args.no_pager
    ctx = smart_pager(pager_cmd) if use_pager else contextlib.nullcontext()

    # Get terminal width for clipping list/search output
    term_width = get_terminal_width() if use_pager else None

    with ctx:
        if args.search:
            search_sessions(sessions, args.search, term_width=term_width)
            return

        if args.session:
            # Try as a list index first
            try:
                idx = int(args.session) - 1
                if 0 <= idx < len(sessions):
                    show_session(sessions[idx])
                    return
            except ValueError:
                pass
            # Try as a session ID
            for s in sessions:
                if s.get("session_id") == args.session:
                    show_session(s)
                    return
            print(f"Session not found: {args.session}")
            return

        n_empty = sum(1 for s in sessions if not s.get("requests"))
        if use_color:
            if n_empty:
                print(
                    f"Found {Fore.CYAN}{len(sessions)}{Style.RESET_ALL} chat session(s), "
                    f"{Fore.YELLOW}{n_empty}{Style.RESET_ALL} empty:\n"
                )
            else:
                print(
                    f"Found {Fore.CYAN}{len(sessions)}{Style.RESET_ALL} chat session(s):\n"
                )
        else:
            if n_empty:
                print(f"Found {len(sessions)} chat session(s), {n_empty} empty:\n")
            else:
                print(f"Found {len(sessions)} chat session(s):\n")
        list_sessions(sessions, args.n, show_all=args.all, term_width=term_width)
        if use_color:
            print(
                f"\nUse: {Fore.CYAN}python {sys.argv[0]} <number>{Style.RESET_ALL} "
                f"to view a session"
            )
        else:
            print(f"\nUse: python {sys.argv[0]} <number> to view a session")


if __name__ == "__main__":
    main()
