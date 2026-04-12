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
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any

from colorama import Fore, init, Style


def _detect_vscode_user_dir() -> list[Path]:
    """Detect VS Code user directories for current environment.

    Returns a list of directories to search, in priority order:
    1. VSCode Server (if .vscode-server exists)
    2. Native VS Code installation for the platform
    3. Windows VS Code via WSL mount (if on WSL)
    """
    dirs: list[Path] = []

    # VSCode Server (for remote SSH, WSL, containers, etc.)
    vscode_server = Path.home() / ".vscode-server" / "data" / "User"
    if vscode_server.is_dir():
        dirs.append(vscode_server)

    # Platform-specific native VS Code
    if sys.platform == "linux":
        dirs.append(Path.home() / ".config" / "Code" / "User")
    elif sys.platform == "win32":
        dirs.append(Path.home() / "AppData" / "Roaming" / "Code" / "User")
    elif sys.platform == "darwin":
        dirs.append(Path.home() / "Library" / "Application Support" / "Code" / "User")

    # Windows via WSL mount (if running on WSL)
    if sys.platform == "linux" and Path("/mnt/c").exists():
        win_user = Path("/mnt/c/Users")
        if win_user.is_dir():
            # Try to find the current user's home in Windows
            for user_dir in win_user.iterdir():
                if user_dir.is_dir():
                    vscode_win = user_dir / "AppData" / "Roaming" / "Code" / "User"
                    if vscode_win.is_dir():
                        dirs.append(vscode_win)
                        break

    return dirs


# Color settings
use_color = True

# Regex to match ANSI escape sequences
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def visible_len(text: str) -> int:
    """Return the visible length of text, excluding ANSI escape sequences."""
    return len(ANSI_ESCAPE.sub("", text))


def highlight_query(text: str, query: str) -> str:
    """Highlight all occurrences of query in text (case-insensitive)."""
    if not use_color:
        return text
    # Replace all occurrences of query with highlighted version (case-insensitive)
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    highlighted = pattern.sub(
        lambda m: f"{Fore.RED}{m.group()}{Style.RESET_ALL}",
        text,
    )
    return highlighted


def clip_to_visible_length(text: str, target_length: int) -> str:
    """Clip text to a target visible length, accounting for ANSI escape codes.

    Splits the string into ANSI-code tokens and plain-character tokens, then
    reconstructs from the left until the visible character count reaches the
    target.  This avoids splitting in the middle of an escape sequence.
    """
    # Tokenize: alternate between non-ANSI runs and ANSI sequences
    tokens = ANSI_ESCAPE.split(text)
    ansi_codes = ANSI_ESCAPE.findall(text)

    result = []
    visible = 0
    # Interleave: tokens[0], ansi_codes[0], tokens[1], ansi_codes[1], ...
    for i, plain in enumerate(tokens):
        remaining = target_length - visible
        if len(plain) <= remaining:
            result.append(plain)
            visible += len(plain)
        else:
            result.append(plain[:remaining])
            visible += remaining
            # Consume any pending ANSI codes to reset state, then stop
            if i < len(ansi_codes):
                result.append(ansi_codes[i])
            break
        if i < len(ansi_codes):
            result.append(ansi_codes[i])  # ANSI codes don't count as visible
    return "".join(result)


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
    """Find all chat session directories across workspaces and global storage.

    Searches for both old format (chatSessions) and new format (GitHub.copilot-chat).
    """
    dirs: list[Path] = []
    search_dirs = _detect_vscode_user_dir()

    for base_dir in search_dirs:
        if not base_dir.is_dir():
            continue

        # Per-workspace sessions (old format)
        ws_root = base_dir / "workspaceStorage"
        if ws_root.is_dir():
            for entry in ws_root.iterdir():
                if not entry.is_dir():
                    continue
                # Old format: chatSessions
                chat_dir = entry / "chatSessions"
                if chat_dir.is_dir():
                    dirs.append(chat_dir)
                # New format: GitHub.copilot-chat
                copilot_dir = entry / "GitHub.copilot-chat"
                if copilot_dir.is_dir():
                    dirs.append(copilot_dir)

        # Global (empty window) sessions (old format)
        global_dir = base_dir / "globalStorage" / "emptyWindowChatSessions"
        if global_dir.is_dir():
            dirs.append(global_dir)

    return dirs


def get_workspace_name(session_dir: Path) -> str:
    """Try to resolve a workspace name from workspace.json next to session dir."""
    ws_json = session_dir.parent / "workspace.json"
    if ws_json.is_file():
        try:
            data = json.loads(ws_json.read_text())
            folder = data.get("folder")
            if folder:
                # "file:///Users/guido/typeagent-py" -> "typeagent-py"
                return folder.rstrip("/").rsplit("/", 1)[-1]
        except (json.JSONDecodeError, OSError):
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


_RE_CUSTOM_TITLE_JSONL = re.compile(
    r'"customTitle"\s*]\s*,\s*"v"\s*:\s*"((?:[^"\\]|\\.)*)"'
)


def parse_jsonl_metadata(path: Path) -> SessionInfo | None:
    """Fast metadata extraction from a .jsonl chat session file.

    Reads the first line (kind-0 session metadata snapshot) and a few KB
    after it (for customTitle patches and first user message) to avoid
    reading multi-MB files fully.
    Falls back to full parse if the first line isn't a valid kind-0 record.
    """
    size = path.stat().st_size
    if size == 0:
        return None

    with open(path, "rb") as fh:
        first_line_bytes = fh.readline()
        line1_end = fh.tell()
        # Read a few KB more for customTitle patches (kind-1, lines 2-3)
        # and possibly the first user message.
        extra = fh.read(min(size - line1_end, 4096)).decode("utf-8", errors="replace")

    first_line = first_line_bytes.decode("utf-8", errors="replace")
    if not first_line.strip():
        return None
    try:
        record = json.loads(first_line)
    except json.JSONDecodeError:
        return None

    if record.get("kind") != 0:
        return parse_jsonl(path)  # fall back

    info: SessionInfo = {
        "path": str(path),
        "session_id": path.stem,
        "title": None,
        "creation_date": None,
        "size": size,
        "requests": [],
    }

    v = record.get("v", {})
    if isinstance(v, dict):
        if ts := v.get("creationDate"):
            info["creation_date"] = ts
        model_info = (
            v.get("inputState", {}).get("selectedModel", {}).get("metadata", {})
        )
        info["model"] = model_info.get("name", "")
        if v.get("customTitle"):
            info["title"] = v["customTitle"]
        # First user message from initial snapshot
        reqs = v.get("requests", [])
        if reqs and isinstance(reqs[0], dict):
            first_user = reqs[0].get("message", {}).get("text", "")
            if first_user:
                info["requests"].append({"user": first_user})

    # Look for customTitle patches in the extra bytes after line 1.
    # Kind-1 patches for customTitle are small lines near the start of the file.
    if not info.get("title") and extra:
        m = _RE_CUSTOM_TITLE_JSONL.search(extra)
        if m:
            info["title"] = m.group(1).replace("\\n", "\n").replace('\\"', '"')

    # Look for first user message in extra bytes (unlikely to be there since
    # "message" is deep in the request patch line, but try anyway).
    if not info["requests"] and extra:
        m = _RE_FIRST_MSG.search(extra)
        if m:
            first_user = m.group(1).replace("\\n", "\n").replace('\\"', '"')
            info["requests"].append({"user": first_user})

    # If we still have no requests but the extra bytes contain a kind-2
    # request splice, the file has requests (we just can't extract the text
    # from a small buffer).
    if not info["requests"] and '"requests"' in extra:
        info["requests"].append({"user": ""})

    return info


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
        "size": path.stat().st_size,
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


# Regexes for fast tail-of-file metadata extraction from JSON files.
_RE_CREATION_DATE = re.compile(r'"creationDate"\s*:\s*(\d+)')
_RE_CUSTOM_TITLE = re.compile(r'"customTitle"\s*:\s*"((?:[^"\\]|\\.)*)"')
_RE_SESSION_ID = re.compile(r'"sessionId"\s*:\s*"([^"]+)"')
# Match "message":{ ... "text": "..." } — the ... allows for "parts" or other
# keys that may appear before "text" in old JSON format (version 3).
# Capture is capped at 200 chars so a closing quote beyond the read buffer
# doesn't prevent the match.
_RE_FIRST_MSG = re.compile(
    r'"message"\s*:\s*\{.*?"text"\s*:\s*"((?:[^"\\]|\\.){0,200})', re.DOTALL
)


def parse_json_metadata(path: Path) -> SessionInfo | None:
    """Fast metadata extraction from a .json chat session file.

    Reads the last 1KB to extract creationDate, customTitle, sessionId
    (which live at the end of the file), and the first 2KB to get the
    first user message.  Falls back to full parse if the tail doesn't
    end with the expected closing brace.
    """
    size = path.stat().st_size
    if size == 0:
        return None

    # Read last 1KB for metadata fields that live at the end.
    with open(path, "rb") as fh:
        fh.seek(max(0, size - 1024))
        tail = fh.read().decode("utf-8", errors="replace")

    # Sanity check: file should end with "}"
    if not tail.rstrip().endswith("}"):
        return parse_json(path)  # fall back to full parse

    m = _RE_CREATION_DATE.search(tail)
    creation_date = int(m.group(1)) if m else None

    m = _RE_CUSTOM_TITLE.search(tail)
    title = m.group(1) if m else None

    m = _RE_SESSION_ID.search(tail)
    session_id = m.group(1) if m else path.stem

    # Read first 4KB for the first user message.
    with open(path, "rb") as fh:
        head = fh.read(4096).decode("utf-8", errors="replace")

    m = _RE_FIRST_MSG.search(head)
    first_user = m.group(1) if m else ""
    # Unescape basic JSON escapes in the extracted string.
    if first_user:
        first_user = first_user.replace("\\n", "\n").replace('\\"', '"')

    # Check whether the file has any requests ("requests": [...])
    has_requests = '"requests"' in head and '"requests": []' not in head

    info: SessionInfo = {
        "path": str(path),
        "session_id": session_id,
        "title": title,
        "creation_date": creation_date,
        "model": "",
        "size": size,
        "requests": [{"user": first_user}] if has_requests else [],
    }
    return info


def parse_json(path: Path) -> SessionInfo | None:
    """Parse a .json chat session file (full parse)."""
    try:
        data = json.loads(path.read_text(errors="replace"))
    except json.JSONDecodeError:
        return None

    info: SessionInfo = {
        "path": str(path),
        "session_id": data.get("sessionId", path.stem),
        "title": data.get("customTitle"),
        "creation_date": data.get("creationDate"),
        "size": path.stat().st_size,
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


def load_all_sessions(
    metadata_only: bool = False, limit: int | None = None
) -> list[SessionInfo]:
    """Load all sessions from disk.

    Handles both old format (JSON/JSONL files) and new format (GitHub.copilot-chat).

    When metadata_only is True, use fast head+tail extraction instead of
    full parsing.  Use load_session_by_path() for full parsing.

    When limit is set and metadata_only is True, use file mtime to
    pre-sort candidates and only parse the most recent ones.  This
    avoids the I/O cost of reading all files over slow filesystems.
    """
    sessions: list[SessionInfo] = []

    # Collect candidate files: (path, workspace, suffix).
    # For new-format directories, create SessionInfo immediately (no parsing).
    candidates: list[tuple[Path, str]] = []

    for session_dir in find_session_dirs():
        workspace = get_workspace_name(session_dir)

        # Check if this is a GitHub.copilot-chat directory (new format)
        if "GitHub.copilot-chat" in str(session_dir):
            # New format: sessions are directories in chat-session-resources/
            chat_resources = session_dir / "chat-session-resources"
            if chat_resources.is_dir():
                for session_uuid_dir in chat_resources.iterdir():
                    if not session_uuid_dir.is_dir():
                        continue
                    session_id = session_uuid_dir.name
                    info: SessionInfo = {
                        "path": str(session_uuid_dir),
                        "session_id": session_id,
                        "title": None,
                        "creation_date": None,
                        "size": 0,
                        "model": "",
                        "requests": [],
                        "workspace": workspace,
                    }
                    sessions.append(info)
        else:
            # Old format: JSON/JSONL files
            for f in session_dir.iterdir():
                if f.suffix in (".jsonl", ".json"):
                    candidates.append((f, workspace))

    # When we have a limit and only need metadata, use mtime to pre-sort
    # so we only parse the most recent files.
    if metadata_only and limit and len(candidates) > limit:
        # stat each file for mtime (cheap compared to open+read+parse)
        mtime_candidates: list[tuple[float, Path, str]] = []
        for f, workspace in candidates:
            try:
                mtime_candidates.append((f.stat().st_mtime, f, workspace))
            except OSError:
                continue
        mtime_candidates.sort(reverse=True)
        # Parse 2x the limit to allow for empty sessions being filtered out.
        candidates = [(f, ws) for _, f, ws in mtime_candidates[: limit * 2]]

    # Parse the candidate files.
    for f, workspace in candidates:
        if metadata_only:
            parsed_info = (
                parse_jsonl_metadata(f)
                if f.suffix == ".jsonl"
                else parse_json_metadata(f)
            )
        else:
            parsed_info = parse_jsonl(f) if f.suffix == ".jsonl" else parse_json(f)
        if parsed_info is not None:
            parsed_info["workspace"] = workspace
            sessions.append(parsed_info)

    # Sort by creation date (newest first)
    sessions.sort(
        key=lambda s: s.get("creation_date") or 0,
        reverse=True,
    )
    return sessions


def load_session_by_path(path_str: str) -> SessionInfo | None:
    """Fully parse a single session file by its path."""
    path = Path(path_str)
    if path.is_dir():
        # New format directory — no full parse available yet
        return None
    if path.suffix == ".jsonl":
        return parse_jsonl(path)
    elif path.suffix == ".json":
        return parse_json(path)
    return None


def format_timestamp(ts: int | None) -> str:
    if not ts:
        return "?"
    # VS Code stores timestamps in milliseconds
    dt = datetime.datetime.fromtimestamp(ts / 1000)
    return dt.strftime("%Y-%m-%d %H:%M")


def get_terminal_width() -> int:
    """Get terminal character width."""
    return shutil.get_terminal_size(fallback=(80, 24)).columns


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
        if not reqs and not show_all:
            continue
        title = s.get("title")
        first_msg = ""
        if reqs:
            first_msg = reqs[0].get("user", "")
        label = title or first_msg or "(empty)"
        # Remove newlines to prevent formatting issues
        label = label.replace("\n", " ").replace("\r", "")
        date_str = format_timestamp(s.get("creation_date"))
        workspace = s.get("workspace", "?")
        size_kb = s.get("size", 0) / 1024
        if size_kb >= 1024:
            size_str = f"{size_kb / 1024:.1f}M"
        else:
            size_str = f"{size_kb:.0f}K"

        if use_color:
            # Colorize the session listing
            line = (
                f"  {Fore.CYAN}{i + 1:3d}{Style.RESET_ALL}. "
                f"[{Fore.YELLOW}{date_str}{Style.RESET_ALL}] "
                f"({Fore.MAGENTA}{workspace}{Style.RESET_ALL}) "
                f"{Fore.GREEN}{size_str:>5}{Style.RESET_ALL} {label}"
            )
        else:
            line = f"  {i + 1:3d}. [{date_str}] ({workspace}) {size_str:>5} {label}"
        # Clip to terminal width (use visible length to account for ANSI codes)
        if visible_len(line) > width:
            line = clip_to_visible_length(line, width - 1)
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
                # Preserve paragraph structure while indenting
                lines = thinking.split("\n")
                indented_lines = ["  " + line for line in lines]
                print(
                    f"\n{Fore.MAGENTA}<thinking>{Style.RESET_ALL}\n"
                    + "\n".join(indented_lines)
                    + f"\n{Fore.MAGENTA}</thinking>{Style.RESET_ALL}"
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
                # Preserve paragraph structure while indenting
                lines = thinking.split("\n")
                indented_lines = ["  " + line for line in lines]
                print(f"\n<thinking>\n" + "\n".join(indented_lines) + "\n</thinking>")

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
    """Search all sessions for messages containing query text.

    Note: Search includes only user and assistant messages, not thinking or tool calls.
    """
    query_lower = query.lower()
    hits = 0
    width = term_width if term_width is not None else 999999
    for i, s in enumerate(sessions):
        for req in s.get("requests", []):
            user = req.get("user", "")
            assistant = req.get("assistant", "")
            if query_lower in user.lower() or query_lower in assistant.lower():
                title = s.get("title") or "(untitled)"
                title = title.replace("\n", " ").replace("\r", "")
                date_str = format_timestamp(s.get("creation_date"))
                workspace = s.get("workspace", "?")
                if use_color:
                    line1 = (
                        f"\n{Fore.CYAN}{i + 1:3d}{Style.RESET_ALL}. "
                        f"[{Fore.YELLOW}{date_str}{Style.RESET_ALL}] "
                        f"({Fore.MAGENTA}{workspace}{Style.RESET_ALL}) "
                        f"{Fore.GREEN}{title}{Style.RESET_ALL}"
                    )
                else:
                    line1 = f"\n{i + 1}. [{date_str}] ({workspace}) {title}"
                if visible_len(line1) > width:
                    line1 = clip_to_visible_length(line1, width - 1)
                print(line1)
                # Show the matching message snippet
                for text, label in [(user, "YOU"), (assistant, "COPILOT")]:
                    idx = text.lower().find(query_lower)
                    if idx >= 0:
                        # Extract enough to fill the line width around the match
                        # Account for prefix length to leave room for: "    YOU/COPILOT: "
                        prefix_len = len("    YOU: ")  # rough estimate
                        available = max(
                            40, width - prefix_len - 10
                        )  # -10 for ANSI codes
                        half_avail = available // 2
                        # Compute initial start/end with match centered
                        start = max(0, idx - half_avail)
                        end = min(len(text), idx + len(query) + half_avail)
                        # If we hit a boundary, use the extra space on the other side
                        left_unused = idx - start
                        right_unused = end - (idx + len(query))
                        if start == 0:
                            end = min(len(text), end + left_unused)
                        elif end == len(text):
                            start = max(0, start - right_unused)
                        snippet = text[start:end].replace("\n", " ")
                        has_start_ellipsis = start > 0
                        has_end_ellipsis = end < len(text)

                        # Highlight the query in the snippet
                        snippet = highlight_query(snippet, query)

                        if has_start_ellipsis:
                            snippet = "..." + snippet
                        if has_end_ellipsis:
                            snippet = snippet + "..."

                        if use_color:
                            prefix = f"    {Fore.CYAN}{label}{Style.RESET_ALL}: "
                        else:
                            prefix = f"    {label}: "

                        line2 = prefix + snippet
                        # Clip to terminal width using visible length, preserving trailing "..."
                        if visible_len(line2) > width:
                            line2 = clip_to_visible_length(line2, width - 4) + "..."
                        print(line2)
                hits += 1
    if hits == 0:
        print(f"No messages found matching '{query}'.")
    else:
        print(f"\n{hits} match(es) found.")


def get_default_pager() -> str | None:
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
    # 4. Platform default: less on Unix, built-in on Windows.
    if sys.platform != "win32":
        return "less"
    return None


def _read_one_key() -> str:
    """Read a single keypress without echo. Returns the character, or '' for
    unrecognised special keys (e.g. arrow keys on Windows)."""
    # Platform-specific imports are inside the function because msvcrt is
    # Windows-only and termios/tty/select are Unix-only.
    if sys.platform == "win32":
        import msvcrt

        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):  # start of a two-byte special key
            msvcrt.getwch()  # discard second byte
            return ""
        return ch
    else:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            # Drain the rest of any escape sequence (e.g. arrow keys).
            if ch == "\x1b":
                while select.select([sys.stdin], [], [], 0.05)[0]:
                    sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch


@contextlib.contextmanager
def builtin_pager() -> Iterator[None]:
    """Built-in forward-only pager: Space=next page, Enter=next line, q=quit."""
    if not sys.stdout.isatty():
        yield
        return

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf  # type: ignore[assignment]
    try:
        yield
    finally:
        sys.stdout = old_stdout

    output = buf.getvalue()
    lines = output.splitlines(keepends=True)
    page_size = max(1, shutil.get_terminal_size().lines - 1)

    if len(lines) <= page_size:
        old_stdout.write(output)
        old_stdout.flush()
        return

    # Show first page.
    pos = min(page_size, len(lines))
    old_stdout.write("".join(lines[:pos]))
    old_stdout.flush()

    prompt = "--More-- (Space=page, Enter=line, q=quit) "
    while pos < len(lines):
        old_stdout.write(prompt)
        old_stdout.flush()
        key = _read_one_key()
        # Erase the prompt line.
        old_stdout.write("\r" + " " * len(prompt) + "\r")
        old_stdout.flush()
        if key in ("q", "Q", "\x1b", "\x03"):  # q, Q, ESC, Ctrl-C
            break
        elif key in ("\r", "\n"):  # Enter — one more line
            old_stdout.write(lines[pos])
            old_stdout.flush()
            pos += 1
        else:  # Space or anything else — next full page
            end = min(pos + page_size, len(lines))
            old_stdout.write("".join(lines[pos:end]))
            old_stdout.flush()
            pos = end


@contextlib.contextmanager
def smart_pager(pager_cmd: str) -> Iterator[None]:
    """Pipe stdout directly through an external pager process.

    For ``less``, LESS=FRX causes it to exit automatically when all output
    fits on one screen.
    """
    if not sys.stdout.isatty():
        yield
        return

    env = os.environ.copy()
    # less: quit-if-one-screen, raw-control-chars, no-init
    env.setdefault("LESS", "FRX")
    try:
        proc = subprocess.Popen(
            shlex.split(pager_cmd),
            shell=False,
            stdin=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    except OSError:
        yield
        return

    old_stdout = sys.stdout
    sys.stdout = proc.stdin  # type: ignore[assignment]
    try:
        yield
    except BrokenPipeError:
        pass
    finally:
        sys.stdout = old_stdout
    try:
        proc.stdin.close()  # type: ignore[union-attr]
    except OSError:
        pass
    proc.wait()


def main() -> None:
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
        help="Pager command (default: from git config, then $GIT_PAGER, $PAGER, built-in)",
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

    explicit_pager = args.pager
    configured_pager = (
        explicit_pager if explicit_pager is not None else get_default_pager()
    )

    # For search, we need full parsing; for everything else, metadata suffices.
    need_full = args.search is not None
    # Pass limit so load_all_sessions can skip parsing old files when -n is set.
    listing_limit = args.n if not need_full and not args.session else None
    sessions = load_all_sessions(metadata_only=not need_full, limit=listing_limit)
    if not sessions:
        if use_color:
            print(f"{Fore.RED}No chat sessions found.{Style.RESET_ALL}")
        else:
            print("No chat sessions found.")
        return

    use_pager = not args.no_pager
    if not use_pager:
        ctx: contextlib.AbstractContextManager[None] = contextlib.nullcontext()
    elif configured_pager is not None:
        ctx = smart_pager(configured_pager)
    else:
        ctx = builtin_pager()

    # Always get terminal width for reasonable snippet extraction and display
    # Only used for clipping if stdout is a TTY or using pager
    term_width = get_terminal_width()

    with ctx:
        if args.search:
            search_sessions(sessions, args.search, term_width=term_width)
            return

        if args.session:
            # Try as a list index first
            try:
                idx = int(args.session) - 1
                if 0 <= idx < len(sessions):
                    full = load_session_by_path(sessions[idx]["path"])
                    show_session(full or sessions[idx])
                    return
            except ValueError:
                pass
            # Try as a session ID
            for s in sessions:
                if s.get("session_id") == args.session:
                    full = load_session_by_path(s["path"])
                    show_session(full or s)
                    return
            print(f"Session not found: {args.session}")
            return

        n_empty = sum(1 for s in sessions if not s.get("requests"))
        if listing_limit:
            # With -n, we only parsed a subset, so don't report total counts.
            print()
        elif use_color:
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
