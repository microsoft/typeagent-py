# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import asyncio
import os
from pathlib import Path
import shelve
import shlex
import sys
import traceback
from typing import Any, Awaitable, Callable, Iterable, Literal

from colorama import Fore

try:
    import readline  # type: ignore
except ImportError:
    pass  # readline not available on Windows

from query import print_result

import typechat

from typeagent.aitools import utils
from typeagent.emails.email_import import import_email_from_file, import_emails_from_dir
from typeagent.emails.email_memory import EmailMemory
from typeagent.emails.email_message import EmailMessage
from typeagent.knowpro import convknowledge, kplib, search_query_schema, searchlang
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.knowpro.interfaces import IConversation
from typeagent.storage.utils import create_storage_provider


class ReallyExit(Exception):
    pass


class EmailContext:
    def __init__(
        self, base_path: Path, db_name: str, conversation: EmailMemory
    ) -> None:
        self.base_path = base_path
        self.db_name = db_name
        self.db_path = base_path.joinpath(db_name)
        self.conversation = conversation
        self.query_translator: (
            typechat.TypeChatJsonTranslator[search_query_schema.SearchQuery] | None
        ) = None
        self.index_log = load_index_log(str(self.db_path), create_new=False)

    def get_translator(self):
        if self.query_translator is None:
            model = convknowledge.create_typechat_model()
            self.query_translator = utils.create_translator(
                model, search_query_schema.SearchQuery
            )
        return self.query_translator

    async def load_conversation(self, db_name: str, create_new: bool = False):
        await self.conversation.settings.storage_provider.close()
        self.db_name = db_name
        self.db_path = self.base_path.joinpath(db_name)
        self.conversation = await load_or_create_email_index(
            str(self.db_path), create_new
        )
        self.index_log = load_index_log(str(self.db_path), create_new)

    # Delete the current conversation and re-create it
    async def restart_conversation(self):
        await self.load_conversation(self.db_name, create_new=True)

    def is_indexed(self, email_id: str | None) -> bool:
        return bool(email_id and self.index_log.get(email_id))

    def log_indexed(self, email_id: str | None) -> None:
        if email_id is not None:
            self.index_log[email_id] = True


CommandHandler = Callable[[EmailContext, list[str]], Awaitable[None]]


# Command decorator
def command(parser: argparse.ArgumentParser):
    def decorator(func: Callable):
        func.parser = parser  # type: ignore
        return func

    return decorator


async def main():

    if sys.argv[1:2]:
        base_path = Path(sys.argv[1])
    elif os.path.exists("/data"):
        base_path = Path("/data/testChat/knowpro/email/")
    else:
        base_path = Path(".")

    try:
        base_path.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        print(e)
        sys.exit(1)

    utils.load_dotenv()

    print("Email Memory Demo")

    default_db = "gmail.db"  # "pyEmails.db"
    db_path = str(base_path.joinpath(default_db))
    context = EmailContext(
        base_path,
        default_db,
        conversation=await load_or_create_email_index(db_path, create_new=False),
    )
    print(f"Using email memory at: {db_path}")
    await print_conversation_stats(context.conversation)

    # Command handlers
    cmd_handlers: dict[str, CommandHandler] = {
        "@exit": exit_app,
        "@quit": exit_app,
        "@add_messages": add_messages,  # Add messages
        "@parse_messages": parse_messages,
        "@load_index": load_index,
        "@reset_index": reset_index,  # Delete  index and start over
        "@search": search_index,  # Search index
        "@answer": generate_answer,  # Question answer
    }

    async def default_handler(context, line):
        return await generate_answer(context, [line])

    print("Type @help for a list of commands")

    while True:
        try:
            line = input("✉>> ").strip()
        except EOFError:
            print()
            break
        if not line:
            continue
        try:
            if not line.startswith("@"):
                await default_handler(context, line)
            else:
                try:
                    args = shlex.split(line, comments=True)
                except ValueError as e:
                    print(Fore.RED + f"Error parsing command: {e}" + Fore.RESET)
                    continue
                if len(args) < 1:
                    continue
                cmd = args.pop(0).lower()
                if cmd == "@help":
                    help(cmd_handlers, args)
                else:
                    cmd_handler = cmd_handlers.get(cmd)
                    if cmd_handler:
                        await cmd_handler(context, args)
                    else:
                        print_commands(cmd_handlers)
        except ReallyExit:
            # Raised by exit_app() to reall exit the app
            sys.exit(0)
        except Exception as e:
            print()
            print(Fore.RED + f"Error\n: {e}" + Fore.RESET)
            traceback.print_exc()
        except SystemExit as e:
            # Command handlers using argparse may see this
            if e.code != 0:
                print(Fore.RED + f"Error: {e}" + Fore.RESET)
        except KeyboardInterrupt:
            print()

        print(Fore.RESET)


# ==
# COMMANDS
# ==


# Adds messages. Takes a path either to a file or to a directory
def _add_messages_def() -> argparse.ArgumentParser:
    cmd = argparse.ArgumentParser(
        description="Add messages to index", prog="@add_messages"
    )
    cmd.add_argument(
        "--path",
        default="",
        help="Path to an .eml file or to a directory with .eml files",
    )
    cmd.add_argument("--ignore_error", type=bool, default=True, help="Ignore errors")
    cmd.add_argument(
        "--knowledge", type=bool, default=True, help="Automatically extract knowledge"
    )
    return cmd


@command(_add_messages_def())
async def add_messages(context: EmailContext, args: list[str]):
    named_args = _add_messages_def().parse_args(args)
    if named_args.path is None:
        print("No path provided")
        return

    # Get the path to the email file or directory of emails to ingest
    src_path = Path(named_args.path)
    emails: Iterable[EmailMessage]
    if src_path.is_file():
        emails = [import_email_from_file(str(src_path))]
    else:
        emails = import_emails_from_dir(str(src_path))

    print(Fore.CYAN + f"Importing from {src_path}" + Fore.RESET)

    semantic_settings = context.conversation.settings.semantic_ref_index_settings
    auto_knowledge = semantic_settings.auto_extract_knowledge
    print(Fore.CYAN + f"auto_extract_knowledge={auto_knowledge}" + Fore.RESET)
    try:
        conversation = context.conversation
        # Add one at a time for debugging etc.
        for i, email in enumerate(emails):
            email_id = email.metadata.id
            email_src = email.src_url if email.src_url is not None else ""
            print_progress(i + 1, None, email.src_url)
            print()
            if context.is_indexed(email_id):
                print(Fore.GREEN + email_src + "[Already indexed]" + Fore.RESET)
                continue

            try:
                await conversation.add_messages_with_indexing([email])
                context.log_indexed(email_id)
            except Exception as e:
                if named_args.ignore_error:
                    print_error(f"{email.src_url}\n{e}")
                    print(
                        Fore.GREEN
                        + f"ignore_error = {named_args.ignore_error}"
                        + Fore.RESET
                    )
                else:
                    raise
    finally:
        semantic_settings.auto_extract_knowledge = auto_knowledge

    await print_conversation_stats(conversation)


async def search_index(context: EmailContext, args: list[str]):
    if not args:
        return
    search_text = args[0].strip()
    if not search_text:
        print_error("No search text")
        return

    print(Fore.CYAN + f"Searching for:\n{search_text} " + Fore.RESET)

    debug_context = searchlang.LanguageSearchDebugContext()
    results = await context.conversation.query_debug(
        search_text=search_text,
        query_translator=context.get_translator(),
        debug_context=debug_context,
    )
    await print_search_results(context.conversation, debug_context, results)


async def generate_answer(context: EmailContext, args: list[str]):
    if len(args) == 0:
        return
    question = args[0].strip()
    if len(question) == 0:
        print_error("No question")
        return

    print(Fore.CYAN + f"Getting answer for:\n{question} " + Fore.RESET)

    answer = await context.conversation.query(question)
    color = Fore.RED if answer.startswith("No answer found:") else Fore.GREEN
    print(color + answer + Fore.RESET)


async def reset_index(context: EmailContext, args: list[str]):
    print(f"Deleting {context.db_path}")
    await context.restart_conversation()
    await print_conversation_stats(context.conversation)


def _load_index_def() -> argparse.ArgumentParser:
    cmdDef = argparse.ArgumentParser(
        description="Load index at given db path", prog="@load_index"
    )
    cmdDef.add_argument(
        "--name", type=str, default="", help="Name of the index to load"
    )
    cmdDef.add_argument("--new", type=bool, default=False)
    return cmdDef


@command(_load_index_def())
async def load_index(context: EmailContext, args: list[str]):
    named_args = _load_index_def().parse_args(args)

    db_name: str = named_args.name
    if len(db_name) == 0:
        return

    if not db_name.endswith(".db"):
        db_name += ".db"
    print(db_name)
    await context.load_conversation(db_name, named_args.new)


def _parse_messages_def() -> argparse.ArgumentParser:
    cmdDef = argparse.ArgumentParser(description="Parse messages in the given path")
    cmdDef.add_argument("--path", type=str, default="")
    cmdDef.add_argument("--verbose", type=bool, default=False)
    return cmdDef


@command(_parse_messages_def())
async def parse_messages(context: EmailContext, args: list[str]):
    named_args = _parse_messages_def().parse_args(args)
    src_path = Path(named_args.path)
    file_paths: list[str]
    if src_path.is_file():
        file_paths = [str(src_path)]
    else:
        file_paths = [
            str(file_path)
            for file_path in Path(src_path).iterdir()
            if file_path.is_file()
        ]

    print(f"Parsing {len(file_paths)} messages")
    for file_path in file_paths:
        try:
            msg = import_email_from_file(file_path)
            print(file_path)
            print("####################")
            print_email(msg)
            if named_args.verbose:
                print_knowledge(msg.get_knowledge())
                print("####################")

        except Exception as e:
            print_error(file_path)
            print_error(str(e))


async def exit_app(context: EmailContext, args: list[str]):
    print("Goodbye")
    raise ReallyExit()


def help(handlers: dict[str, CommandHandler], args: list[str]):
    if len(args) > 0:
        name = args[0]
        if not name.startswith("@"):
            name = "@" + name
        cmd = handlers.get(name)
        if cmd is not None:
            print_help(cmd)
            return

    print_commands(handlers)
    print("@help <commandName> for details")


#
# Utilities
#
async def load_or_create_email_index(db_path: str, create_new: bool) -> EmailMemory:
    if create_new:
        delete_sqlite_db(db_path)

    settings = ConversationSettings()
    settings.storage_provider = await create_storage_provider(
        settings.message_text_index_settings,
        settings.related_term_index_settings,
        db_path,
        EmailMessage,
    )
    email_memory = await EmailMemory.create(settings)
    return email_memory


def load_index_log(db_path: str, create_new: bool) -> shelve.Shelf[Any]:
    log_path = db_path + ".index_log"
    index_log = shelve.open(log_path)
    if create_new:
        index_log.clear()
    return index_log


def delete_sqlite_db(db_path: str):
    if os.path.exists(db_path):
        os.remove(db_path)  # Delete existing database for clean test
        # Also delete -shm and -wal files if they exist
        shm_path = db_path + "-shm"
        wal_path = db_path + "-wal"
        if os.path.exists(shm_path):
            os.remove(shm_path)
        if os.path.exists(wal_path):
            os.remove(wal_path)


# =========================
#
# Printing
#
# =========================


def print_help(handler: CommandHandler):
    if hasattr(handler, "parser"):
        parser: argparse.ArgumentParser = handler.parser  # type: ignore
        print(parser.format_help())
        print()


def print_commands(commands: dict[str, CommandHandler]):
    names = list(commands.keys())
    names.append("@help")
    names.sort()
    print_list(Fore.GREEN, names, "COMMANDS", "ul")


def print_email(email: EmailMessage):
    print("From:", email.metadata.sender)
    print("To:", ", ".join(email.metadata.recipients))
    if email.metadata.cc:
        print("Cc:", ", ".join(email.metadata.cc))
    if email.metadata.bcc:
        print("Bcc:", ", ".join(email.metadata.bcc))
    if email.metadata.subject:
        print("Subject:", email.metadata.subject)
    print("Date:", email.timestamp)

    print("Body:")
    for chunk in email.text_chunks:
        print(Fore.CYAN + chunk + Fore.RESET)

    print(Fore.RESET)


def print_knowledge(knowledge: kplib.KnowledgeResponse):
    print_list(Fore.GREEN, knowledge.topics, "Topics")
    print()
    print_list(Fore.GREEN, knowledge.entities, "Entities")
    print()
    print_list(Fore.GREEN, knowledge.actions, "Actions")
    print()
    print(Fore.RESET)


async def print_conversation_stats(conversation: IConversation):
    print(f"Conversation index stats".upper())
    print(f"Message count: {await conversation.messages.size()}")
    print(f"Semantic Ref count: {await conversation.semantic_refs.size()}")


async def print_search_results(
    conversation: IConversation,
    debug_context: searchlang.LanguageSearchDebugContext,
    results: typechat.Result[list[searchlang.ConversationSearchResult]],
):
    print(Fore.CYAN)
    utils.pretty_print(debug_context.search_query)
    utils.pretty_print(debug_context.search_query_expr)
    if isinstance(results, typechat.Failure):
        print_error(results.message)
    else:
        print(Fore.GREEN, "### SEARCH RESULTS")
        print()
        search_results = results.value
        for search_result in search_results:
            print(Fore.GREEN, search_result.raw_query_text)
            await print_result(search_result, conversation)
    print(Fore.RESET)


def print_list(
    color, list: Iterable[Any], title: str, type: Literal["plain", "ol", "ul"] = "plain"
):
    print(color)
    if title:
        print(f"# {title}\n")
    if type == "plain":
        for item in list:
            print(item)
    elif type == "ul":
        for item in list:
            print(f"- {item}")
    elif type == "ol":
        for i, item in enumerate(list):
            print(f"{i + 1}. {item}")
    print(Fore.RESET)


def print_error(msg: str):
    print(Fore.RED + msg + Fore.RESET)


def print_progress(cur: int, total: int | None = None, suffix: str | None = "") -> None:
    if suffix is None:
        suffix = ""
    if total is not None:
        print(f"[{cur} / {total}] {suffix}\r", end="", flush=True)
    else:
        print(f"[{cur}] {suffix}\r", end="", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, BrokenPipeError):
        print()
        sys.exit(1)
