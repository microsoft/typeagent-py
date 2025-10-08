# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import shlex
import asyncio
import sys
import traceback
from typing import (
    Any, 
    Literal,
    Iterable, 
    Callable, 
    Awaitable
)
from colorama import Fore
from pathlib import Path
import argparse

import typechat

from typeagent.aitools import utils
from typeagent.knowpro import ( 
    kplib, 
    searchlang, 
)
from typeagent.knowpro.interfaces import (
    IConversation
)
from typeagent.emails.email_import import (
    import_email_from_file, 
    import_emails_from_dir
)
from typeagent.emails.email_memory import EmailMemory
from typeagent.emails.email_message import EmailMessage

from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.storage.utils import create_storage_provider

from utool import print_result

class EmailContext:
    def __init__(self, base_path: Path, db_name: str, conversation: EmailMemory) -> None:
        self.base_path = base_path
        self.db_path = base_path.joinpath(db_name)
        self.conversation = conversation

    async def load_conversation(self, db_name: str, create_new:bool = False):
        await self.conversation.settings.conversation_settings.storage_provider.close()
        self.db_path = self.base_path.joinpath(db_name)
        self.conversation = await load_or_create_email_index(str(self.db_path), create_new) 

    # Delete the current conversation and re-create it
    async def restart_conversation(self):
        await self.conversation.settings.conversation_settings.storage_provider.close()
        self.conversation = await load_or_create_email_index(str(self.db_path), create_new=True) 


CommandHandler = Callable[[EmailContext, list[str]], Awaitable[None]]

# Command decorator
def command(parser: argparse.ArgumentParser):
    def decorator(func: Callable):
        func.parser = parser # type: ignore
        return func
    return decorator

# Just simple test code
# TODO : Once stable, move creation etc to utool.py
async def main():

    base_path = Path("/data/testChat/knowpro/email/") 
    base_path.mkdir(parents=True, exist_ok=True)
    utils.load_dotenv()

    print("Email Memory Demo")
    print("Type @help for a list of commands")
        
    db_path = str(base_path.joinpath("pyEmails.db"))
    context = EmailContext(
        base_path,
        "pyEmails.db",
        conversation=await load_or_create_email_index(db_path, create_new=False)
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
        "@build_index": build_index, # Build index
        "@reset_index": reset_index, # Delete  index and start over
        "@search": search_index,   # Search index 
        "@answer": generate_answer # Question answer
    }
    default_handler = generate_answer
    while True:
        line = input("✉>>").strip()
        if len(line) == 0:
            continue
        try:
            args = shlex.split(line)
            if len(args) < 1:
                continue
            cmd = args[0].lower()
            args.pop(0)
            if cmd == "@help":
                help(cmd_handlers, args)
            else:
                cmd_handler = cmd_handlers.get(cmd)
                if cmd_handler is None and not cmd.startswith("@"):
                    cmd_handler = default_handler
                if cmd_handler:
                    await cmd_handler(context, args)
                else:
                    print_commands(cmd_handlers)
        except Exception as e:
            print()
            print(Fore.RED, f"Error\n: {e}")
            traceback.print_exc()

        print(Fore.RESET)

# ==
# COMMANDS 
# ==

# Adds messages. Takes a path either to a file or to a directory
def _add_messages_def() -> argparse.ArgumentParser:
    cmd = argparse.ArgumentParser(
        description="Add messages to index"
    )
    cmd.add_argument(
        "--path", 
        default="", 
        help="Path to an .eml file or to a directory with .eml files"
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
    emails: list[EmailMessage]
    if src_path.is_file():
        emails = [import_email_from_file(str(src_path))]
    else:
        emails = import_emails_from_dir(str(src_path))

    print(Fore.CYAN, f"Importing {len(emails)} emails".capitalize())
    print()

    conversation = context.conversation
    for email in emails:
        print_email(email)
        print()
        # knowledge = email.metadata.get_knowledge()
        # print_knowledge(knowledge)

        print("Adding email...")
        await conversation.add_message(email)

    await print_conversation_stats(conversation)

async def build_index(context: EmailContext, args: list[str]):
    conversation = context.conversation
    print(Fore.GREEN, "Building index")
    await print_conversation_stats(conversation)
    await conversation.build_index()
    print(Fore.GREEN + "Built index.")

async def search_index(context:EmailContext, args: list[str]):
    if len(args) == 0:
        return
    search_text = args[0].strip()
    if len(search_text) == 0:
        print_error("No search text")
        return
    
    print(Fore.CYAN, f"Searching for:\n{search_text} ")
    
    debug_context = searchlang.LanguageSearchDebugContext()
    results = await context.conversation.search_with_language(
        search_text=search_text,
        debug_context=debug_context
    )
    await print_search_results(context.conversation, debug_context, results)

async def generate_answer(context: EmailContext, args:list[str]):
    if len(args) == 0:
        return
    question = args[0].strip()
    if len(question) == 0:
        print_error("No question")
        return
    
    print(Fore.CYAN, f"Getting answer for:\n{question} ")
    result = await context.conversation.get_answer_with_language(
        question=question
    ) 
    if isinstance(result, typechat.Failure):
        print_error(result.message)
        return

    all_answers, _ = result.value
    utils.pretty_print(all_answers)

async def reset_index(context: EmailContext, args: list[str]):
    print(f"Deleting {context.db_path}")
    await context.restart_conversation()
    await print_conversation_stats(context.conversation)


def _load_index_def() -> argparse.ArgumentParser:
    cmdDef = argparse.ArgumentParser(
        description="Load index at given db path"
    )
    cmdDef.add_argument("--name", type=str, default="", help="Name of the index to load")
    cmdDef.add_argument("--new", type=bool, default=False)
    return cmdDef

@command(_load_index_def())
async def load_index(context: EmailContext, args: list[str]):
    named_args = _load_index_def().parse_args(args)
    
    db_name: str = named_args.name
    if (len(db_name) == 0):
        return
    
    if not db_name.endswith(".db"):
        db_name += ".db"
    print(db_name)
    await context.load_conversation(db_name, named_args.new)

def _parse_messages_def() -> argparse.ArgumentParser:
    cmdDef = argparse.ArgumentParser(
        description="Parse messages in the given path"
    )
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
        file_paths = [str(file_path) for file_path in Path(src_path).iterdir() if file_path.is_file()]

    print(f"Parsing {len(file_paths)} messages")
    for file_path in file_paths:
        try:
            msg = import_email_from_file(file_path)
            print(file_path)
            if named_args.verbose:
                print("####################")
                print_email(msg)
                print_knowledge(msg.get_knowledge())
                print("####################")

        except Exception as e:
            print_error(file_path)
            print_error(str(e))

async def exit_app(context: EmailContext, args: list[str]):
    print("Goodbye")
    sys.exit(0)

def help(handlers: dict[str, CommandHandler], args: list[str]):
    if len(args) > 0:
        cmd = handlers.get(args[0])
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
        print(f"Deleting {db_path}")
        delete_sqlite_db(db_path)

    settings = ConversationSettings()
    settings.storage_provider = await create_storage_provider(
        settings.message_text_index_settings,
        settings.related_term_index_settings,
        db_path,
        EmailMessage
    )
    return await EmailMemory.create(settings)

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

#=========================
#
# Printing
#
#=========================

def print_help(handler: CommandHandler):
    if hasattr(handler, "parser"):
        parser: argparse.ArgumentParser = handler.parser # type: ignore
        print(parser.format_help())
        print()

def print_commands(commands: dict[str, CommandHandler]):
    names = sorted(commands.keys())
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
        print(Fore.CYAN + chunk)

    print(Fore.RESET)

def print_knowledge(knowledge: kplib.KnowledgeResponse):
    print_list(Fore.GREEN, knowledge.topics, "Topics")
    print()
    print_list(Fore.GREEN, knowledge.entities, "Entities")
    print()
    print_list(Fore.GREEN, knowledge.actions, "Actions")
    print()
    print(Fore.RESET)

def print_list(color, list: Iterable[Any], title: str, type: Literal["plain", "ol", "ul"] = "plain"):
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
    print(Fore.RED + msg)
    print(Fore.RESET)

async def print_conversation_stats(conversation: IConversation):
    print(f"Conversation index stats".upper())
    print(f"Message count: {await conversation.messages.size()}")
    print(f"Semantic Ref count: {await conversation.semantic_refs.size()}")

async def print_search_results(
        conversation: IConversation,
        debug_context: searchlang.LanguageSearchDebugContext,
        results: typechat.Result[list[searchlang.ConversationSearchResult]]
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

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, BrokenPipeError):
        print()
        sys.exit(1)
