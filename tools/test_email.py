# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import shlex
import asyncio
import sys
import traceback
from typing import Any, Iterable
from colorama import Fore
from pathlib import Path

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
    def __init__(self, db_path: str, conversation: EmailMemory) -> None:
        self.db_path = db_path
        self.conversation = conversation

    async def reset(self):
        await self.conversation.settings.conversation_settings.storage_provider.close()
        self.conversation = await load_or_create_email_index(self.db_path, create_new=True) 

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
        db_path=db_path,
        conversation=await load_or_create_email_index(db_path, create_new=False)
    )
    print(f"Using email memory at: {db_path}")
    await print_conversation_stats(context.conversation)

    # Command handlers
    cmd_handlers = {
        "@add_messages": add_messages,  # Add messages
        "@build_index": build_index, # Build index
        "@reset_index": reset_index, # Delete  index and start over
        "@search": search_index,   # Search index 
        "@answer": generate_answer # Question answer
    }
    while True:
        line = input("✉>>").strip()
        if len(line) == 0:
            continue
        elif line == "exit":
            break        
        args = shlex.split(line)
        if len(args) < 1:
            continue
        try:
            cmd = args[0].lower()
            if cmd == "@help":
                print_commands(cmd_handlers.keys())
            else:
                cmd_handler = cmd_handlers.get(cmd)
                if cmd_handler:
                    args.pop(0)
                    await cmd_handler(context, args)
                else:
                    print_commands(cmd_handlers.keys())      
        except Exception as e:
            print()
            print(Fore.RED, f"Error\n: {e}")
            traceback.print_exc()

        print(Fore.RESET)

# ==
# COMMANDS 
# ==

# Adds messages. Takes a path either to a file or to a directory
async def add_messages(context: EmailContext, args: list[str]):
    if len(args) < 1:
        print_error("No path provided")
        return
    
    # Get the path to the email file or directory of emails to ingest 
    src_path = Path(args[0])
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
    await context.reset()
    await print_conversation_stats(context.conversation)
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


#
# Printing
#

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

def print_commands(names: Iterable[str]):
    print_list(Fore.GREEN, sorted(names), "Commands")

def print_list(color, list: Iterable[Any], title: str):
    if title:
        print(color + f"# {title}")
        print()
    for item in list:
        print(color + " -", item)

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
