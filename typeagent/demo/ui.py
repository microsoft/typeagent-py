# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
from dataclasses import asdict
import io
import readline
import shutil
import sys
import traceback
from typing import cast

from black import format_str, FileMode
from click import File
from colorama import AnsiToWin32
import typechat

from ..aitools.auth import load_dotenv
from ..knowpro.convknowledge import create_typechat_model
from ..knowpro.interfaces import (
    DateRange,
    Datetime,
    IConversation,
    IMessage,
    ITermToSemanticRefIndex,
    SemanticRef,
    Topic,
)
from ..knowpro.kplib import Action, ActionParam, ConcreteEntity, Quantity
from ..knowpro.query import QueryEvalContext
from ..knowpro.search import ConversationSearchResult, SearchQueryExpr, run_search_query
from ..knowpro.searchlang import SearchQueryCompiler
from ..knowpro.search_query_schema import SearchQuery
from ..podcasts.podcast import Podcast

from .answer_context_schema import AnswerContext, RelevantKnowledge, RelevantMessage
from .answer_response_schema import AnswerResponse

cap = min  # More readable name for capping a value at some limit


def pretty_print(obj: object) -> None:
    """Pretty-print an object using black.

    Only works if the repr() is a valid Python expression.
    """
    line_width = cap(200, shutil.get_terminal_size().columns)
    print(format_str(repr(obj), mode=FileMode(line_length=line_width)))


def main() -> None:
    load_dotenv()
    model = create_typechat_model()
    translator = create_translator(model, SearchQuery)
    file = "testdata/Episode_53_AdrianTchaikovsky_index"
    pod = Podcast.read_from_file(file)
    assert pod is not None, f"Failed to load podcast from {file!r}"
    context = QueryEvalContext(pod)
    print("TypeAgent demo UI 0.1 (type 'q' to exit)")
    if sys.stdin.isatty():
        try:
            readline.read_history_file(".ui_history")
        except FileNotFoundError:
            pass  # Ignore if history file does not exist.
    try:
        process_inputs(translator, context, cast(io.TextIOWrapper, sys.stdin))
    except KeyboardInterrupt:
        print()
    finally:
        if sys.stdin.isatty():
            readline.write_history_file(".ui_history")


def create_translator[T](
    model: typechat.TypeChatLanguageModel,
    schema: type[T],
) -> typechat.TypeChatJsonTranslator[T]:
    validator = typechat.TypeChatValidator[T](schema)
    return typechat.TypeChatJsonTranslator[T](model, validator, schema)


def process_inputs[TMessage: IMessage, TIndex: ITermToSemanticRefIndex](
    translator: typechat.TypeChatJsonTranslator[SearchQuery],
    context: QueryEvalContext[TMessage, TIndex],
    stream: io.TextIOWrapper,
) -> None:
    ps1 = "--> "
    while True:
        query_text = read_one_line(ps1, stream)
        if query_text is None:  # EOF
            break
        if not query_text:
            continue
        if query_text.lower() in ("exit", "quit", "q"):
            readline.remove_history_item(readline.get_current_history_length() - 1)
            break
        if query_text == "pdb":
            breakpoint()
            continue

        asyncio.run(wrap_process_query(query_text, context.conversation, translator))

        print()


def read_one_line(ps1: str, stream: io.TextIOWrapper) -> str | None:
    """Read a single line from the input stream. Return None for EOF."""
    if stream is sys.stdin and stream.isatty():
        try:
            return input(ps1).strip()
        except EOFError:
            print()
            return None
    else:
        if stream.isatty():
            print(ps1, end="", flush=True)
        line = stream.readline()
        if not line:
            print()
            return None
        return line.strip()


async def wrap_process_query[TMessage: IMessage, TIndex: ITermToSemanticRefIndex](
    query_text: str,
    conversation: IConversation[TMessage, TIndex],
    translator: typechat.TypeChatJsonTranslator[SearchQuery],
) -> None:
    """Wrap the process_query function to handle exceptions."""
    try:
        await process_query(query_text, conversation, translator)
    except Exception as exc:
        traceback.print_exc()
        # traceback.print_exception(type(exc), exc, exc.__traceback__.tb_next)


async def process_query[TMessage: IMessage, TIndex: ITermToSemanticRefIndex](
    query_text: str,
    conversation: IConversation[TMessage, TIndex],
    translator: typechat.TypeChatJsonTranslator[SearchQuery],
):
    # Gradually turn the query text into something we can use to search.

    # TODO: # 0. Recognize @-commands like "@search" and handle them specially.

    # 1. With LLM help, translate to SearchQuery (a tree but not yet usable to query)
    print("Search query:")
    search_query = await translate_text_to_search_query(
        conversation, translator, query_text
    )
    if search_query is None:
        print("Failed to translate command to search terms.")
        return
    pretty_print(search_query)
    print()

    # 2. Translate the search query into something directly usable as a query.
    print("Search query expressions:")
    query_exprs = translate_search_query_to_search_query_exprs(search_query)
    if not query_exprs:
        print("Failed to translate search query to query expressions.")
        return
    for i, query_expr in enumerate(query_exprs):
        print(f"---------- {i} ----------")
        pretty_print(query_expr)
    print()

    # 3. Search!
    for i, query_expr in enumerate(query_exprs):
        print(f"Searching with expression {i}:")
        results = await run_search_query(conversation, query_expr)
        if results is None:
            print(f"No results for expression {i}.")
        else:
            print(f"Results for expression {i}:")
            # pretty_print(results)
            for result in results:
                print_result(result, conversation)
                # pretty_print(result)
                answer = await generate_answer(result, conversation)
                if answer is None:
                    print("No answer generated.")
                elif answer.type == "NoAnswer":
                    print("Why no answer:", answer.whyNoAnswer)
                elif answer.type == "Answered":
                    print("Generated answer:")
                    print(answer.answer)


def print_result[TMessage: IMessage, TIndex: ITermToSemanticRefIndex](
    result: ConversationSearchResult, conversation: IConversation[TMessage, TIndex]
) -> None:
    print(f"Raw query: {result.raw_query_text}")
    if result.message_matches:
        print("Message matches:", result.message_matches)
    if result.knowledge_matches:
        print("Knowledge matches:")
        for key, value in sorted(result.knowledge_matches.items()):
            print(f"Type {key}:")
            print(f"  {value.term_matches}")
            for scored_sem_ref_ord in value.semantic_ref_matches:
                score = scored_sem_ref_ord.score
                sem_ref_ord = scored_sem_ref_ord.semantic_ref_ordinal
                if conversation.semantic_refs is None:
                    print(f"  Ord: {sem_ref_ord} (score {score})")
                else:
                    sem_ref = conversation.semantic_refs[sem_ref_ord]
                    msg_ord = sem_ref.range.start.message_ordinal
                    chunk_ord = sem_ref.range.start.chunk_ordinal
                    msg = conversation.messages[msg_ord]
                    print(
                        f"({score:4.1f}) {msg_ord:3d}: "
                        f"{msg.speaker:>15.15s}: "  # type: ignore  # It's a PodcastMessage
                        f"{repr(msg.text_chunks[chunk_ord].strip())[1:-1]:<50.50s}  "
                        f"{summarize_knowledge(sem_ref)}"
                    )


async def generate_answer[TMessage: IMessage, TIndex: ITermToSemanticRefIndex](
    context: ConversationSearchResult, conversation: IConversation[TMessage, TIndex]
) -> AnswerResponse | None:
    # TODO: lift translator creation out of the outermost loop.
    model = create_typechat_model()
    translator = create_translator(model, AnswerResponse)
    assert context.raw_query_text is not None, "Raw query text must not be None"
    request = f"{create_question_prompt(context.raw_query_text)}\n\n{create_context_prompt(make_context(context, conversation))}"
    # print(request)
    result = await translator.translate(request)
    if isinstance(result, typechat.Failure):
        print(f"Error generating answer: {result.message}")
        return None
    answer = result.value
    return answer


def create_question_prompt(question: str) -> str:
    prompt = [
        "The following is a user question:",
        "===",
        question,
        "===",
        "- The included [ANSWER CONTEXT] contains information that MAY be relevant to answering the question.",
        "- Answer the user question PRECISELY using ONLY relevant topics, entities, actions, messages and time ranges/timestamps found in [ANSWER CONTEXT].",
        "- Return 'NoAnswer' if unsure or if the topics and entity names/types in the question are not in [ANSWER CONTEXT].",
        "- Use the 'name', 'type' and 'facets' properties of the provided JSON entities to identify those highly relevant to answering the question.",
        "- When asked for lists, ensure the the list contents answer the question and nothing else.",
        "E.g. for the question 'List all books': List only the books in [ANSWER CONTEXT].",
        "- Use direct quotes only when needed or asked. Otherwise answer in your own words.",
        "- Your answer is readable and complete, with appropriate formatting: line breaks, numbered lists, bullet points etc.",
    ]
    return "\n".join(prompt)


def create_context_prompt(context: AnswerContext) -> str:
    prompt = [
        "[ANSWER CONTEXT]",
        "===",
        format_str(str(asdict(context)), mode=FileMode(line_length=200)),
        "===",
    ]
    return "\n".join(prompt)


def make_context[TMessage: IMessage, TIndex: ITermToSemanticRefIndex](
    context: ConversationSearchResult, conversation: IConversation[TMessage, TIndex]
) -> AnswerContext:
    a = AnswerContext([], [], [])
    a.entities = []
    a.topics = []
    a.messages = []
    for smo in context.message_matches:
        msg = conversation.messages[smo.message_ordinal]
        # TODO: Dedupe messages
        a.messages.append(
            RelevantMessage(
                from_=None,
                to=None,
                timestamp=None,
                messageText=" ".join(msg.text_chunks),
            )
        )
    for ktype, knowledge in context.knowledge_matches.items():
        assert conversation.semantic_refs is not None, "Semantic refs must not be None"
        match ktype:
            case "entity":
                for scored_sem_ref_ord in knowledge.semantic_ref_matches:
                    sem_ref = conversation.semantic_refs[
                        scored_sem_ref_ord.semantic_ref_ordinal
                    ]
                    entity = cast(ConcreteEntity, sem_ref.knowledge)
                    # TODO: Dedupe entities
                    a.entities.append(
                        RelevantKnowledge(
                            knowledge=asdict(entity),
                            origin=None,
                            audience=None,
                            timeRange=None,
                        )
                    )
            case "topic":
                topic = cast(Topic, knowledge)
                a.topics.append(
                    RelevantKnowledge(
                        knowledge=asdict(topic),
                        origin=None,
                        audience=None,
                        timeRange=None,
                    )
                )
            case _:
                pass  # TODO: Actions and topics too???
    return a


async def translate_text_to_search_query[
    TMessage: IMessage, TIndex: ITermToSemanticRefIndex
](
    conversation: IConversation[TMessage, TIndex],
    translator: typechat.TypeChatJsonTranslator[SearchQuery],
    text: str,
) -> SearchQuery | None:
    prompt_preamble: list[typechat.PromptSection] = []
    time_range_preamble = get_time_range_prompt_section_for_conversation(conversation)
    if time_range_preamble is not None:
        prompt_preamble.append(time_range_preamble)
    result: typechat.Result[SearchQuery] = await translator.translate(
        text, prompt_preamble=prompt_preamble
    )
    if isinstance(result, typechat.Failure):
        print(f"Error translating {text!r}: {result.message}")
        return None
    return result.value


def translate_search_query_to_search_query_exprs(
    search_query: SearchQuery,
) -> list[SearchQueryExpr]:
    return SearchQueryCompiler().compile_query(search_query)


def summarize_knowledge(sem_ref: SemanticRef) -> str:
    """Summarize the knowledge in a SemanticRef."""
    knowledge = sem_ref.knowledge
    if knowledge is None:
        return "<No knowledge>"
    match sem_ref.knowledge_type:
        case "entity":
            entity = knowledge
            assert isinstance(entity, ConcreteEntity)
            res = [f"{entity.name} [{', '.join(entity.type)}]"]
            if entity.facets:
                for facet in entity.facets:
                    value = facet.value
                    if isinstance(value, Quantity):
                        value = f"{value.amount} {value.units}"
                    res.append(f"<{facet.name}:{value}>")
            return " ".join(res)
        case "action":
            action = knowledge
            assert isinstance(action, Action)
            res = []
            res.append("/".join(repr(verb) for verb in action.verbs))
            if action.verb_tense:
                res.append(f"[{action.verb_tense}]")
            if action.subject_entity_name != "none":
                res.append(f"subj={action.subject_entity_name!r}")
            if action.object_entity_name != "none":
                res.append(f"obj={action.object_entity_name!r}")
            if action.indirect_object_entity_name != "none":
                res.append(f"ind_obj={action.indirect_object_entity_name}")
            if action.params:
                for param in action.params:
                    if isinstance(param, ActionParam):
                        res.append(f"<{param.name}:{param.value}>")
                    else:
                        res.append(f"<{param}>")
            if action.subject_entity_facet is not None:
                res.append(f"subj_facet={action.subject_entity_facet}")
            return " ".join(res)
        case "topic":
            topic = knowledge
            assert isinstance(topic, Topic)
            return repr(topic.text)
        case "tag":
            tag = knowledge
            assert isinstance(tag, str)
            return f"#{tag}"
        case _:
            return str(sem_ref.knowledge)


# TODO: Move to conversation.py
def get_time_range_prompt_section_for_conversation[
    TMessage: IMessage, TIndex: ITermToSemanticRefIndex
](
    conversation: IConversation[TMessage, TIndex],
) -> typechat.PromptSection | None:
    time_range = get_time_range_for_conversation(conversation)
    if time_range is not None:
        return typechat.PromptSection(
            role="system",
            content=f"ONLY IF user request explicitly asks for time ranges, "
            f'THEN use the CONVERSATION TIME RANGE: "{time_range.start} to {time_range.end}"',
        )


# TODO: Move to conversation.py
def get_time_range_for_conversation[
    TMessage: IMessage, TIndex: ITermToSemanticRefIndex
](
    conversation: IConversation[TMessage, TIndex],
) -> DateRange | None:
    messages = conversation.messages
    if len(messages) > 0:
        start = messages[0].timestamp
        if start is not None:
            end = messages[-1].timestamp
            return DateRange(
                start=Datetime.fromisoformat(start),
                end=Datetime.fromisoformat(end) if end else None,
            )
    return None


if __name__ == "__main__":
    main()
