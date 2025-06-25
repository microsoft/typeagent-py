# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Literal, Annotated
from typing_extensions import Doc
from pydantic.dataclasses import dataclass

AnswerType = Literal[
    "NoAnswer",  # If question cannot be accurately answered from [ANSWER CONTEXT]
    "Answered",  # Fully answer question
]


@dataclass
class AnswerResponse:
    type: Annotated[
        AnswerType,
        Doc(
            'use "NoAnswer" if no highly relevant answer found in the [ANSWER CONTEXT]'
        ),
    ]
    answer: Annotated[
        str | None,
        Doc(
            "the answer to display if [ANSWER CONTEXT] is highly relevant and can be used to answer the user's question"
        ),
    ] = None
    whyNoAnswer: Annotated[
        str | None,
        Doc(
            "If NoAnswer, explain why..\nparticularly explain why you didn't use any supplied entities"
        ),
    ] = None
