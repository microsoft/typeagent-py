# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from typing import Any
from enum import Enum

from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic import Field

from email.utils import parseaddr

from ..knowpro import kplib
from ..knowpro.field_helpers import CamelCaseField
from ..knowpro.interfaces import (
    IKnowledgeSource,
    IMessage,
    IMessageMetadata,
)


@pydantic_dataclass
class EmailMessageMeta(IKnowledgeSource, IMessageMetadata):
    """Metadata for email messages."""

    sender: str
    recipients: list[str] = Field(default_factory=list)
    cc: list[str] = Field(default_factory=list)
    bcc: list[str] = Field(default_factory=list)
    subject: str | None = None
    id: str | None = None

    @property
    def source(self) -> str | None:  # type: ignore[reportIncompatibleVariableOverride]
        return self.sender

    @property
    def dest(self) -> str | list[str] | None:  # type: ignore[reportIncompatibleVariableOverride]
        return self.recipients

    def get_knowledge(self) -> kplib.KnowledgeResponse:
        return kplib.KnowledgeResponse(
            entities=self.to_entities(),
            actions=self.to_actions(),
            inverse_actions=[],
            topics=self.to_topics(),
        )

    def to_entities(self) -> list[kplib.ConcreteEntity]:
        entities: list[kplib.ConcreteEntity] = []

        if self.sender:
            entities.extend(self._email_address_to_entities(self.sender))

        if self.recipients:
            for recipient in self.recipients:
                entities.extend(self._email_address_to_entities(recipient))

        if self.cc:
            for cc in self.cc:
                entities.extend(self._email_address_to_entities(cc))

        if self.bcc:
            for bcc in self.bcc:
                entities.extend(self._email_address_to_entities(bcc))

        entities.append(kplib.ConcreteEntity(name="email", type=["message"]))
        return entities

    def to_topics(self) -> list[str]:
        topics: list[str] = []
        if self.subject:
            topics.append(self.subject)
        return topics

    def to_actions(self) -> list[kplib.Action]:
        actions: list[kplib.Action] = []
        if self.sender and self.recipients:
            for recipient in self.recipients:
                actions.extend(self._createActions("sent", self.sender, recipient))
                actions.extend(self._createActions("received", recipient, self.sender))
        return actions

    # Returns the knowledge entities for a given email address string
    def _email_address_to_entities(
        self, email_address: str
    ) -> list[kplib.ConcreteEntity]:
        entities: list[kplib.ConcreteEntity] = []
        display_name, address = parseaddr(email_address)
        if display_name:
            entity = kplib.ConcreteEntity(
                name=display_name,
                type=["person"],
            )
            if address:
                entity.facets = [
                    kplib.Facet(
                        name="email_address",
                        value=address,
                    )
                ]
            entities.append(entity)

        if address:
            entities.append(
                kplib.ConcreteEntity(
                    name=address,
                    type=["email_address", "alias"],
                )
            )
        return entities

    def _createActions(
        self, verb: str, sender: str, recipient: str
    ) -> list[kplib.Action]:
        sender_display_name, sender_address = parseaddr(sender)
        actions: list[kplib.Action] = []
        if sender_display_name:
            self._add_actions_for_sender(actions, verb, sender_display_name, recipient)

        if sender_address:
            self._add_actions_for_sender(actions, verb, sender_address, recipient)

        return actions

    def _add_actions_for_sender(
        self, actions: list[kplib.Action], verb: str, sender: str, recipient: str
    ) -> None:
        recipient_display_name, recipient_address = parseaddr(recipient)
        if recipient_display_name:
            actions.append(self._createAction(verb, sender, recipient_display_name))

        if recipient_address:
            actions.append(self._createAction(verb, sender, recipient_address))

    def _createAction(
        self, verb: str, sender: str, recipient: str, useIndirect: bool = True
    ) -> kplib.Action:
        if useIndirect:
            return kplib.Action(
                verbs=[verb],
                verb_tense="past",
                subject_entity_name=sender,
                object_entity_name="email",
                indirect_object_entity_name=recipient,
            )
        else:
            return kplib.Action(
                verbs=[verb],
                verb_tense="past",
                subject_entity_name=sender,
                object_entity_name=recipient,
                indirect_object_entity_name="email",
            )


@pydantic_dataclass
class EmailMessage(IMessage):
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

    text_chunks: list[str] = CamelCaseField("The text chunks of the email message")
    metadata: EmailMessageMeta = CamelCaseField(
        "Metadata associated with the email message"
    )
    tags: list[str] = CamelCaseField(
        "Tags associated with the message", default_factory=list
    )
    timestamp: str | None = None  # Use metadata.sent_on for the actual sent time
    src_url: str | None = None  # Source file or uri for this email

    def get_knowledge(self) -> kplib.KnowledgeResponse:
        return self.metadata.get_knowledge()

    def add_timestamp(self, timestamp: str) -> None:
        self.timestamp = timestamp

    def add_content(self, content: str) -> None:
        if self.text_chunks:
            self.text_chunks[0] += content
        else:
            self.text_chunks = [content]

    def serialize(self) -> dict:
        return self.__pydantic_serializer__.to_python(self, by_alias=True)  # type: ignore

    @staticmethod
    def deserialize(message_data: dict) -> "EmailMessage":
        return EmailMessage.__pydantic_validator__.validate_python(message_data)  # type: ignore
