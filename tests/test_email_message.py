# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typeagent.emails.email_message import EmailMessage, EmailMessageMeta


def make_meta(
    sender: str = "Alice <alice@example.com>",
    recipients: list[str] | None = None,
    cc: list[str] | None = None,
    bcc: list[str] | None = None,
    subject: str | None = None,
) -> EmailMessageMeta:
    return EmailMessageMeta(
        sender=sender,
        recipients=recipients or [],
        cc=cc or [],
        bcc=bcc or [],
        subject=subject,
    )


class TestEmailMessageMetaProperties:
    def test_source_returns_sender(self) -> None:
        meta = make_meta(sender="bob@example.com")
        assert meta.source == "bob@example.com"

    def test_dest_returns_recipients(self) -> None:
        meta = make_meta(recipients=["a@b.com", "c@d.com"])
        assert meta.dest == ["a@b.com", "c@d.com"]

    def test_dest_empty_list(self) -> None:
        meta = make_meta(recipients=[])
        assert meta.dest == []


class TestEmailAddressToEntities:
    def test_plain_address_no_display_name(self) -> None:
        meta = make_meta()
        entities = meta._email_address_to_entities("bob@example.com")
        names = [e.name for e in entities]
        assert "bob@example.com" in names
        assert len(entities) == 1

    def test_address_with_display_name(self) -> None:
        meta = make_meta()
        entities = meta._email_address_to_entities("Alice <alice@example.com>")
        names = [e.name for e in entities]
        assert "Alice" in names
        assert "alice@example.com" in names
        assert len(entities) == 2

    def test_display_name_entity_has_email_facet(self) -> None:
        meta = make_meta()
        entities = meta._email_address_to_entities("Alice <alice@example.com>")
        person_entity = next(e for e in entities if e.name == "Alice")
        assert person_entity.facets is not None
        assert len(person_entity.facets) == 1
        assert person_entity.facets[0].name == "email_address"
        assert person_entity.facets[0].value == "alice@example.com"

    def test_display_name_only_no_address(self) -> None:
        # parseaddr("Alice") returns ("", "Alice") — treated as address only
        meta = make_meta()
        entities = meta._email_address_to_entities("Alice")
        # No display name, just the address "Alice"
        assert len(entities) == 1
        assert entities[0].name == "Alice"


class TestToEntities:
    def test_entities_include_sender(self) -> None:
        meta = make_meta(sender="Alice <alice@example.com>")
        entities = meta.to_entities()
        names = [e.name for e in entities]
        assert "Alice" in names
        assert "alice@example.com" in names

    def test_entities_include_recipient(self) -> None:
        meta = make_meta(
            sender="alice@example.com",
            recipients=["Bob <bob@example.com>"],
        )
        entities = meta.to_entities()
        names = [e.name for e in entities]
        assert "Bob" in names
        assert "bob@example.com" in names

    def test_entities_include_cc(self) -> None:
        meta = make_meta(
            sender="a@x.com",
            cc=["cc@example.com"],
        )
        entities = meta.to_entities()
        names = [e.name for e in entities]
        assert "cc@example.com" in names

    def test_entities_include_bcc(self) -> None:
        meta = make_meta(
            sender="a@x.com",
            bcc=["bcc@example.com"],
        )
        entities = meta.to_entities()
        names = [e.name for e in entities]
        assert "bcc@example.com" in names

    def test_entities_always_include_email_message_entity(self) -> None:
        meta = make_meta()
        entities = meta.to_entities()
        msg_entity = next((e for e in entities if e.name == "email"), None)
        assert msg_entity is not None
        assert "message" in msg_entity.type


class TestToTopics:
    def test_no_subject_returns_empty(self) -> None:
        meta = make_meta(subject=None)
        assert meta.to_topics() == []

    def test_subject_returned_as_topic(self) -> None:
        meta = make_meta(subject="Hello World")
        topics = meta.to_topics()
        assert topics == ["Hello World"]


class TestToActions:
    def test_no_recipients_returns_empty(self) -> None:
        meta = make_meta(sender="alice@example.com", recipients=[])
        assert meta.to_actions() == []

    def test_sent_and_received_actions_created(self) -> None:
        meta = make_meta(
            sender="Alice <alice@example.com>",
            recipients=["Bob <bob@example.com>"],
        )
        actions = meta.to_actions()
        verbs = [a.verbs[0] for a in actions]
        assert "sent" in verbs
        assert "received" in verbs

    def test_multiple_recipients_produce_actions(self) -> None:
        meta = make_meta(
            sender="alice@example.com",
            recipients=["bob@example.com", "carol@example.com"],
        )
        actions = meta.to_actions()
        assert len(actions) > 0

    def test_action_subject_is_sender(self) -> None:
        meta = make_meta(
            sender="alice@example.com",
            recipients=["bob@example.com"],
        )
        actions = meta.to_actions()
        sent_actions = [a for a in actions if "sent" in a.verbs]
        assert all(a.subject_entity_name == "alice@example.com" for a in sent_actions)


class TestGetKnowledge:
    def test_get_knowledge_returns_response(self) -> None:
        meta = make_meta(
            sender="Alice <alice@example.com>",
            recipients=["Bob <bob@example.com>"],
            subject="Test Subject",
        )
        result = meta.get_knowledge()
        assert result is not None
        assert len(result.entities) > 0
        assert len(result.topics) > 0
        assert len(result.actions) > 0


class TestEmailMessage:
    def test_basic_construction(self) -> None:
        meta = make_meta(sender="alice@example.com")
        msg = EmailMessage(
            text_chunks=["Hello world"],
            metadata=meta,
        )
        assert msg.text_chunks == ["Hello world"]
        assert msg.metadata is meta

    def test_get_knowledge_delegates_to_metadata(self) -> None:
        meta = make_meta(
            sender="Alice <alice@example.com>",
            recipients=["bob@example.com"],
            subject="Hi",
        )
        msg = EmailMessage(text_chunks=["body"], metadata=meta)
        result = msg.get_knowledge()
        assert result is not None

    def test_add_timestamp(self) -> None:
        meta = make_meta()
        msg = EmailMessage(text_chunks=["body"], metadata=meta)
        msg.add_timestamp("2025-01-01T00:00:00")
        assert msg.timestamp == "2025-01-01T00:00:00"

    def test_add_content_empty_chunks(self) -> None:
        meta = make_meta()
        msg = EmailMessage(text_chunks=[], metadata=meta)
        msg.add_content("new content")
        assert msg.text_chunks == ["new content"]

    def test_add_content_existing_chunk(self) -> None:
        meta = make_meta()
        msg = EmailMessage(text_chunks=["existing"], metadata=meta)
        msg.add_content(" more")
        assert msg.text_chunks[0] == "existing more"

    def test_serialize_roundtrip(self) -> None:
        meta = make_meta(
            sender="Alice <alice@example.com>",
            recipients=["bob@example.com"],
            subject="Hi",
        )
        msg = EmailMessage(text_chunks=["Hello"], metadata=meta, tags=["work"])
        data = msg.serialize()
        assert isinstance(data, dict)
        restored = EmailMessage.deserialize(data)
        assert restored.text_chunks == msg.text_chunks
        assert restored.metadata.sender == msg.metadata.sender
        assert restored.tags == msg.tags
