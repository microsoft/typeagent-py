# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typeagent.knowpro.kplib import (
    Quantity,
    Facet,
    ConcreteEntity,
    ActionParam,
    Action,
    KnowledgeResponse,
)


def test_quantity():
    """Test the Quantity dataclass."""
    quantity = Quantity(amount=5.0, units="kg")
    assert quantity.amount == 5.0
    assert quantity.units == "kg"


def test_facet_repr_and_str():
    """Test the Facet class's __repr__ and __str__ methods."""
    facet = Facet(name="color", value="blue")
    assert repr(facet) == "Facet('color', 'blue')"
    assert str(facet) == "Facet('color', 'blue')"

    quantity_facet = Facet(name="weight", value=Quantity(amount=5.0, units="kg"))
    assert repr(quantity_facet) == "Facet('weight', Quantity(amount=5.0, units='kg'))"
    assert str(quantity_facet) == "Facet('weight', Quantity(amount=5.0, units='kg'))"


def test_concrete_entity():
    """Test the ConcreteEntity dataclass."""
    facets = [
        Facet(name="color", value="blue"),
        Facet(name="weight", value=Quantity(amount=5.0, units="kg")),
    ]
    entity = ConcreteEntity(
        name="ExampleEntity", type=["object", "example"], facets=facets
    )

    assert entity.name == "ExampleEntity"
    assert entity.type == ["object", "example"]
    assert entity.facets == facets
    assert (
        repr(entity)
        == "ConcreteEntity('ExampleEntity', ['object', 'example'], [Facet('color', 'blue'), Facet('weight', Quantity(amount=5.0, units='kg'))])"
    )


def test_action_param():
    """Test the ActionParam dataclass."""
    param = ActionParam(name="speed", value=100.0)
    assert param.name == "speed"
    assert param.value == 100.0


def test_action():
    """Test the Action dataclass."""
    params = [ActionParam(name="speed", value=100.0), "simple_param"]
    subject_facet = Facet(name="hobby", value="reading")
    action = Action(
        verbs=["run", "jump"],
        verb_tense="present",
        subject_entity_name="John",
        object_entity_name="Ball",
        indirect_object_entity_name="none",
        params=params,
        subject_entity_facet=subject_facet,
    )

    assert action.verbs == ["run", "jump"]
    assert action.verb_tense == "present"
    assert action.subject_entity_name == "John"
    assert action.object_entity_name == "Ball"
    assert action.indirect_object_entity_name == "none"
    assert action.params == params
    assert action.subject_entity_facet == subject_facet


def test_knowledge_response():
    """Test the KnowledgeResponse dataclass."""
    entities = [
        ConcreteEntity(
            name="John", type=["person"], facets=[Facet(name="age", value=30)]
        ),
        ConcreteEntity(name="Ball", type=["object"], facets=None),
    ]
    actions = [
        Action(
            verbs=["throw"],
            verb_tense="past",
            subject_entity_name="John",
            object_entity_name="Ball",
        )
    ]
    inverse_actions = [
        Action(
            verbs=["receive"],
            verb_tense="past",
            subject_entity_name="Ball",
            object_entity_name="John",
        )
    ]
    topics = ["sports", "games"]

    response = KnowledgeResponse(
        entities=entities,
        actions=actions,
        inverse_actions=inverse_actions,
        topics=topics,
    )

    assert response.entities == entities
    assert response.actions == actions
    assert response.inverse_actions == inverse_actions
    assert response.topics == topics
