# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import unittest

from decision_rules.conditions import AttributesCondition
from decision_rules.conditions import CompoundCondition
from decision_rules.conditions import ElementaryCondition
from decision_rules.conditions import LogicOperators
from decision_rules.conditions import NominalCondition
from decision_rules.serialization import JSONSerializer


class TestNominalConditionSerializer(unittest.TestCase):

    def test_serializing_deserializing(self):
        condition = NominalCondition(
            column_index=2,
            value='value',
        )
        condition.negated = True

        serializer_cond = JSONSerializer.serialize(condition)
        deserializer_cond = JSONSerializer.deserialize(
            serializer_cond,
            NominalCondition
        )

        self.assertEqual(
            condition, deserializer_cond,
            'Serializing and deserializing should lead to the the same object'
        )


class TestElementaryConditionSerializer(unittest.TestCase):

    def test_serializing_deserializing(self):
        condition = ElementaryCondition(
            column_index=2, left=-1, right=2.0, left_closed=True, right_closed=False
        )
        condition.negated = True

        serializer_cond = JSONSerializer.serialize(condition)
        deserializer_cond = JSONSerializer.deserialize(
            serializer_cond,
            ElementaryCondition
        )

        self.assertEqual(
            condition, deserializer_cond,
            'Serializing and deserializing should lead to the the same object'
        )


class TestAttributesConditionSerializer(unittest.TestCase):

    def test_serializing_deserializing(self):
        condition = AttributesCondition(
            column_left=2, column_right=3, operator='>'
        )
        condition.negated = True

        serializer_cond = JSONSerializer.serialize(condition)
        deserializer_cond = JSONSerializer.deserialize(
            serializer_cond,
            AttributesCondition
        )

        self.assertEqual(
            condition, deserializer_cond,
            'Serializing and deserializing should lead to the the same object'
        )


class TestCompoundConditionSerializer(unittest.TestCase):

    def test_serializing_deserializing(self):
        condition = CompoundCondition(
            subconditions=[
                AttributesCondition(
                    column_left=2, column_right=3, operator='>'
                ),
                ElementaryCondition(
                    column_index=2, left=-1, right=2.0, left_closed=True, right_closed=False
                ),
                NominalCondition(
                    column_index=2,
                    value='value',
                )
            ],
            logic_operator=LogicOperators.ALTERNATIVE
        )

        serializer_cond = JSONSerializer.serialize(condition)
        deserializer_cond = JSONSerializer.deserialize(
            serializer_cond,
            CompoundCondition
        )

        self.assertEqual(
            condition, deserializer_cond,
            'Serializing and deserializing should lead to the the same object'
        )


if __name__ == '__main__':
    unittest.main()
