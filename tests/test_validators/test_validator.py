import pytest

from theseus.validators._validator import Validator


def test_validator_instantiation():
    with pytest.raises(TypeError):
        Validator()
