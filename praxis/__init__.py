"""Shaun — AI-native intermediate language for agentic workflows."""

from praxis.grammar import make_parser, parse
from praxis.validator import Validator, ShaunValidationError
from praxis.executor import Executor

__version__ = "0.1.0"
__all__ = ["make_parser", "parse", "Validator", "ShaunValidationError", "Executor"]
