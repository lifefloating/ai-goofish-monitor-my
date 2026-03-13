"""Tests for AI response JSON parser."""

import json
import pytest
from src.services.ai_response_parser import parse_ai_response_json


class TestParseAiResponseJson:
    """Tests for parse_ai_response_json."""

    def test_plain_json(self):
        result = parse_ai_response_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_in_code_fences(self):
        text = '```json\n{"result": "ok"}\n```'
        assert parse_ai_response_json(text) == {"result": "ok"}

    def test_json_with_surrounding_text(self):
        text = 'Here is the result: {"answer": 42} hope that helps'
        assert parse_ai_response_json(text) == {"answer": 42}

    def test_multiple_json_objects_returns_first(self):
        text = '{"key": "value"}\n{"key2": "value2"}'
        result = parse_ai_response_json(text)
        assert result == {"key": "value"}

    def test_json_in_code_fences_with_trailing_text(self):
        text = '```json\n{"result": "ok"}\n```\nSome explanation text'
        result = parse_ai_response_json(text)
        assert result == {"result": "ok"}

    def test_multiple_json_objects_in_code_fences(self):
        text = '```json\n{"first": 1}\n{"second": 2}\n```'
        result = parse_ai_response_json(text)
        assert result == {"first": 1}

    def test_nested_json_parses_correctly(self):
        obj = {"outer": {"inner": [1, 2, 3]}}
        result = parse_ai_response_json(json.dumps(obj))
        assert result == obj

    def test_no_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            parse_ai_response_json("no json here")

    def test_empty_braces(self):
        assert parse_ai_response_json("{}") == {}
