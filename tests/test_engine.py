"""Tests for Engine — uses mocking to avoid loading a real model."""

import json
from unittest.mock import MagicMock, patch

import pytest

from localagent.core.engine import Engine


class TestJsonExtraction:
    """Test the static _extract_json helper."""

    def test_plain_json(self):
        text = '{"key": "value"}'
        result = Engine._extract_json(text)
        assert result == {"key": "value"}

    def test_json_in_code_fence(self):
        text = 'Here is the result:\n```json\n{"key": "value"}\n```\nDone.'
        result = Engine._extract_json(text)
        assert result == {"key": "value"}

    def test_json_in_plain_fence(self):
        text = '```\n{"a": 1}\n```'
        result = Engine._extract_json(text)
        assert result == {"a": 1}

    def test_json_with_surrounding_text(self):
        text = 'The answer is: {"taxonomy": {"Docs": "documents"}} hope that helps!'
        result = Engine._extract_json(text)
        assert result["taxonomy"]["Docs"] == "documents"

    def test_nested_json(self):
        text = '{"outer": {"inner": {"deep": true}}}'
        result = Engine._extract_json(text)
        assert result["outer"]["inner"]["deep"] is True

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No JSON"):
            Engine._extract_json("no json here at all")

    def test_invalid_json_raises(self):
        with pytest.raises((ValueError, json.JSONDecodeError)):
            Engine._extract_json("{broken json")


class TestGenerateJson:
    """Test generate_json with mocked model calls."""

    @patch.object(Engine, "_ensure_loaded")
    @patch.object(Engine, "generate_text")
    def test_successful_json_generation(self, mock_gen, mock_load):
        engine = Engine()
        mock_gen.return_value = '```json\n{"taxonomy": {}, "assignments": {}}\n```'

        result = engine.generate_json([{"role": "user", "content": "test"}])
        assert "taxonomy" in result
        assert "assignments" in result

    @patch.object(Engine, "_ensure_loaded")
    @patch.object(Engine, "generate_text")
    def test_retry_on_bad_json(self, mock_gen, mock_load):
        engine = Engine()
        # First call returns garbage, second returns valid JSON
        mock_gen.side_effect = [
            "I'm not sure what you mean",
            '{"taxonomy": {"Misc": "misc files"}, "assignments": {}}',
        ]

        result = engine.generate_json(
            [{"role": "user", "content": "test"}], retries=1
        )
        assert result["taxonomy"]["Misc"] == "misc files"
        assert mock_gen.call_count == 2

    @patch.object(Engine, "_ensure_loaded")
    @patch.object(Engine, "generate_text")
    def test_raises_after_exhausted_retries(self, mock_gen, mock_load):
        engine = Engine()
        mock_gen.return_value = "never valid json"

        with pytest.raises(ValueError, match="Failed to get valid JSON"):
            engine.generate_json(
                [{"role": "user", "content": "test"}], retries=1
            )
