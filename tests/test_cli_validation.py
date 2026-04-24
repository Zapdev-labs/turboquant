"""Tests for CLI model validation functions."""

from turboquant.cli import _format_suggestions, _validate_hf_model


class TestValidateHFModel:
    """Test the _validate_hf_model function."""

    def test_invalid_model_returns_false(self):
        """Test that non-existent models return False with error message."""
        is_valid, error_msg = _validate_hf_model("definitely-not-a-real-model-12345")
        assert is_valid is False
        assert "not found" in error_msg.lower() or "error" in error_msg.lower()

    def test_fake_description_model(self):
        """Test that fake description-style model names are caught."""
        fake_models = [
            "Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-GGUF",
            "GPT-4-Turbo-Reasoning-Distilled",
            "Llama-3-70B-Opus-Style",
        ]
        for model in fake_models:
            is_valid, error_msg = _validate_hf_model(model)
            assert is_valid is False, f"Model {model} should be invalid"


class TestFormatSuggestions:
    """Test the _format_suggestions function."""

    def test_claude_description_suggestion(self):
        """Test that 'claude' in model name triggers description warning."""
        model = "Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-GGUF"
        suggestions = _format_suggestions(model)
        assert "description" in suggestions.lower() or "copied" in suggestions.lower()

    def test_gguf_format_suggestion(self):
        """Test that GGUF in name triggers format suggestion."""
        model = "some-model-GGUF"
        suggestions = _format_suggestions(model)
        assert "bartowski" in suggestions.lower() or "gguf" in suggestions.lower()

    def test_missing_slash_suggestion(self):
        """Test that missing slash triggers format suggestion."""
        model = "Qwen3.5-2B"
        suggestions = _format_suggestions(model)
        assert "owner/model-name" in suggestions
        assert "Qwen/Qwen3.5-2B" in suggestions


class TestIntegration:
    """Integration tests for download command validation."""

    def test_suggestions_for_common_mistakes(self):
        """Test that common user mistakes produce helpful suggestions."""
        test_cases = [
            # (input, expected suggestion pattern)
            ("Qwen3.5-2B", "owner/model-name"),  # Missing org
            ("bartowski/llama-gguf", "bartowski"),  # GGUF hint
            ("Some-Claude-Opus-Model", "description"),  # Description vs name
        ]

        for model, expected_pattern in test_cases:
            is_valid, error_msg = _validate_hf_model(model)
            assert is_valid is False
            suggestions = _format_suggestions(model)
            assert (
                expected_pattern.lower() in suggestions.lower()
            ), f"Expected '{expected_pattern}' in suggestions for {model}"
