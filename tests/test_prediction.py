"""Tests for prediction module."""
from __future__ import annotations

import numpy as np

from sahi.prediction import PredictionScore


class TestPrediction:
    """Test cases for prediction functionality."""

    def test_prediction_score(self) -> None:
        """Test PredictionScore value and comparison operations."""
        prediction_score = PredictionScore(np.array(0.6))
        assert isinstance(prediction_score.value, float)
        assert prediction_score.is_greater_than_threshold(0.5)
        assert not prediction_score.is_greater_than_threshold(0.7)
