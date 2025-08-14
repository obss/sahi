# OBSS SAHI Tool
# Code written by Fatih C Akyon , 2025.

import numpy as np

from sahi.prediction import PredictionScore


class TestPrediction:
    def test_prediction_score(self):
        prediction_score = PredictionScore(np.array(0.6))
        assert isinstance(prediction_score.value, float)
        assert prediction_score.is_greater_than_threshold(0.5)
        assert not prediction_score.is_greater_than_threshold(0.7)
