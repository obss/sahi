import unittest

import numpy as np

from sahi.prediction import PredictionScore


class TestPrediction(unittest.TestCase):
    def test_prediction_score(self):

        prediction_score = PredictionScore(np.array(0.6))
        self.assertEqual(type(prediction_score.value), float)
        self.assertEqual(prediction_score.is_greater_than_threshold(0.5), True)
        self.assertEqual(prediction_score.is_greater_than_threshold(0.7), False)


if __name__ == "__main__":
    unittest.main()
