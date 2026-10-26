
import json
import os
import pickle
import unittest


class TestMLPipeline(unittest.TestCase):

    def test_model_exists(self):
        self.assertTrue(os.path.exists("student_result_model.pkl"))

    def test_metrics_exist(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy(self):
        with open("metrics.json") as file:
            metrics = json.load(file)

        self.assertGreaterEqual(metrics["accuracy"], 0.85)

    def test_prediction(self):
        with open("student_result_model.pkl", "rb") as file:
            model = pickle.load(file)

        result = model.predict([[90, 85, 88, 80]])

        self.assertIn(int(result[0]), [0, 1])


if __name__ == "__main__":
    unittest.main()
