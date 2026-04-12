from __future__ import annotations

import unittest

import torch

from model_utils import process_detection_output, process_segmentation_output


class ModelUtilsTestCase(unittest.TestCase):
    def test_process_detection_output_filters_and_sorts_predictions(self):
        prediction = [{
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [2.0, 2.0, 8.0, 8.0]]),
            "scores": torch.tensor([0.4, 0.9]),
        }]

        detections = process_detection_output(prediction, threshold=0.5)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["box"], [2, 2, 8, 8])
        self.assertAlmostEqual(detections[0]["score"], 0.9, places=5)

    def test_process_segmentation_output_returns_expected_metrics(self):
        logits = torch.full((1, 1, 4, 4), -10.0)
        logits[0, 0, :, 1] = 10.0

        metrics, mask = process_segmentation_output(
            logits,
            original_size=(4, 4),
            threshold=0.5,
            scale_param=0.25,
        )

        self.assertTrue(metrics["is_positive"])
        self.assertEqual(metrics["status"], "positive")
        self.assertEqual(metrics["area_px"], 4)
        self.assertEqual(metrics["length_px"], 4)
        self.assertAlmostEqual(metrics["area_mm"], 0.25, places=6)
        self.assertAlmostEqual(metrics["length_mm"], 1.0, places=6)
        self.assertEqual(mask.shape, (4, 4))
