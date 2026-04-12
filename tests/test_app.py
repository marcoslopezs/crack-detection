from __future__ import annotations

import io
import unittest

import torch
from fastapi.testclient import TestClient
from PIL import Image

import app as api_module


class DummyDetectionModel:
    def __call__(self, image_tensor):
        return [{
            "boxes": torch.tensor([[5.0, 5.0, 25.0, 25.0]]),
            "scores": torch.tensor([0.9]),
        }]


class DummySegmentationModel:
    def __call__(self, image_tensor):
        height, width = image_tensor.shape[-2:]
        output = torch.full((1, 1, height, width), -10.0)
        output[:, :, :, width // 2] = 10.0
        return output


def build_test_image_bytes():
    image = Image.new("RGB", (64, 64), color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        api_module.models.clear()
        api_module.models.update({
            "faster_rcnn": DummyDetectionModel(),
            "unetpp": DummySegmentationModel(),
        })
        self.client = TestClient(api_module.app)
        self.image_bytes = build_test_image_bytes()

    def test_detection_endpoint_returns_detection_metadata(self):
        response = self.client.post(
            "/predict/detection?threshold=0.5",
            files={"file": ("sample.png", self.image_bytes, "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["is_positive"])
        self.assertEqual(payload["detection_count"], 1)
        self.assertEqual(payload["status"], "positive")

    def test_detection_image_endpoint_returns_jpeg(self):
        response = self.client.post(
            "/predict/detection/image?threshold=0.5",
            files={"file": ("sample.png", self.image_bytes, "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertGreater(len(response.content), 0)

    def test_segmentation_endpoint_returns_metrics(self):
        response = self.client.post(
            "/predict/segmentation?threshold=0.5&scale_param=0.25",
            files={"file": ("sample.png", self.image_bytes, "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["is_positive"])
        self.assertGreater(payload["area_px"], 0)
        self.assertGreater(payload["length_px"], 0)
        self.assertIn("area_mm", payload)
        self.assertIn("length_mm", payload)

    def test_segmentation_image_endpoint_returns_jpeg(self):
        response = self.client.post(
            "/predict/segmentation/image?threshold=0.5",
            files={"file": ("sample.png", self.image_bytes, "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertGreater(len(response.content), 0)
