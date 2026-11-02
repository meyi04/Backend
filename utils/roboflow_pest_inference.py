import logging
from collections.abc import Mapping

from inference_sdk import InferenceConfiguration, InferenceHTTPClient

from config import Config

logger = logging.getLogger(__name__)


class RoboflowPestService:
    """Run cucumber pest detection with the configured Roboflow model."""

    _client = None

    @classmethod
    def _get_client(cls):
        if cls._client is not None:
            return cls._client

        if not Config.ROBOFLOW_API_KEY:
            raise RuntimeError("ROBOFLOW_API_KEY is not configured")

        cls._client = InferenceHTTPClient(
            api_url=Config.ROBOFLOW_API_URL,
            api_key=Config.ROBOFLOW_API_KEY,
        ).configure(InferenceConfiguration(api_key_transport="header"))
        return cls._client

    @classmethod
    def predict(cls, image_path):
        result = cls._get_client().infer(
            image_path,
            model_id=Config.ROBOFLOW_PEST_MODEL_ID,
        )
        return cls._normalize_result(result)

    @staticmethod
    def _normalize_result(result):
        if not isinstance(result, Mapping):
            raise ValueError("Roboflow returned an invalid response")

        raw_predictions = result.get("predictions", [])
        if not isinstance(raw_predictions, list):
            raise ValueError("Roboflow returned invalid predictions")

        detections = []
        for prediction in raw_predictions:
            if not isinstance(prediction, Mapping):
                continue

            class_name = str(prediction.get("class", "Unknown pest"))
            confidence = float(prediction.get("confidence", 0))
            detection = {
                "class": class_name,
                "confidence": confidence,
            }
            for coordinate in ("x", "y", "width", "height"):
                if coordinate in prediction:
                    detection[coordinate] = prediction[coordinate]
            detections.append(detection)

        detections.sort(key=lambda item: item["confidence"], reverse=True)
        pest_names = list(dict.fromkeys(item["class"] for item in detections))
        top_detection = detections[0] if detections else None
        top_pest = top_detection["class"] if top_detection else "No pest detected"
        top_confidence = top_detection["confidence"] if top_detection else 0

        return {
            "pest": top_pest,
            "confidence": top_confidence,
            "available_pests": pest_names,
            "detections": detections,
            "mode": "roboflow",
            "model_id": Config.ROBOFLOW_PEST_MODEL_ID,
            "recommendations": [],
        }
