"""Gemma-4 multimodal detection model wrapper for SAHI.

Provides integration with Google's Gemma-4 multimodal large language models for
prompt-driven open-vocabulary object detection. Gemma-4 returns bounding boxes
as JSON in a ``[y_min, x_min, y_max, x_max]`` layout normalized to a ``1000x1000``
canvas, which is converted to SAHI ``ObjectPrediction`` instances here.

Reference:
    - https://huggingface.co/google/gemma-4-E4B-it
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import numpy as np
from PIL import Image

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.import_utils import ensure_package_minimum_version

# Gemma-4 returns bounding box coordinates normalized to a 1000x1000 canvas
# regardless of the original image size. The notebook resizes the input to the
# nearest multiple of 48 before inference for best results.
_GEMMA4_COORD_RANGE = 1000
_GEMMA4_PATCH_MULTIPLE = 48
_DEFAULT_MODEL_ID = "google/gemma-4-E4B-it"


class Gemma4DetectionModel(DetectionModel):
    """Gemma-4 multimodal object detection model.

    Gemma-4 is a promptable vision-language model that returns bounding boxes
    as JSON. This wrapper drives the model with a text query listing the
    classes to detect, parses the JSON response, and converts each box to a
    ``ObjectPrediction`` in the original image's coordinate system.

    Args:
        model_path: HuggingFace repo id (default: ``google/gemma-4-E4B-it``).
        model: Optional pre-loaded ``AutoModelForMultimodalLM`` instance.
        processor: Optional pre-loaded ``AutoProcessor`` instance.
        classes: Class names to detect (required before calling ``perform_inference``).
            May also be set later via :py:meth:`set_classes`.
        max_new_tokens: Maximum number of new tokens to generate per query.
        token: HuggingFace access token; falls back to ``HF_TOKEN`` env var.

    !!! example "Usage"
        ```python
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction

        detection_model = AutoDetectionModel.from_pretrained(
            model_type="gemma4",
            model_path="google/gemma-4-E4B-it",
            classes=["person", "car", "bicycle"],
            confidence_threshold=0.3,
            device="cuda:0",
        )

        result = get_sliced_prediction("image.jpg", detection_model)
        ```
    """

    def __init__(
        self,
        model_path: str | None = None,
        model: object | None = None,
        processor: object | None = None,
        config_path: str | None = None,
        device: str | None = None,
        mask_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        category_mapping: dict | None = None,
        category_remapping: dict | None = None,
        load_at_init: bool = True,
        image_size: int | None = None,
        classes: list[str] | None = None,
        max_new_tokens: int = 1024,
        token: str | None = None,
        torch_dtype: str | Any = None,
        quantization: str | None = None,
        verbose: bool = False,
        instruction: str | None = None,
    ) -> None:
        """Initialize Gemma-4 detection model.

        Args:
            torch_dtype: Weight dtype passed to ``from_pretrained``. Accepts a
                ``torch.dtype`` or a string such as ``"float16"``, ``"bfloat16"``,
                ``"float32"``, or ``"auto"``. Halving weights is the cheapest way
                to fit Gemma-4 on a mid-range GPU.
            quantization: Optional on-the-fly quantization. Pass ``"4bit"`` or
                ``"8bit"`` to load with ``bitsandbytes`` (install separately).
            verbose: If True, log the raw generated text and per-detection
                filter reasons — useful when ``perform_inference`` returns
                zero detections and you want to see what the model actually said.
            instruction: Optional prompt text that replaces the default
                "Detect every instance of ..." instruction. The JSON schema
                ``[{box_2d, label}, ...]`` is always appended, so focus this
                string on scene context and detection intent (e.g. "This is an
                aerial parking lot; find every car even if partially occluded").
                May also be set/updated later via :py:meth:`set_instruction`.
        """
        self._processor = processor
        self._token = token
        self._classes: list[str] = list(classes) if classes else []
        self._max_new_tokens = max_new_tokens
        self._torch_dtype = torch_dtype
        self._quantization = quantization
        self._verbose = verbose
        self._instruction = instruction
        self._original_shapes: list[tuple[int, ...]] = []
        existing_packages = getattr(self, "required_packages", None) or []
        self.required_packages = [*list(existing_packages), "torch", "transformers"]
        ensure_package_minimum_version("transformers", "4.42.0")
        super().__init__(
            model_path,
            model,
            config_path,
            device,
            mask_threshold,
            confidence_threshold,
            category_mapping,
            category_remapping,
            load_at_init,
            image_size,
        )

    @property
    def processor(self) -> Any:
        """Return the multimodal processor."""
        return self._processor

    @property
    def classes(self) -> list[str]:
        """Return the list of prompted class names."""
        return list(self._classes)

    @property
    def num_categories(self) -> int:
        """Return number of prompted classes."""
        return len(self._classes)

    def set_classes(self, classes: list[str]) -> None:
        """Set the classes to detect and rebuild the category mapping."""
        if not classes:
            raise ValueError("classes must be a non-empty list of class names.")
        self._classes = list(classes)
        self.category_mapping = {i: name for i, name in enumerate(self._classes)}

    def set_instruction(self, instruction: str | None) -> None:
        """Override (or clear) the prompt instruction used for detection.

        Pass ``None`` to restore the default instruction built from ``classes``.
        The JSON schema is appended automatically; your text should describe
        scene context and what to detect.
        """
        self._instruction = instruction

    def load_model(self) -> None:
        """Load the Gemma-4 model and processor from HuggingFace."""
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        hf_token = os.getenv("HF_TOKEN", self._token)
        model_source = self.model_path or _DEFAULT_MODEL_ID

        kwargs: dict[str, Any] = {
            "device_map": self.device,
            "token": hf_token,
            "low_cpu_mem_usage": True,
        }
        dtype = self._resolve_torch_dtype(self._torch_dtype)
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        quant_config = self._build_quantization_config(self._quantization)
        if quant_config is not None:
            kwargs["quantization_config"] = quant_config
            # bitsandbytes handles device placement itself; device_map must be "auto".
            kwargs["device_map"] = "auto"

        model = AutoModelForMultimodalLM.from_pretrained(model_source, **kwargs)
        processor = AutoProcessor.from_pretrained(model_source, token=hf_token)
        self.set_model(model, processor)

    @staticmethod
    def _resolve_torch_dtype(value: Any) -> Any:
        """Convert a string dtype (``"float16"``) to ``torch.dtype``; pass through otherwise."""
        if value is None:
            return None
        if isinstance(value, str):
            import torch

            alias = {"fp16": "float16", "half": "float16", "bf16": "bfloat16", "fp32": "float32"}
            name = alias.get(value, value)
            if name == "auto":
                return "auto"
            dtype = getattr(torch, name, None)
            if dtype is None:
                raise ValueError(f"Unknown torch_dtype string: {value!r}")
            return dtype
        return value

    @staticmethod
    def _build_quantization_config(spec: str | None) -> Any:
        """Build a ``BitsAndBytesConfig`` from a short spec like ``"4bit"``."""
        if spec is None:
            return None
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            raise ImportError(
                "quantization requires transformers with bitsandbytes support: pip install bitsandbytes"
            ) from e
        spec = spec.lower()
        if spec in ("4bit", "nf4"):
            import torch

            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        if spec == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        raise ValueError(f"Unknown quantization spec: {spec!r} (expected '4bit' or '8bit')")

    def set_model(self, model: Any, processor: Any | None = None, **kwargs: Any) -> None:
        """Assign a pre-loaded model and processor."""
        processor = processor or self.processor
        if processor is None:
            raise ValueError(f"'processor' is required to be set, got {processor}.")
        self.model = model
        self._processor = processor
        if self._classes and not self.category_mapping:
            self.category_mapping = {i: name for i, name in enumerate(self._classes)}

    def _build_messages(self, image: Image.Image) -> list[dict]:
        """Build the chat template messages for detection."""
        schema_suffix = (
            " Return a JSON array where each item is "
            '{"box_2d": [y_min, x_min, y_max, x_max], "label": "..."} '
            "with coordinates normalized to 0-1000 and label matching one of "
            f"the requested classes ({', '.join(self._classes)}). Return only the JSON."
        )
        if self._instruction:
            instruction = self._instruction.rstrip() + schema_suffix
        elif len(self._classes) == 1:
            instruction = f"Detect every {self._classes[0]} in the image." + schema_suffix
        else:
            class_list = ", ".join(self._classes)
            instruction = f"Detect every instance of the following classes in the image: {class_list}." + schema_suffix
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            }
        ]

    @staticmethod
    def _resize_to_patch_multiple(image: Image.Image) -> Image.Image:
        """Crop image so its dimensions are multiples of the model patch size."""
        w, h = image.size
        new_w = max(_GEMMA4_PATCH_MULTIPLE, (w // _GEMMA4_PATCH_MULTIPLE) * _GEMMA4_PATCH_MULTIPLE)
        new_h = max(_GEMMA4_PATCH_MULTIPLE, (h // _GEMMA4_PATCH_MULTIPLE) * _GEMMA4_PATCH_MULTIPLE)
        if (new_w, new_h) == (w, h):
            return image
        return image.crop((0, 0, new_w, new_h))

    @staticmethod
    def _extract_json(text: str) -> list[dict]:
        """Extract a JSON array of detections from the model's generated text."""
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
            if not match:
                return []
            try:
                parsed = json.loads(match.group(1))
            except json.JSONDecodeError:
                return []
        if isinstance(parsed, dict):
            parsed = [parsed]
        return [item for item in parsed if isinstance(item, dict)]

    def _generate_detections(self, image_pil: Image.Image) -> list[dict]:
        """Run the model once and return parsed detections."""
        import torch

        messages = self._build_messages(image_pil)
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.model.device)

        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self._max_new_tokens, do_sample=False)
        input_len = inputs["input_ids"].shape[-1]
        generated_text = self._processor.decode(output[0, input_len:], skip_special_tokens=True)

        # Free per-call activations/KV so sliced inference doesn't fragment VRAM.
        del inputs, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        parsed = self._extract_json(generated_text)
        if self._verbose:
            from sahi.logger import logger

            preview = generated_text.strip()
            if len(preview) > 800:
                preview = preview[:800] + "... [truncated]"
            logger.info(f"[gemma4] raw response ({len(generated_text)} chars): {preview}")
            logger.info(f"[gemma4] parsed {len(parsed)} raw detections before score/label filtering")
        return parsed

    def perform_inference(self, image: list | np.ndarray) -> None:
        """Run Gemma-4 on a single image or a list of images.

        Args:
            image: ``np.ndarray`` (H, W, C) in RGB order, or a list of such arrays.
        """
        if self.model is None or self._processor is None:
            raise RuntimeError("Model is not loaded, load it by calling .load_model()")
        if not self._classes:
            raise RuntimeError("No classes set. Pass classes=[...] at construction time or call set_classes(...).")

        images = image if isinstance(image, list) else [image]
        self._original_shapes = [img.shape for img in images]

        predictions: list[list[dict]] = []
        for img in images:
            pil = Image.fromarray(img) if isinstance(img, np.ndarray) else img
            pil = pil.convert("RGB") if pil.mode != "RGB" else pil
            pil = self._resize_to_patch_multiple(pil)
            predictions.append(self._generate_detections(pil))
        self._original_predictions = predictions

    def perform_batch_inference(self, images: list[np.ndarray]) -> None:
        """Run inference over a batch of images (sequential under the hood)."""
        self.perform_inference(images)

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: list[list[int | float]] | None = [[0, 0]],
        full_shape_list: list[list[int | float]] | None = None,
    ) -> None:
        """Convert Gemma-4 raw JSON detections to SAHI ObjectPrediction lists."""
        assert self._original_predictions is not None
        original_predictions: list[list[dict]] = self._original_predictions

        shift_amount_list_typed = fix_shift_amount_list(shift_amount_list)
        full_shape_list_typed = fix_full_shape_list(full_shape_list)

        label_to_id = {name.lower(): i for i, name in enumerate(self._classes)}
        dropped_labels: dict[str, int] = {}

        object_prediction_list_per_image: list[list[ObjectPrediction]] = []
        for image_ind, detections in enumerate(original_predictions):
            image_height, image_width, _ = self._original_shapes[image_ind]
            shift_amount = [int(x) for x in shift_amount_list_typed[image_ind]]
            full_shape = None if full_shape_list_typed is None else [int(x) for x in full_shape_list_typed[image_ind]]

            object_prediction_list: list[ObjectPrediction] = []
            for det in detections:
                box = det.get("box_2d") or det.get("bbox")
                if not box or len(box) != 4:
                    continue
                label = str(det.get("label", self._classes[0])).strip()
                score = float(det.get("score", det.get("confidence", 1.0)))
                if score < self.confidence_threshold:
                    continue

                category_id = self._match_label(label, label_to_id)
                if category_id is None:
                    dropped_labels[label] = dropped_labels.get(label, 0) + 1
                    continue

                ymin, xmin, ymax, xmax = (float(v) for v in box)
                xmin = (xmin / _GEMMA4_COORD_RANGE) * image_width
                xmax = (xmax / _GEMMA4_COORD_RANGE) * image_width
                ymin = (ymin / _GEMMA4_COORD_RANGE) * image_height
                ymax = (ymax / _GEMMA4_COORD_RANGE) * image_height
                if xmax <= xmin or ymax <= ymin:
                    continue

                bbox = [
                    max(0.0, min(xmin, image_width)),
                    max(0.0, min(ymin, image_height)),
                    max(0.0, min(xmax, image_width)),
                    max(0.0, min(ymax, image_height)),
                ]

                object_prediction_list.append(
                    ObjectPrediction(
                        bbox=bbox,
                        segmentation=None,
                        category_id=category_id,
                        category_name=self._classes[category_id],
                        shift_amount=shift_amount,
                        score=score,
                        full_shape=full_shape,
                    )
                )
            object_prediction_list_per_image.append(object_prediction_list)

        if self._verbose and dropped_labels:
            from sahi.logger import logger

            logger.info(f"[gemma4] dropped labels not in classes={self._classes}: {dropped_labels}")

        self._object_prediction_list_per_image = object_prediction_list_per_image

    @staticmethod
    def _match_label(label: str, label_to_id: dict[str, int]) -> int | None:
        """Match a model-emitted label to one of the prompted class indices.

        Tolerates common variants: case, whitespace, trailing plurals, and
        simple substring overlap (e.g. ``"sports car"`` → ``"car"``).
        """
        if not label:
            return None
        key = label.strip().lower()
        if key in label_to_id:
            return label_to_id[key]
        # plural → singular
        if key.endswith("es") and key[:-2] in label_to_id:
            return label_to_id[key[:-2]]
        if key.endswith("s") and key[:-1] in label_to_id:
            return label_to_id[key[:-1]]
        # Whole-word token match. Among classes whose tokens are all present
        # in the label, prefer the most specific one (largest token set).
        key_tokens = set(key.replace("-", " ").replace("_", " ").split())
        best: tuple[int, int] | None = None
        for class_name, idx in label_to_id.items():
            class_tokens = set(class_name.split())
            if class_tokens and class_tokens.issubset(key_tokens):
                size = len(class_tokens)
                if best is None or size > best[0]:
                    best = (size, idx)
        return best[1] if best else None
