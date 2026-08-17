from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms


from src.config.settings import (
    L2CS_ARCHITECTURE,
    L2CS_BIN_COUNT,
    L2CS_INPUT_HEIGHT,
    L2CS_INPUT_WIDTH,
    L2CS_MODEL_PATH,
)


@dataclass
class GazePrediction:
    """
    L2CS gaze prediction.

    Angles are stored internally in radians.

    Important:
        The official L2CS pipeline exposes the first model output
        as pitch and the second output as yaw during inference.
    """

    pitch: float
    yaw: float

    @property
    def pitch_degrees(self) -> float:
        return float(
            np.degrees(self.pitch)
        )

    @property
    def yaw_degrees(self) -> float:
        return float(
            np.degrees(self.yaw)
        )

    def as_degrees(
        self,
    ) -> tuple[float, float]:
        return (
            self.pitch_degrees,
            self.yaw_degrees,
        )


class L2CS(nn.Module):
    """
    L2CS-Net architecture compatible with the official
    ResNet-based implementation.
    """

    def __init__(
        self,
        block: type[nn.Module],
        layers: list[int],
        num_bins: int,
    ) -> None:
        super().__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(
            inplace=True
        )

        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.layer1 = self._make_layer(
            block,
            64,
            layers[0],
        )

        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
        )

        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
        )

        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
        )

        self.avgpool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.fc_yaw_gaze = nn.Linear(
            512 * block.expansion,
            num_bins,
        )

        self.fc_pitch_gaze = nn.Linear(
            512 * block.expansion,
            num_bins,
        )

        self._initialize_weights()

    def _make_layer(
        self,
        block: type[nn.Module],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:

        downsample: Optional[nn.Module] = None

        if (
            stride != 1
            or self.inplanes
            != planes * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    planes * block.expansion
                ),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
            )
        ]

        self.inplanes = (
            planes * block.expansion
        )

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                )
            )

        return nn.Sequential(
            *layers
        )

    def _initialize_weights(
        self,
    ) -> None:

        for module in self.modules():

            if isinstance(
                module,
                nn.Conv2d,
            ):
                n = (
                    module.kernel_size[0]
                    * module.kernel_size[1]
                    * module.out_channels
                )

                module.weight.data.normal_(
                    0,
                    np.sqrt(
                        2.0 / n
                    ),
                )

            elif isinstance(
                module,
                nn.BatchNorm2d,
            ):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(
            x,
            start_dim=1,
        )

        yaw_logits = (
            self.fc_yaw_gaze(x)
        )

        pitch_logits = (
            self.fc_pitch_gaze(x)
        )

        # Keep the actual model output order identical to
        # the official L2CS implementation:
        #
        #   model(x)
        #       -> yaw_logits
        #       -> pitch_logits
        #
        return (
            yaw_logits,
            pitch_logits,
        )


class L2CSNet:

    def __init__(
        self,
        model_path: str | Path = L2CS_MODEL_PATH,
        architecture: str = L2CS_ARCHITECTURE,
        num_bins: int = L2CS_BIN_COUNT,
        device: Optional[str] = None,
    ) -> None:

        self.model_path = Path(
            model_path
        )

        self.architecture = architecture

        self.num_bins = int(
            num_bins
        )

        self.device = self._resolve_device(
            device
        )

        self.model: Optional[
            L2CS
        ] = None

        self._softmax = nn.Softmax(
            dim=1
        )

        self._idx_tensor = torch.arange(
            self.num_bins,
            dtype=torch.float32,
            device=self.device,
        )

        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),

                transforms.Resize(
                    (
                        L2CS_INPUT_HEIGHT,
                        L2CS_INPUT_WIDTH,
                    )
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                ),
            ]
        )

        self._initialized = False

    # ========================================================
    # Device
    # ========================================================

    @staticmethod
    def _resolve_device(
        device: Optional[str],
    ) -> torch.device:

        if device:

            requested = torch.device(
                device
            )

            if (
                requested.type == "cuda"
                and not torch.cuda.is_available()
            ):
                return torch.device(
                    "cpu"
                )

            return requested

        if torch.cuda.is_available():
            return torch.device(
                "cuda"
            )

        return torch.device(
            "cpu"
        )

    # ========================================================
    # Architecture
    # ========================================================

    def _build_model(self) -> L2CS:

        architecture = (
            self.architecture
            .lower()
            .replace("-", "")
            .replace("_", "")
        )

        if architecture == "resnet18":

            block = (
                torchvision.models
                .resnet
                .BasicBlock
            )

            layers = [
                2,
                2,
                2,
                2,
            ]

        elif architecture == "resnet34":

            block = (
                torchvision.models
                .resnet
                .BasicBlock
            )

            layers = [
                3,
                4,
                6,
                3,
            ]

        elif architecture == "resnet101":

            block = (
                torchvision.models
                .resnet
                .Bottleneck
            )

            layers = [
                3,
                4,
                23,
                3,
            ]

        elif architecture == "resnet152":

            block = (
                torchvision.models
                .resnet
                .Bottleneck
            )

            layers = [
                3,
                8,
                36,
                3,
            ]

        else:

            block = (
                torchvision.models
                .resnet
                .Bottleneck
            )

            layers = [
                3,
                4,
                6,
                3,
            ]

        return L2CS(
            block=block,
            layers=layers,
            num_bins=self.num_bins,
        )

    # ========================================================
    # Checkpoint
    # ========================================================

    @staticmethod
    def _extract_state_dict(
        checkpoint: Any,
    ) -> dict[str, torch.Tensor]:

        if isinstance(
            checkpoint,
            dict,
        ):

            if all(
                isinstance(
                    value,
                    torch.Tensor,
                )
                for value
                in checkpoint.values()
            ):
                return checkpoint

            for key in (
                "state_dict",
                "model_state_dict",
                "model",
                "net",
            ):

                candidate = checkpoint.get(
                    key
                )

                if isinstance(
                    candidate,
                    dict,
                ):

                    if all(
                        isinstance(
                            value,
                            torch.Tensor,
                        )
                        for value
                        in candidate.values()
                    ):
                        return candidate

        raise RuntimeError(
            "Unable to extract model state_dict."
        )

    @staticmethod
    def _remove_module_prefix(
        state_dict: dict[
            str,
            torch.Tensor,
        ],
    ) -> dict[
        str,
        torch.Tensor,
    ]:

        cleaned = {}

        for key, value in (
            state_dict.items()
        ):

            if key.startswith(
                "module."
            ):
                key = key[
                    len("module.") :
                ]

            cleaned[key] = value

        return cleaned

    # ========================================================
    # Load
    # ========================================================

    def load(self) -> None:

        if self._initialized:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"L2CS model not found:\n"
                f"{self.model_path}"
            )

        model = self._build_model()

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
        )

        state_dict = (
            self._extract_state_dict(
                checkpoint
            )
        )

        state_dict = (
            self._remove_module_prefix(
                state_dict
            )
        )

        incompatible = (
            model.load_state_dict(
                state_dict,
                strict=False,
            )
        )

        # The official checkpoint contains weights that are
        # expected to match the two gaze heads.
        required_prefixes = (
            "fc_yaw_gaze",
            "fc_pitch_gaze",
        )

        missing_critical = [
            key
            for key in incompatible.missing_keys
            if key.startswith(
                required_prefixes
            )
        ]

        if missing_critical:
            raise RuntimeError(
                "The L2CS checkpoint does not match "
                "the expected gaze model.\n"
                f"Missing: {missing_critical}"
            )

        model.to(
            self.device
        )

        model.eval()

        self.model = model

        self._initialized = True

    # ========================================================
    # Input
    # ========================================================

    def _prepare_single_image(
        self,
        image: np.ndarray,
    ) -> torch.Tensor:

        if image is None:
            raise ValueError(
                "Input image cannot be None."
            )

        if not isinstance(
            image,
            np.ndarray,
        ):
            raise TypeError(
                "Input must be numpy.ndarray."
            )

        if image.size == 0:
            raise ValueError(
                "Input image is empty."
            )

        if image.ndim != 3:
            raise ValueError(
                "Input must have shape HxWxC."
            )

        if image.shape[2] != 3:
            raise ValueError(
                "Input must contain 3 channels."
            )

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        return self._transform(
            rgb
        )

    def prepare_batch(
        self,
        images: list[np.ndarray],
    ) -> torch.Tensor:

        if not images:
            raise ValueError(
                "At least one image is required."
            )

        tensors = [
            self._prepare_single_image(
                image
            )
            for image in images
        ]

        batch = torch.stack(
            tensors,
            dim=0,
        )

        return batch.to(
            self.device
        )

    # ========================================================
    # Prediction
    # ========================================================

    @torch.inference_mode()
    def predict(
        self,
        images: list[np.ndarray],
    ) -> list[GazePrediction]:

        if not self._initialized:
            self.load()

        if self.model is None:
            raise RuntimeError(
                "L2CS model is not loaded."
            )

        batch = self.prepare_batch(
            images
        )

        # Important:
        # The model itself returns:
        #
        #   yaw_logits, pitch_logits
        #
        # But the official L2CS pipeline currently
        # unpacks them as:
        #
        #   gaze_pitch, gaze_yaw = self.model(...)
        #
        # We reproduce the official inference behavior
        # here because the supplied pretrained checkpoint
        # is expected to be interpreted through that pipeline.

        model_yaw_logits, model_pitch_logits = (
            self.model(batch)
        )

        gaze_pitch_logits = (
            model_yaw_logits
        )

        gaze_yaw_logits = (
            model_pitch_logits
        )

        pitch_probabilities = (
            self._softmax(
                gaze_pitch_logits
            )
        )

        yaw_probabilities = (
            self._softmax(
                gaze_yaw_logits
            )
        )

        pitch_degrees = (
            torch.sum(
                pitch_probabilities
                * self._idx_tensor,
                dim=1,
            )
            * 4.0
            - 180.0
        )

        yaw_degrees = (
            torch.sum(
                yaw_probabilities
                * self._idx_tensor,
                dim=1,
            )
            * 4.0
            - 180.0
        )

        pitch_radians = torch.deg2rad(
            pitch_degrees
        )

        yaw_radians = torch.deg2rad(
            yaw_degrees
        )

        results: list[
            GazePrediction
        ] = []

        for pitch, yaw in zip(
            pitch_radians.cpu().numpy(),
            yaw_radians.cpu().numpy(),
        ):

            results.append(
                GazePrediction(
                    pitch=float(
                        pitch
                    ),
                    yaw=float(
                        yaw
                    ),
                )
            )

        return results

    @torch.inference_mode()
    def predict_single(
        self,
        face_crop: np.ndarray,
    ) -> GazePrediction:

        results = self.predict(
            [face_crop]
        )

        return results[0]

    # ========================================================
    # Properties
    # ========================================================

    @property
    def is_loaded(self) -> bool:
        return self._initialized

    @property
    def device_name(self) -> str:
        return str(
            self.device
        )

    # ========================================================
    # Cleanup
    # ========================================================

    def unload(self) -> None:

        self.model = None

        self._initialized = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(
        self,
    ) -> "L2CSNet":

        self.load()

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:

        self.unload()