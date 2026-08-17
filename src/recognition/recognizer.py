from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import json
import numpy as np


try:
    import insightface
    from insightface.app import FaceAnalysis
except ImportError as exc:
    raise ImportError(
        "InsightFace is required for face recognition.\n"
        "Install it with:\n"
        "pip install insightface onnxruntime"
    ) from exc


# ============================================================
# Data Models
# ============================================================

@dataclass
class RecognitionResult:
    """
    Result of face recognition for one face.
    """

    person_id: Optional[str]

    similarity: float

    is_known: bool

    embedding: Optional[np.ndarray] = None

    @property
    def confidence(self) -> float:
        """
        Convert cosine similarity into a bounded confidence-like
        score.

        This value is NOT a calibrated probability.
        """

        if not self.is_known:
            return 0.0

        # Typical cosine similarity is [-1, 1].
        normalized = (
            self.similarity + 1.0
        ) / 2.0

        return float(
            np.clip(
                normalized,
                0.0,
                1.0,
            )
        )

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            "similarity": float(
                self.similarity
            ),
            "confidence": self.confidence,
            "is_known": self.is_known,
        }


@dataclass
class KnownPerson:
    """
    Persistent identity stored by the recognition gallery.
    """

    person_id: str

    embedding: np.ndarray

    metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            "embedding": self.embedding.tolist(),
            "metadata": self.metadata or {},
        }


# ============================================================
# Face Recognizer
# ============================================================

class FaceRecognizer:
    """
    ArcFace-based face recognition service using InsightFace.

    Responsibilities:
        - Load InsightFace recognition models
        - Extract face embeddings
        - Maintain a local identity gallery
        - Compare new embeddings against known identities
        - Return persistent Person IDs

    Important:
        Track IDs belong to the tracking subsystem.

        Person IDs belong to this recognition subsystem.

        They must never be treated as the same thing.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        similarity_threshold: float = 0.50,
        device_id: int = -1,
        gallery_path: Optional[
            str | Path
        ] = None,
    ) -> None:

        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                "similarity_threshold must be between 0 and 1."
            )

        self.model_name = model_name

        self.similarity_threshold = float(
            similarity_threshold
        )

        self.device_id = int(device_id)

        self.gallery_path = (
            Path(gallery_path)
            if gallery_path is not None
            else None
        )

        self._app: Optional[FaceAnalysis] = None

        self._gallery: dict[
            str,
            KnownPerson,
        ] = {}

        self._initialized = False

    # ========================================================
    # Initialization
    # ========================================================

    def load(self) -> None:
        """
        Load InsightFace models.

        CPU:
            device_id = -1

        GPU:
            device_id >= 0

        InsightFace uses ONNX Runtime providers internally.
        """

        if self._initialized:
            return

        providers = self._build_providers()

        self._app = FaceAnalysis(
            name=self.model_name,
            providers=providers,
            allowed_modules=[
                "detection",
                "recognition",
            ],
        )

        ctx_id = (
            self.device_id
            if self.device_id >= 0
            else -1
        )

        self._app.prepare(
            ctx_id=ctx_id,
            det_thresh=0.5,
            det_size=(640, 640),
        )

        self._initialized = True

        if self.gallery_path is not None:
            self.load_gallery(
                self.gallery_path
            )

    def _build_providers(self) -> list[str]:
        """
        Build ONNX Runtime execution provider list.

        GPU is requested only when a non-negative device ID
        is supplied.
        """

        if self.device_id >= 0:
            return [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]

        return [
            "CPUExecutionProvider",
        ]

    # ========================================================
    # Embedding Extraction
    # ========================================================

    def extract_embedding(
        self,
        face_crop: np.ndarray,
    ) -> np.ndarray:
        """
        Extract a normalized ArcFace embedding from a face crop.

        The crop is expected to be a BGR OpenCV image.
        """

        if not self._initialized:
            self.load()

        if self._app is None:
            raise RuntimeError(
                "InsightFace recognizer is not initialized."
            )

        if face_crop is None:
            raise ValueError(
                "face_crop cannot be None."
            )

        if not isinstance(
            face_crop,
            np.ndarray,
        ):
            raise TypeError(
                "face_crop must be a numpy.ndarray."
            )

        if face_crop.size == 0:
            raise ValueError(
                "face_crop cannot be empty."
            )

        if face_crop.ndim != 3:
            raise ValueError(
                "face_crop must have shape HxWxC."
            )

        if face_crop.shape[2] != 3:
            raise ValueError(
                "face_crop must have 3 channels."
            )

        faces = self._app.get(
            face_crop,
            max_num=1,
        )

        if not faces:
            raise RuntimeError(
                "No face could be recognized from the supplied crop."
            )

        face = faces[0]

        embedding = getattr(
            face,
            "normed_embedding",
            None,
        )

        if embedding is None:
            embedding = getattr(
                face,
                "embedding",
                None,
            )

        if embedding is None:
            raise RuntimeError(
                "InsightFace did not return a face embedding."
            )

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(-1)

        norm = np.linalg.norm(
            embedding
        )

        if norm <= 1e-8:
            raise RuntimeError(
                "Invalid zero-norm face embedding."
            )

        embedding /= norm

        return embedding

    # ========================================================
    # Similarity
    # ========================================================

    @staticmethod
    def cosine_similarity(
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        """

        a = np.asarray(
            embedding_a,
            dtype=np.float32,
        ).reshape(-1)

        b = np.asarray(
            embedding_b,
            dtype=np.float32,
        ).reshape(-1)

        if a.size == 0 or b.size == 0:
            raise ValueError(
                "Embeddings cannot be empty."
            )

        if a.shape != b.shape:
            raise ValueError(
                "Embedding dimensions do not match."
            )

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a <= 1e-8:
            raise ValueError(
                "First embedding has zero norm."
            )

        if norm_b <= 1e-8:
            raise ValueError(
                "Second embedding has zero norm."
            )

        return float(
            np.dot(a, b)
            / (norm_a * norm_b)
        )

    # ========================================================
    # Gallery
    # ========================================================

    def register_person(
        self,
        person_id: str,
        embedding: np.ndarray,
        metadata: Optional[dict] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Register a new known person.
        """

        if not person_id:
            raise ValueError(
                "person_id cannot be empty."
            )

        if (
            person_id in self._gallery
            and not overwrite
        ):
            raise ValueError(
                f"Person '{person_id}' already exists."
            )

        normalized = self._normalize_embedding(
            embedding
        )

        self._gallery[person_id] = (
            KnownPerson(
                person_id=person_id,
                embedding=normalized,
                metadata=metadata or {},
            )
        )

    def register_from_face(
        self,
        person_id: str,
        face_crop: np.ndarray,
        metadata: Optional[dict] = None,
        overwrite: bool = False,
    ) -> np.ndarray:
        """
        Extract an embedding from a face image and register it.
        """

        embedding = self.extract_embedding(
            face_crop
        )

        self.register_person(
            person_id=person_id,
            embedding=embedding,
            metadata=metadata,
            overwrite=overwrite,
        )

        return embedding

    def unregister_person(
        self,
        person_id: str,
    ) -> None:
        """
        Remove a person from the gallery.
        """

        self._gallery.pop(
            person_id,
            None,
        )

    def get_person(
        self,
        person_id: str,
    ) -> Optional[KnownPerson]:
        """
        Retrieve one known person.
        """

        return self._gallery.get(
            person_id
        )

    def has_person(
        self,
        person_id: str,
    ) -> bool:
        return person_id in self._gallery

    def clear_gallery(self) -> None:
        """
        Remove all known identities from memory.
        """

        self._gallery.clear()

    @property
    def gallery_size(self) -> int:
        return len(self._gallery)

    # ========================================================
    # Recognition
    # ========================================================

    def recognize_embedding(
        self,
        embedding: np.ndarray,
    ) -> RecognitionResult:
        """
        Find the closest known identity for an embedding.
        """

        if not self._gallery:
            return RecognitionResult(
                person_id=None,
                similarity=0.0,
                is_known=False,
                embedding=embedding,
            )

        query = self._normalize_embedding(
            embedding
        )

        best_person_id: Optional[
            str
        ] = None

        best_similarity = -1.0

        for person_id, person in (
            self._gallery.items()
        ):
            similarity = (
                self.cosine_similarity(
                    query,
                    person.embedding,
                )
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_person_id = person_id

        is_known = (
            best_person_id is not None
            and best_similarity
            >= self.similarity_threshold
        )

        if not is_known:
            best_person_id = None

        return RecognitionResult(
            person_id=best_person_id,
            similarity=float(
                best_similarity
            ),
            is_known=is_known,
            embedding=query,
        )

    def recognize(
        self,
        face_crop: np.ndarray,
    ) -> RecognitionResult:
        """
        Extract embedding and recognize a face.
        """

        embedding = self.extract_embedding(
            face_crop
        )

        return self.recognize_embedding(
            embedding
        )

    # ========================================================
    # Track Integration
    # ========================================================

    def recognize_track(
        self,
        face_crop: np.ndarray,
        track_id: int,
    ) -> RecognitionResult:
        """
        Recognize a tracked face.

        This method intentionally does not mutate the tracker.

        The caller receives a RecognitionResult and can then
        assign the returned Person ID to its Track object.
        """

        if track_id < 0:
            raise ValueError(
                "track_id cannot be negative."
            )

        return self.recognize(
            face_crop
        )

    # ========================================================
    # Gallery Persistence
    # ========================================================

    def save_gallery(
        self,
        path: Optional[str | Path] = None,
    ) -> None:
        """
        Save the identity gallery to JSON.

        Embeddings are stored as floating-point arrays.
        """

        target = (
            Path(path)
            if path is not None
            else self.gallery_path
        )

        if target is None:
            raise ValueError(
                "No gallery path was supplied."
            )

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "version": 1,
            "model_name": self.model_name,
            "persons": [
                person.to_dict()
                for person in self._gallery.values()
            ],
        }

        with target.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=4,
            )

    def load_gallery(
        self,
        path: str | Path,
    ) -> None:
        """
        Load an identity gallery from JSON.
        """

        path = Path(path)

        if not path.exists():
            return

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        version = int(
            data.get(
                "version",
                1,
            )
        )

        if version != 1:
            raise ValueError(
                f"Unsupported gallery version: {version}"
            )

        persons = data.get(
            "persons",
            [],
        )

        self._gallery.clear()

        for item in persons:
            person_id = item.get(
                "person_id"
            )

            embedding = item.get(
                "embedding"
            )

            metadata = item.get(
                "metadata",
                {},
            )

            if not person_id:
                continue

            if not embedding:
                continue

            self.register_person(
                person_id=person_id,
                embedding=np.asarray(
                    embedding,
                    dtype=np.float32,
                ),
                metadata=metadata,
                overwrite=True,
            )

    # ========================================================
    # Utilities
    # ========================================================

    @staticmethod
    def _normalize_embedding(
        embedding: np.ndarray,
    ) -> np.ndarray:
        """
        L2-normalize an embedding.
        """

        vector = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(-1)

        if vector.size == 0:
            raise ValueError(
                "Embedding cannot be empty."
            )

        if not np.all(
            np.isfinite(vector)
        ):
            raise ValueError(
                "Embedding contains invalid values."
            )

        norm = np.linalg.norm(
            vector
        )

        if norm <= 1e-8:
            raise ValueError(
                "Embedding has zero norm."
            )

        return vector / norm

    @property
    def is_loaded(self) -> bool:
        return self._initialized

    @property
    def device(self) -> str:
        if self.device_id >= 0:
            return f"cuda:{self.device_id}"

        return "cpu"

    # ========================================================
    # Resource Management
    # ========================================================

    def close(self) -> None:
        """
        Release InsightFace resources.
        """

        self._app = None
        self._initialized = False

    def __enter__(
        self,
    ) -> "FaceRecognizer":
        self.load()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()