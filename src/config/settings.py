from pathlib import Path


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
CALIBRATION_DIR = DATA_DIR / "calibration"
FACES_DIR = DATA_DIR / "faces"

MEDIAPIPE_MODELS_DIR = MODELS_DIR / "mediapipe"
L2CS_MODELS_DIR = MODELS_DIR / "l2cs"


# ============================================================
# MediaPipe
# ============================================================

FACE_LANDMARKER_MODEL = (
    MEDIAPIPE_MODELS_DIR / "face_landmarker.task"
)

FACE_LANDMARKER_NUM_FACES = 10
FACE_LANDMARKER_MIN_FACE_DETECTION_CONFIDENCE = 0.5
FACE_LANDMARKER_MIN_FACE_PRESENCE_CONFIDENCE = 0.5
FACE_LANDMARKER_MIN_TRACKING_CONFIDENCE = 0.5


# ============================================================
# L2CS-Net
# ============================================================

L2CS_MODEL_PATH = (
    L2CS_MODELS_DIR / "L2CSNet_gaze360.pkl"
)

L2CS_ARCHITECTURE = "ResNet50"

L2CS_BIN_COUNT = 90

L2CS_INPUT_WIDTH = 448
L2CS_INPUT_HEIGHT = 448


# ============================================================
# Camera / Video
# ============================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

TARGET_FPS = 30


# ============================================================
# Processing
# ============================================================

PROCESS_EVERY_N_FRAMES = 1

MAX_FACES = 10


# ============================================================
# Gaze
# ============================================================

GAZE_YAW_LIMIT = 90.0
GAZE_PITCH_LIMIT = 90.0

GAZE_SMOOTHING_WINDOW = 5


# ============================================================
# Visualization
# ============================================================

SHOW_FACE_LANDMARKS = True
SHOW_FACE_BOX = True
SHOW_GAZE_VECTOR = True
SHOW_GAZE_VALUES = True

WINDOW_NAME = "Advertisement Gaze Analytics"


# ============================================================
# Validation
# ============================================================

def validate_paths() -> None:
    """
    Validate required model files and create required
    runtime directories.
    """

    required_directories = [
        MODELS_DIR,
        DATA_DIR,
        INPUT_DIR,
        OUTPUT_DIR,
        CALIBRATION_DIR,
        FACES_DIR,
        MEDIAPIPE_MODELS_DIR,
        L2CS_MODELS_DIR,
    ]

    for directory in required_directories:
        directory.mkdir(parents=True, exist_ok=True)

    if not FACE_LANDMARKER_MODEL.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found:\n{FACE_LANDMARKER_MODEL}"
        )

    if not L2CS_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"L2CS model not found:\n{L2CS_MODEL_PATH}"
        )