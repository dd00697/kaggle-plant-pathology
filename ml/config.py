from pathlib import Path

# Root of the repo

ROOT_DIR = Path(__file__).resolve().parents[1]

#Data Paths
DATA_DIR = ROOT_DIR / "data" / "plant"
IMAGES_DIR = DATA_DIR / "images"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

# Data Labels
TARGET_COLS = ['healthy', 'multiple_diseases', 'rust', 'scab']

# Model Paths
MODELS_DIR = ROOT_DIR / "models"

BEST_MODEL_NAME = "tf_efficientnet_b4_ns_fold0_best.pkl"
BEST_MODEL_PATH = MODELS_DIR / BEST_MODEL_NAME

# Training Config

N_FOLDS = 5
IMG_SIZE = 380
BATCH_SIZE = 32
EPOCHS = 5
LR = 2e-3
ARCH = "tf_efficientnet_b4_ns"

