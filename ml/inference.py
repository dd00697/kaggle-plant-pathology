# ml/inference.py
import io
from typing import Dict, Any

import torch
import pathlib
from fastai.vision.all import load_learner, PILImage
from .config import BEST_MODEL_PATH


if isinstance(pathlib.Path(), pathlib.WindowsPath):
    pathlib.PosixPath = pathlib.WindowsPath  # type: ignore

print(f"Loading model from {BEST_MODEL_PATH}")
learn = load_learner(BEST_MODEL_PATH)
learn.model.eval()

if torch.cuda.is_available():
    device = torch.device("cuda")
    learn.dls = learn.dls.cuda()
    learn.model.to(device)
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


def predict_image_bytes(image_bytes: bytes) -> Dict[str, Any]:
    """
    Take raw image bytes and return:
      - predicted label
      - predicted index
      - probabilities per class
    """
    img = PILImage.create(io.BytesIO(image_bytes))

    pred_class, pred_idx, probs = learn.predict(img)

    classes = list(map(str, learn.dls.vocab))
    probs_list = [float(p) for p in probs]
    prob_dict = dict(zip(classes, probs_list))

    return {
        "predicted_label": str(pred_class),
        "predicted_index": int(pred_idx),
        "probabilities": prob_dict,
        "device": DEVICE,
    }
