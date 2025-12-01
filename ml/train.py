from pathlib import Path
import torch
from wwf.vision.timm import *
from fastai.vision.all import *
from fastai.callback.tracker import SaveModelCallback
from.config import (
    MODELS_DIR, ARCH, N_FOLDS, EPOCHS, LR
)
from .data import load_train_df, add_folds, get_dols_for_fold


def train_single_fold(fold: int) -> Path:
    print(f"==== Training fold {fold} ====")

    train_df = load_train_df()
    train_df = add_folds(train_df)

    dls = get_dols_for_fold(train_df, fold)

    learn = timm_learner(
        dls, 
        ARCH,
        metrics=accuracy,
        pretrained=True,
    )

    cbs = [
        SaveModelCallback(
            monitor='accuracy',
            fname=f'{ARCH}_fold{fold}_best'
        )
    ]

    if torch.cuda.is_available():
        learn.to_fp16()
        print("Using GPU")

    learn.fine_tune(EPOCHS, base_lr=LR, cbs=cbs)

    export_path = MODELS_DIR / f'{ARCH}_fold{fold}_best.pkl'
    learn.export(export_path)
    print(f"Exported fold {fold} model to {export_path}")

    return export_path


def train_all_folds():
    paths = []
    for fold in range(N_FOLDS):
        paths.append(train_single_fold(fold))
    print("Done training all folds: ")
    for p in paths:
        print(" -", p)

if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train_all_folds()