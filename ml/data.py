from pathlib import Path
from typing import Tuple
import pandas as pd 
from sklearn.model_selection import StratifiedKFold
from fastai.vision.all import * 
from .config import TRAIN_CSV, TARGET_COLS, IMAGE_DIR, N_FOLDS, IMG_SIZE, BATCH_SIZE


def load_train_df() -> pd.DataFrame:
    #Read train.csv, creat 'label_name', and 'label' columns.
    train_df = pd.read_csv(TRAIN_CSV)

    # One hot encode the columns so we have a single label name
    train_df['label_name'] = train_df[TARGET_COLS].idxmax(axis=1)

    label2idx = {name: i for i, name in enumerate(TARGET_COLS)}
    train_df['label'] = train_df['label_name'].map(label2idx)

    return train_df

def add_folds(train_df: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = 42) -> pd.DataFrame:
    #Add a fold column to the data, and use StratifiedKFold.
    df = train_df.copy()
    df['fold'] = -1

    skf = StratifiedKFold(n_splits = n_folds, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(skf.split(df, df['label'])):
        df.loc[val_idx, 'fold'] = fold

    return df

def get_dols_for_fold(train_df: pd.DataFrame, fold: int, img_size: int = IMG_SIZE, bs: int = BATCH_SIZE) -> DataLoaders:
    #Creates a fastai Dataloaders for a given fold.
    valid_idx = train_df.index[train_df['fold'] == fold].tolist()

    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x = ColReader("image_id", pref=IMAGES_DIR, suff=".jpg"),
        get_y = ColReader("label"),
        splitter = IndexSplitter(valid_idx),
        item_tfms=Resize(img_size),
        batch_tfms=[
            *aug_transforms(
                size=img_size,
                do_flip=True,
                flip_vert=True,
                max_rotate=20,
                max_zoom=1.2,
                max_lighting=0.3,
                p_lighting=0.9,
            ),
            Normalize.from_stats(*imagenet_stats),
        ]
    )

    return dblock.dataloaders(train_df, bs=bs)
    