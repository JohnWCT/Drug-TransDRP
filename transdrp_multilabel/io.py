import os
import shutil
import pandas as pd
import json
import pickle
from typing import Any

def read_csv(path: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)

def write_csv(df: pd.DataFrame, path: str, **kwargs) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False, **kwargs)

def write_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)

def write_pickle(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def read_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)

def ensure_clean_dir(path: str, overwrite: bool = True) -> None:
    if os.path.exists(path):
        if overwrite:
            shutil.rmtree(path)
        else:
            raise FileExistsError(f"Directory {path} already exists and overwrite is set to False.")
    os.makedirs(path, exist_ok=True)
