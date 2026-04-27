import logging
import os
import time

import pandas as pd

# Configure storage path outside backend/ to prevent uvicorn reloads.
STORAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data_storage"))
MAX_FILES = 15
MAX_AGE_HOURS = 24

if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)


def get_file_path(dataset_id: int):
    return os.path.join(STORAGE_DIR, f"ds_{dataset_id}.parquet")


def get_pickle_path(dataset_id: int):
    return os.path.join(STORAGE_DIR, f"ds_{dataset_id}.pkl")


def save_dataset(dataset_id: int, df: pd.DataFrame):
    """Saves a dataframe to parquet, falling back to pickle if pyarrow is unavailable."""
    try:
        path = get_file_path(dataset_id)
        df.to_parquet(path, engine="pyarrow", compression="snappy")
        logging.info(f"Dataset {dataset_id} saved to {path}")
        return True
    except Exception as e:
        logging.warning(f"Parquet save failed for dataset {dataset_id}; using pickle fallback: {str(e)}")
        try:
            path = get_pickle_path(dataset_id)
            df.to_pickle(path)
            logging.info(f"Dataset {dataset_id} saved to {path}")
            return True
        except Exception as fallback_error:
            logging.error(f"Failed to save dataset {dataset_id}: {str(fallback_error)}")
            return False


def load_dataset(dataset_id: int) -> pd.DataFrame:
    """Loads a dataframe from parquet or pickle fallback."""
    try:
        path = get_file_path(dataset_id)
        if os.path.exists(path):
            return pd.read_parquet(path, engine="pyarrow")
    except Exception as e:
        logging.warning(f"Parquet load failed for dataset {dataset_id}; trying pickle fallback: {str(e)}")

    try:
        fallback_path = get_pickle_path(dataset_id)
        if os.path.exists(fallback_path):
            return pd.read_pickle(fallback_path)
    except Exception as fallback_error:
        logging.error(f"Failed to load dataset {dataset_id}: {str(fallback_error)}")

    return None


def cleanup_storage():
    """
    Removes files that:
    1. Are older than MAX_AGE_HOURS
    2. Exceed the MAX_FILES limit (removes oldest first)
    """
    try:
        files = []
        for f in os.listdir(STORAGE_DIR):
            if f.endswith((".parquet", ".pkl")):
                path = os.path.join(STORAGE_DIR, f)
                stat = os.stat(path)
                files.append({
                    "path": path,
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                })

        if not files:
            return

        now = time.time()
        age_limit = now - (MAX_AGE_HOURS * 3600)

        remaining_files = []
        for f in files:
            if f["mtime"] < age_limit:
                os.remove(f["path"])
                logging.info(f"Deleted old dataset: {f['path']}")
            else:
                remaining_files.append(f)

        if len(remaining_files) > MAX_FILES:
            remaining_files.sort(key=lambda x: x["mtime"], reverse=True)
            to_delete = remaining_files[MAX_FILES:]
            for f in to_delete:
                os.remove(f["path"])
                logging.info(f"Deleted dataset exceeding count limit: {f['path']}")

    except Exception as e:
        logging.error(f"Storage cleanup failed: {str(e)}")


def delete_dataset(dataset_id: int):
    """Manually delete a dataset file."""
    try:
        deleted = False
        for path in (get_file_path(dataset_id), get_pickle_path(dataset_id)):
            if os.path.exists(path):
                os.remove(path)
                deleted = True
        return deleted
    except Exception as e:
        logging.error(f"Failed to delete dataset {dataset_id}: {str(e)}")
        return False
