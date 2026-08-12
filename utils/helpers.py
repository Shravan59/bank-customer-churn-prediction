"""
Utility helper functions for the Bank Customer Churn Prediction project.

This module provides reusable utilities for file handling, logging,
data validation, serialization support, formatting, and project-level
operations. It intentionally contains no machine learning, exploratory
data analysis, or prediction logic.

Author
------
Shravan Pandey

Project
-------
Predictive Modeling and Risk Scoring for Bank Customer Churn
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import joblib
import pandas as pd

from config.config import (
    LOG_FORMAT,
    LOG_LEVEL,
    LOG_FILE,
)

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

_LOGGER_INITIALIZED = False


def setup_logger(name: str | None = None) -> logging.Logger:
    """
    Configure and return a project logger.

    Parameters
    ----------
    name : str | None, default=None
        Logger name.

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    global _LOGGER_INITIALIZED

    logger = logging.getLogger(name)

    if not _LOGGER_INITIALIZED:
        log_path = Path(LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        formatter = logging.Formatter(LOG_FORMAT)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logging.basicConfig(
            level=LOG_LEVEL,
            handlers=[file_handler, stream_handler],
            force=True,
        )

        _LOGGER_INITIALIZED = True

    logger.setLevel(LOG_LEVEL)
    return logger


logger = setup_logger(__name__)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

SUPPORTED_PICKLE_EXTENSIONS = {".pkl", ".joblib"}
SUPPORTED_JSON_EXTENSION = ".json"
SUPPORTED_CSV_EXTENSION = ".csv"

# ---------------------------------------------------------------------
# File Utilities
# ---------------------------------------------------------------------


def ensure_directory(directory: str | Path) -> Path:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    directory : str or Path
        Directory path.

    Returns
    -------
    Path
        Resolved directory path.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def file_exists(file_path: str | Path) -> bool:
    """
    Check whether a file exists.

    Parameters
    ----------
    file_path : str or Path
        File path.

    Returns
    -------
    bool
        True if file exists.
    """
    return Path(file_path).is_file()


def read_csv(
    file_path: str | Path,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Read a CSV file safely.

    Parameters
    ----------
    file_path : str or Path
        CSV file path.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValueError
        If CSV is empty.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    if path.suffix.lower() != SUPPORTED_CSV_EXTENSION:
        raise ValueError(f"Unsupported CSV file: {path}")

    logger.info("Reading CSV: %s", path)

    dataframe = pd.read_csv(path, **kwargs)

    if dataframe.empty:
        raise ValueError(f"CSV file contains no records: {path}")

    return dataframe


def write_csv(
    dataframe: pd.DataFrame,
    file_path: str | Path,
    *,
    index: bool = False,
    **kwargs: Any,
) -> Path:
    """
    Save DataFrame to CSV.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame to save.

    file_path : str or Path
        Destination path.

    index : bool, default=False
        Save index.

    Returns
    -------
    Path
        Saved file path.
    """
    validate_dataframe(dataframe)

    path = Path(file_path)
    ensure_directory(path.parent)

    dataframe.to_csv(path, index=index, **kwargs)

    logger.info("CSV written: %s", path)

    return path.resolve()


# ---------------------------------------------------------------------
# DataFrame Validation
# ---------------------------------------------------------------------


def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str],
) -> None:
    """
    Validate required DataFrame columns.

    Parameters
    ----------
    dataframe : pandas.DataFrame

    required_columns : Iterable[str]

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    missing = sorted(
        set(required_columns) - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def validate_dataframe(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str] | None = None,
) -> None:
    """
    Validate DataFrame integrity.

    Parameters
    ----------
    dataframe : pandas.DataFrame

    required_columns : iterable of str, optional

    Raises
    ------
    TypeError
        Invalid object.

    ValueError
        Validation failure.
    """
    if dataframe is None:
        raise ValueError("DataFrame is None.")

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Expected pandas.DataFrame."
        )

    if dataframe.empty:
        raise ValueError(
            "DataFrame is empty."
        )

    duplicated_columns = dataframe.columns[
        dataframe.columns.duplicated()
    ]

    if len(duplicated_columns):
        raise ValueError(
            "Duplicate columns detected: "
            f"{duplicated_columns.tolist()}"
        )

    duplicated_rows = dataframe.duplicated().sum()

    if duplicated_rows:
        raise ValueError(
            f"Duplicate rows detected: {duplicated_rows}"
        )

    if required_columns is not None:
        validate_columns(
            dataframe,
            required_columns,
        )

    logger.info(
        "Validated DataFrame successfully "
        "(rows=%d, cols=%d)",
        dataframe.shape[0],
        dataframe.shape[1],
    )

# ---------------------------------------------------------------------
# Serialization Utilities
# ---------------------------------------------------------------------

import json
from datetime import datetime
from functools import wraps
from time import perf_counter


def load_pickle(file_path: str | Path) -> Any:
    """
    Load a Joblib/Pickle artifact safely.

    Parameters
    ----------
    file_path : str or Path
        Artifact path.

    Returns
    -------
    Any
        Loaded Python object.

    Raises
    ------
    FileNotFoundError
        If artifact does not exist.

    RuntimeError
        If artifact cannot be loaded.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")

    if path.suffix.lower() not in SUPPORTED_PICKLE_EXTENSIONS:
        raise ValueError(f"Unsupported artifact format: {path.suffix}")

    try:
        logger.info("Loading artifact: %s", path)
        return joblib.load(path)
    except Exception as exc:
        logger.exception("Failed to load artifact.")
        raise RuntimeError(
            f"Unable to load artifact: {path}"
        ) from exc


def save_pickle(
    obj: Any,
    file_path: str | Path,
    *,
    compress: int = 3,
) -> Path:
    """
    Save an object using Joblib.

    Parameters
    ----------
    obj : Any
        Python object.

    file_path : str or Path
        Destination path.

    compress : int, default=3
        Joblib compression level.

    Returns
    -------
    Path
        Saved artifact path.
    """
    if obj is None:
        raise ValueError("Cannot serialize None object.")

    path = Path(file_path)

    ensure_directory(path.parent)

    if path.exists():
        logger.warning("Replacing existing artifact: %s", path)

    joblib.dump(
        obj,
        path,
        compress=compress,
    )

    logger.info("Artifact saved successfully: %s", path)

    return path.resolve()


def load_json(file_path: str | Path) -> dict[str, Any]:
    """
    Load a JSON file.

    Parameters
    ----------
    file_path : str or Path

    Returns
    -------
    dict
        Parsed JSON object.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_json(
    data: dict[str, Any],
    file_path: str | Path,
    *,
    indent: int = 4,
) -> Path:
    """
    Save dictionary to JSON.

    Parameters
    ----------
    data : dict

    file_path : str or Path

    indent : int
        JSON indentation.

    Returns
    -------
    Path
    """
    path = Path(file_path)

    ensure_directory(path.parent)

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=indent,
            ensure_ascii=False,
        )

    logger.info("JSON saved: %s", path)

    return path.resolve()


# ---------------------------------------------------------------------
# Formatting Utilities
# ---------------------------------------------------------------------


def format_metric(value: float | int) -> str:
    """
    Convert numeric values into human-readable format.

    Examples
    --------
    1200 -> 1.2K
    3200000 -> 3.2M
    """
    abs_value = abs(float(value))

    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"

    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"

    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"

    return f"{value}"


def calculate_file_size(file_path: str | Path) -> str:
    """
    Return human-readable file size.

    Parameters
    ----------
    file_path : str or Path

    Returns
    -------
    str
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(path)

    size = path.stat().st_size

    units = [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]

    index = 0

    while size >= 1024 and index < len(units) - 1:
        size /= 1024
        index += 1

    return f"{size:.2f} {units[index]}"


# ---------------------------------------------------------------------
# Timestamp Utilities
# ---------------------------------------------------------------------


def get_timestamp() -> str:
    """
    Return current timestamp.

    Returns
    -------
    str
        ISO formatted timestamp.
    """
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------
# Performance Utilities
# ---------------------------------------------------------------------


def execution_time(func):
    """
    Decorator that logs execution time.

    Parameters
    ----------
    func : Callable

    Returns
    -------
    Callable
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()

        try:
            return func(*args, **kwargs)

        finally:
            elapsed = perf_counter() - start

            logger.info(
                "%s executed in %.4f seconds",
                func.__name__,
                elapsed,
            )

    return wrapper


# ---------------------------------------------------------------------
# Memory Utilities
# ---------------------------------------------------------------------


def memory_usage(dataframe: pd.DataFrame) -> str:
    """
    Return DataFrame memory usage.

    Parameters
    ----------
    dataframe : pandas.DataFrame

    Returns
    -------
    str
        Human-readable memory size.
    """
    validate_dataframe(dataframe)

    memory = dataframe.memory_usage(
        deep=True,
    ).sum()

    units = [
        "Bytes",
        "KB",
        "MB",
        "GB",
    ]

    value = float(memory)

    index = 0

    while value >= 1024 and index < len(units) - 1:
        value /= 1024
        index += 1

    return f"{value:.2f} {units[index]}"

# ---------------------------------------------------------------------
# Prediction & Model Validation Utilities
# ---------------------------------------------------------------------

from numbers import Integral
from random import Random

from config.config import (
    PROJECT_AUTHOR,
    PROJECT_DESCRIPTION,
    PROJECT_NAME,
    PROJECT_VERSION,
    MODEL_DIR,
    TARGET_COLUMN,
    RANDOM_STATE,
    DATA_DIR,
    MODEL_VISUALS_DIR,
    EDA_VISUALS_DIR,
    DASHBOARD_VISUALS_DIR,
)


def validate_binary_target(
    dataframe: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> None:
    """
    Validate that the target column contains only binary values.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input dataframe.

    target_column : str, default=TARGET_COLUMN
        Name of target column.

    Raises
    ------
    ValueError
        If the target column is missing or contains
        values other than 0 and 1.
    """
    validate_dataframe(dataframe, [target_column])

    unique_values = set(dataframe[target_column].dropna().unique())

    if not unique_values.issubset({0, 1}):
        raise ValueError(
            f"Target column '{target_column}' must contain only "
            f"binary values {{0, 1}}. Found: {sorted(unique_values)}"
        )

    logger.info("Binary target validation passed.")


def validate_prediction_input(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str],
) -> None:
    """
    Validate prediction input.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Prediction input.

    required_columns : Iterable[str]
        Expected feature columns.

    Raises
    ------
    ValueError
        If validation fails.
    """
    validate_dataframe(dataframe, required_columns)

    if dataframe.isna().any().any():
        raise ValueError(
            "Prediction input contains missing values."
        )

    unexpected = sorted(
        set(dataframe.columns) - set(required_columns)
    )

    if unexpected:
        logger.warning(
            "Unexpected columns detected: %s",
            unexpected,
        )

    logger.info("Prediction input validated successfully.")


def validate_probability(probability: float | None) -> None:
    """
    Validate probability value.

    Parameters
    ----------
    probability : float | None

    Raises
    ------
    ValueError
        If probability is outside [0, 1].
    """
    if probability is None:
        return

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"Probability must lie within [0, 1]. "
            f"Received {probability}."
        )


def validate_model_file(
    model_path: str | Path,
) -> Path:
    """
    Validate model artifact.

    Parameters
    ----------
    model_path : str or Path

    Returns
    -------
    pathlib.Path

    Raises
    ------
    FileNotFoundError
        If artifact does not exist.
    """
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}"
        )

    if path.suffix.lower() not in SUPPORTED_PICKLE_EXTENSIONS:
        raise ValueError(
            "Unsupported model artifact extension."
        )

    logger.info("Validated model artifact: %s", path)

    return path.resolve()


# ---------------------------------------------------------------------
# General Utilities
# ---------------------------------------------------------------------


def safe_delete(file_path: str | Path) -> bool:
    """
    Safely delete a file.

    Parameters
    ----------
    file_path : str or Path

    Returns
    -------
    bool
        True if file was deleted.
    """
    path = Path(file_path)

    if not path.exists():
        logger.warning(
            "File does not exist: %s",
            path,
        )
        return False

    try:
        path.unlink()
        logger.info("Deleted file: %s", path)
        return True

    except OSError as exc:
        logger.exception("Failed deleting file.")
        raise RuntimeError(
            f"Unable to delete file: {path}"
        ) from exc


def list_files(
    directory: str | Path,
    pattern: str = "*",
) -> list[Path]:
    """
    Return files inside a directory.

    Parameters
    ----------
    directory : str or Path

    pattern : str
        Glob pattern.

    Returns
    -------
    list[pathlib.Path]
    """
    directory = Path(directory)

    if not directory.exists():
        return []

    return sorted(
        [
            file
            for file in directory.glob(pattern)
            if file.is_file()
        ]
    )


def display_project_info() -> dict[str, Any]:
    """
    Return project metadata.

    Returns
    -------
    dict
        Project information.
    """
    return {
        "project_name": PROJECT_NAME,
        "version": PROJECT_VERSION,
        "description": PROJECT_DESCRIPTION,
        "author": PROJECT_AUTHOR,
    }


def check_random_state(
    random_state: int | None = RANDOM_STATE,
) -> int:
    """
    Validate random state.

    Parameters
    ----------
    random_state : int | None

    Returns
    -------
    int

    Raises
    ------
    ValueError
    """
    if random_state is None:
        return RANDOM_STATE

    if not isinstance(random_state, Integral):
        raise TypeError(
            "Random state must be an integer."
        )

    Random(int(random_state))

    return int(random_state)


def create_output_directories() -> None:
    """
    Create project output directories defined in config.

    This function is safe to call multiple times.
    """
    directories = [
        DATA_DIR,
        MODEL_DIR,
        EDA_VISUALS_DIR,
        MODEL_VISUALS_DIR,
        DASHBOARD_VISUALS_DIR,
    ]

    for directory in directories:
        ensure_directory(directory)

    logger.info(
        "Project output directories verified."
    )

def format_number(value):
    """
    Format large numbers for display.
    """

    if value is None:
        return "-"

    if value >= 1_000_000_000:
        return f"{value/1_000_000_000:.2f}B"

    if value >= 1_000_000:
        return f"{value/1_000_000:.2f}M"

    if value >= 1_000:
        return f"{value/1_000:.2f}K"

    return str(value)

def load_dataset(file_path):
    """
    Load dataset safely.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    return df