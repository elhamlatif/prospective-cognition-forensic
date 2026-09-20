"""
io_utils.py
===========

Logging setup for the analysis scripts.

The project runs everything through a single logger ("analysis") so that
the run log and the live terminal output stay in sync. Each call to
setup_logging() rebuilds the handlers from scratch, which means the same
script can be re-run inside one Python session without duplicate lines
piling up in the log file.

Used by: main_analyses.py
"""

import logging
from pathlib import Path


def setup_logging(log_file, level=logging.INFO):
    """Return a logger that writes to both `log_file` and the console.

    Parameters
    ----------
    log_file : str or Path
        Where to write the log. Parent directories are created if missing.
    level : int
        Logging level (default: INFO).

    Returns
    -------
    logging.Logger
        A logger named "analysis" with a file handler and a console
        handler attached. Existing handlers are cleared first, so calling
        this function twice in one session does not duplicate output.
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("analysis")
    logger.setLevel(level)

    # Remove any handlers from a previous call so we don't log twice.
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- File handler: full log kept on disk for later inspection. ---
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # --- Console handler: same messages, but on stdout so the user
    #     sees progress while the script is running. ---
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger