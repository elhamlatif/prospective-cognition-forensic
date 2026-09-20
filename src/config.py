"""
config.py
=========

Central place for every filesystem path the project uses.

All paths are defined relative to the project root (the folder that
contains this file's parent), so the analysis runs correctly no matter
where the repository is cloned to. If you move the project, nothing
here needs to change.

Layout assumed:

    <project root>/
        data/                       raw SPSS file lives here
        outputs/                    result tables are written here
        logs/                       run logs are written here
        cleaned_data.csv            produced by 01_data_preparation.py
        src/config.py               <- this file
"""

from pathlib import Path

# Project root = the folder that contains this file's parent (src/).
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Input ---
# Raw SPSS dataset (PLOS ONE supplementary material, N = 123).
RAW_FILE = BASE_DIR / "data" / "pone.0191551.s001.sav"

# --- Intermediate ---
# Cleaned dataset produced by 01_data_preparation.py.
# Re-running that script will overwrite this file.
CLEANED_FILE = BASE_DIR / "cleaned_data.csv"

# --- Output ---
# Directory where every result table (Table2_*.csv, Appendix_*.csv, ...)
# is written. Created automatically if it does not exist.
OUTPUT_DIR = BASE_DIR / "outputs"

# --- Logging ---
# Directory and file used for the run log.
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "analysis.log"