import json
import os
from datetime import datetime
from typing import Any, Dict, List

LOG_DIR = os.path.expanduser("~/.localai/logs")


def log_interaction(
    query: str,
    steps: List[Dict[str, Any]],
    final_answer: str,
    total_steps: int,
) -> str:
    """
    Disabled logging for privacy: no chat data is written to disk.
    """
    return ""
