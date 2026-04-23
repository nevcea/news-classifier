from pathlib import Path
import shutil
from uuid import uuid4

import pytest


@pytest.fixture
def tmp_path() -> Path:
    path = Path("tests") / "_tmp_runtime" / f"pytest_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
