from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brainmemory import Brain, BrainConfig


@pytest.fixture
def tiny_brain() -> Brain:
    return Brain(BrainConfig.tiny(seed=3), device="cpu")


@pytest.fixture
def small_brain() -> Brain:
    return Brain(BrainConfig.small(seed=11), device="cpu")
