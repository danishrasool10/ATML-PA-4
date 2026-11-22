import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pa4.data.tiny import ByteTokenizer, make_tiny_model  # noqa: E402
from pa4.task1.calibration import make_batches  # noqa: E402
from pa4.task1.model_utils import enumerate_units  # noqa: E402


@pytest.fixture()
def model():
    return make_tiny_model()


@pytest.fixture()
def units(model):
    return enumerate_units(model)


@pytest.fixture()
def calib_seqs():
    g = torch.Generator().manual_seed(1)
    return torch.randint(3, 280, (8, 64), generator=g).tolist()


@pytest.fixture()
def batches(calib_seqs):
    return make_batches(calib_seqs, 4)


@pytest.fixture()
def tok():
    return ByteTokenizer()
