# SPDX-License-Identifier: MIT
"""Keep GRWrite collection and Torch default-device state isolated."""

import pytest


@pytest.fixture(autouse=True)
def _isolate_default_device():
    torch = pytest.importorskip("torch")
    from torch.utils import _device

    # Other suites may set a DeviceContext during collection; run these tests without it.
    previous = _device.CURRENT_DEVICE
    torch.set_default_device(None)
    try:
        yield
    finally:
        torch.set_default_device(previous)
