# SPDX-License-Identifier: MIT
"""Keep GRRead collection and Torch default-device state isolated."""

import pytest

collect_ignore = ["results", "experiments"]


@pytest.fixture(autouse=True)
def _isolate_default_device():
	torch = pytest.importorskip("torch")
	from torch.utils import _device

	# Other suites set a DeviceContext during collection. Torch 2.9 cannot
	# compile Tensor.unflatten under it, even when the default device is CPU.
	previous = _device.CURRENT_DEVICE
	torch.set_default_device(None)
	try:
		yield
	finally:
		torch.set_default_device(previous)
