"""Device requirements for the FlyDSL CDNA4 GEMM suites."""

import pytest


@pytest.fixture(scope="module")
def cdna4_device():
    torch = pytest.importorskip("torch")
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("FlyDSL GEMM requires a ROCm CDNA4 (gfx950) GPU")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if props.gcnArchName.split(":", 1)[0] != "gfx950":
        pytest.skip(f"FlyDSL GEMM requires CDNA4 (gfx950); got {props.gcnArchName}")
    pytest.importorskip("flydsl.compiler")
    torch.manual_seed(0)