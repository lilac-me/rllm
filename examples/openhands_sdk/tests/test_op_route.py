"""Unit test for op_route (routing key).

Dep-light (stdlib only) so it runs on a dev machine without NPU/rllm.
Run: `python3 tests/test_op_route.py`  or  `pytest tests/test_op_route.py`.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import op_route  # noqa: E402


def test_op_route_default_is_triton():
    assert op_route.op_route({}) == "triton"                              # fallback = triton
    assert op_route.op_route({"operator_backend": "triton"}) == "triton"
    assert op_route.op_route({"operator_backend": "TRITON"}) == "triton"  # case-insensitive
    assert op_route.op_route({"operator_backend": "ascendc"}) == "ascendc"
    assert op_route.op_route({"operator_backend": "AscendC"}) == "ascendc"


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for n, f in fns:
        f()
        print(f"PASS {n}")
    print(f"ALL {len(fns)} PASS")
