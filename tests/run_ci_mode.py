from __future__ import annotations

import subprocess
import sys


MODE_TESTS = {
    "metrust-only": [
        "tests/test_python_compat.py",
        "tests/test_delegation_ledger.py",
    ],
    "metpy": [
        "tests/test_signature_parity.py",
        "tests/test_runtime_parity.py",
        "tests/test_runtime_parity_sounding_core.py",
        "tests/test_runtime_parity_thermo_layers.py",
        "tests/test_runtime_parity_wind_profiles.py",
        "tests/test_runtime_parity_remaining.py",
        "tests/test_delegation_ledger.py",
        "tests/test_cookbook_replays.py::test_cookbook_sounding_workflow_replay",
    ],
    "metpy-xarray": [
        "tests/test_metpy_dropin_compat.py",
        "tests/test_runtime_parity_interp_dataset.py",
        "tests/test_runtime_parity_kinematics_extra.py",
        "tests/test_runtime_parity_utils_misc.py",
        "tests/test_delegation_ledger.py",
        "tests/test_cookbook_replays.py::test_cookbook_grid_workflow_replay",
        "tests/test_cookbook_replays.py::test_cookbook_xarray_workflow_replay",
    ],
}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in MODE_TESTS:
        valid = ", ".join(sorted(MODE_TESTS))
        raise SystemExit(f"usage: {sys.argv[0]} <mode>; valid modes: {valid}")

    mode = sys.argv[1]
    cmd = [sys.executable, "-m", "pytest", "-q", *MODE_TESTS[mode]]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
