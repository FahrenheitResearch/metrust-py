import subprocess
import sys
import textwrap


def test_public_signatures_match_metpy_even_when_metrust_imports_first():
    script = textwrap.dedent(
        """
        import inspect
        import metrust.calc as mrcalc
        import metpy.calc as mpcalc

        ignore = {"set_module"}
        mp = {}
        mr = {}

        for name in dir(mpcalc):
            if name.startswith("_") or name in ignore:
                continue
            obj = getattr(mpcalc, name)
            if callable(obj):
                mp[name] = obj

        for name in dir(mrcalc):
            if name.startswith("_") or name in ignore:
                continue
            obj = getattr(mrcalc, name)
            if callable(obj):
                mr[name] = obj

        mismatches = []
        for name in sorted(set(mp) & set(mr)):
            try:
                mp_sig = str(inspect.signature(mp[name]))
                mr_sig = str(inspect.signature(mr[name]))
            except Exception:
                continue
            if mp_sig != mr_sig:
                mismatches.append((name, mp_sig, mr_sig))

        if mismatches:
            for name, mp_sig, mr_sig in mismatches:
                print(f"{name}: {mp_sig} != {mr_sig}")
            raise SystemExit(1)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
