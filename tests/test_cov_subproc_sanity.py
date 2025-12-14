# tests/test_cov_subproc_sanity.py
import os, subprocess, sys

def test_subprocess_cov_is_injected():
    code = r"""
import os, sys
ok = 'COVERAGE_PROCESS_START' in os.environ
print('COVERAGE_PROCESS_START:', ok)
try:
    import sitecustomize  # injected by pytest-cov
    print('sitecustomize:', True)
except Exception as e:
    print('sitecustomize:', False, type(e).__name__, str(e))
"""
    r = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    print(r.stdout)
    assert "COVERAGE_PROCESS_START: True" in r.stdout
    assert "sitecustomize: True" in r.stdout
