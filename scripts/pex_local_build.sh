# from repo root
/usr/bin/env python3.11 -m venv .venv-pex
source .venv-pex/bin/activate
python -m pip install -U pip build pex

# 1) Build your project wheel (uses pyproject backend)
python -m build --wheel
ls -l dist

# 2) Build the PEX from the wheel (no sdist builds inside PEX)
export PEX_VERBOSE=9
export PEX_PIP_VERSION=latest   # use modern pip for resolution
pex dist/*.whl \
  -c k2p \
  -o dist/k2p-macos-$(uname -m)_local.pex \
  --venv prepend \
  --strip-pex-env \
  --interpreter-constraint 'CPython==3.11.*'

# 3) Smoke test
python dist/k2p-macos-$(uname -m).pex --help