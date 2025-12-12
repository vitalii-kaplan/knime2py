#!/usr/bin/env bash
# scripts/pex_local_build.sh
# Build a PEX for this project. Version is taken from src/<pkg>/__init__.py (__version__),
# with a safe fallback to pyproject.toml's [project].version.

set -euo pipefail
IFS=$'\n\t'

# ---- Config (overridable via env) ------------------------------------------------------------
PYTHON="${PYTHON:-python3}"
MODE="${1:-online}"                 # online | wheelhouse | offline | lock
WHEELHOUSE="${WHEELHOUSE:-wheelhouse}"
PEX_EXTRA_FLAGS="${PEX_EXTRA_FLAGS:-}" # e.g. "--venv --seed"
CLI_NAME="${CLI_NAME:-knime2py}"       # console_script to expose; defaults to "knime2py"

command -v "${PYTHON}" >/dev/null || { echo "ERROR: ${PYTHON} not found"; exit 1; }
command -v pex        >/dev/null || { echo "ERROR: pex not found in PATH"; exit 1; }

# ---- Infer PKG_NAME from pyproject.toml (fallback to knime2py) -------------------------------
read_pyproject_name() {
  "${PYTHON}" - <<'PY' || true
import sys, os, tomllib
if not os.path.exists("pyproject.toml"):
    sys.exit(1)
with open("pyproject.toml","rb") as f:
    data = tomllib.load(f)
name = (data.get("project") or {}).get("name")
if name and isinstance(name,str):
    print(name)
PY
}
PKG_NAME="${PKG_NAME:-$(read_pyproject_name || true)}"
PKG_NAME="${PKG_NAME:-knime2py}"

# ---- Locate __init__.py ----------------------------------------------------------------------
INIT_PATH=""
if [[ -f "src/${PKG_NAME}/__init__.py" ]]; then
  INIT_PATH="src/${PKG_NAME}/__init__.py"
elif [[ -f "${PKG_NAME}/__init__.py" ]]; then
  INIT_PATH="${PKG_NAME}/__init__.py"
else
  # Try to discover a single src/*/__init__.py
  CANDIDATES=($(find src -maxdepth 2 -type f -name __init__.py 2>/dev/null || true))
  if [[ ${#CANDIDATES[@]} -eq 1 ]]; then
    INIT_PATH="${CANDIDATES[0]}"
    # Derive PKG_NAME from that path if we didn't have it
    [[ -z "${PKG_NAME}" ]] && PKG_NAME="$(basename "$(dirname "${INIT_PATH}")")"
  fi
fi

if [[ -z "${INIT_PATH}" || ! -f "${INIT_PATH}" ]]; then
  echo "ERROR: Cannot find __init__.py (looked under src/${PKG_NAME}/ and ${PKG_NAME}/)."
  exit 1
fi

# ---- Read version from __init__.py; fallback to pyproject.toml -------------------------------
read_version() {
  "${PYTHON}" - "${INIT_PATH}" <<'PY'
import ast, sys, io, os, tomllib
init_path = sys.argv[1]
version = None

if os.path.exists(init_path):
    with io.open(init_path, "r", encoding="utf-8") as f:
        t = ast.parse(f.read(), filename=init_path)
    for n in t.body:
        if isinstance(n, ast.Assign):
            for target in n.targets:
                if isinstance(target, ast.Name) and target.id == "__version__":
                    v = n.value
                    if isinstance(v, ast.Constant) and isinstance(v.value, str):
                        version = v.value
                        break
        if version:
            break

# Fallback: pyproject.toml [project].version
if not version and os.path.exists("pyproject.toml"):
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    pv = (data.get("project") or {}).get("version")
    if isinstance(pv, str) and pv:
        version = pv

if not version:
    sys.stderr.write("FATAL: __version__ not found as a string literal in %s and no [project].version in pyproject.toml\n" % init_path)
    sys.exit(2)
print(version)
PY
}

VERSION="$(read_version)"
OUT_DIR="dist"
OUT_PEX="${OUT_DIR}/${PKG_NAME}-${VERSION}.pex"
mkdir -p "${OUT_DIR}"

echo "Package: ${PKG_NAME}"
echo "Version: ${VERSION}"
echo "Mode:    ${MODE}"
echo "Python:  ${PYTHON}"
echo "CLI:     ${CLI_NAME}"
echo "Output:  ${OUT_PEX}"

# Avoid surprises coming from the environment
unset PEX_PIP_OPTIONS || true

# ---- Build the wheel first -------------------------------------------------------------------
echo "==> Building wheel"
"${PYTHON}" -m build --wheel >/dev/null

# ---- Helper to build a PEX -------------------------------------------------------------------
build_pex() {
  local find_links="$1"; shift || true
  local no_index_flag="${1:-}"; shift || true
  echo "==> Building PEX (${MODE})"
  # shellcheck disable=SC2086
  pex -f "${find_links}" ${no_index_flag} \
      "${PKG_NAME}==${VERSION}" \
      -c "${CLI_NAME}" \
      -o "${OUT_PEX}" \
      ${PEX_EXTRA_FLAGS}
  chmod +x "${OUT_PEX}"
  echo "==> Built ${OUT_PEX}"
}

# ---- Modes -----------------------------------------------------------------------------------
case "${MODE}" in
  online)
    # Online resolve from PyPI while preferring your local wheel via -f dist
    build_pex "dist"
    ;;

  wheelhouse)
    echo "==> Refreshing wheelhouse at ${WHEELHOUSE}"
    rm -rf "${WHEELHOUSE}"
    mkdir -p "${WHEELHOUSE}"
    # Download your wheel (from dist) and all deps (from PyPI) as binary wheels
    "${PYTHON}" -m pip download -d "${WHEELHOUSE}" -f dist --only-binary=:all: "${PKG_NAME}==${VERSION}"
    build_pex "${WHEELHOUSE}" "--no-index"
    ;;

  offline)
    if [[ ! -d "${WHEELHOUSE}" ]]; then
      echo "ERROR: ${WHEELHOUSE} not found. Run '$0 wheelhouse' first."
      exit 1
    fi
    build_pex "${WHEELHOUSE}" "--no-index"
    ;;

  lock)
    echo "==> Creating PEX lock (pex.lock)"
    pex3 lock create -f dist "${PKG_NAME}==${VERSION}" -o pex.lock
    # shellcheck disable=SC2086
    pex -r pex.lock -c "${CLI_NAME}" -o "${OUT_PEX}" ${PEX_EXTRA_FLAGS}
    chmod +x "${OUT_PEX}"
    echo "==> Built ${OUT_PEX}"
    ;;

  *)
    echo "Usage: $0 [online|wheelhouse|offline|lock]"
    exit 2
    ;;
esac

# ---- Smoke test ------------------------------------------------------------------------------
echo "==> Smoke test: ${OUT_PEX} --version"
if ! "${OUT_PEX}" --help >/dev/null 2>&1; then
  echo "WARN: Help command failed. The PEX may still be fine if your CLI requires arguments."
fi

echo "Done."
