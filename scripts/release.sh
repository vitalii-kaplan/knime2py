#!/usr/bin/env bash
# Release helper: bump version in src/knime2py/__init__.py, commit, tag, push.
# Usage:
#   VERSION=0.1.14 MESSAGE="Short release note" ./scripts/release.sh
#   ./scripts/release.sh 0.1.14 "Short release note"
# Env/Overrides:
#   PKG_INIT: path to __init__.py (default: src/knime2py/__init__.py)
#   REMOTE:   git remote to push to (default: origin)

set -euo pipefail

# ------------------------------- args -----------------------------------------
VERSION="${1:-${VERSION:-}}"
MESSAGE="${2:-${MESSAGE:-}}"
if [[ -z "${VERSION}" ]]; then
  echo "ERROR: VERSION is required (env var or first arg)." >&2
  exit 2
fi
: "${MESSAGE:=Release ${VERSION}}"

# Basic sanity for version (semver-ish; allow pre-release/build)
if ! [[ "${VERSION}" =~ ^[0-9]+(\.[0-9]+){1,2}([.-][0-9A-Za-z.-]+)?$ ]]; then
  echo "WARNING: VERSION '${VERSION}' does not look like semver; continuing..." >&2
fi

REMOTE="${REMOTE:-origin}"
PKG_INIT="${PKG_INIT:-src/knime2py/__init__.py}"

# ----------------------------- repo checks ------------------------------------
ROOT="$(git rev-parse --show-toplevel)"
cd "${ROOT}"

if [[ ! -f "${PKG_INIT}" ]]; then
  echo "ERROR: ${PKG_INIT} not found at ${ROOT}" >&2
  exit 2
fi

# Refuse if tag already exists
if git rev-parse -q --verify "refs/tags/v${VERSION}" >/dev/null; then
  echo "ERROR: tag v${VERSION} already exists." >&2
  exit 2
fi

# Refuse if working tree is dirty (excluding untracked)
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: working tree has changes; commit or stash first." >&2
  exit 2
fi

# ------------------------- read current version --------------------------------
current_version="$(
  awk '
    BEGIN{FS="\""; found=0}
    /^__version__[[:space:]]*=/{
      for(i=1;i<=NF;i++){ if($i ~ /^[0-9]+(\.[0-9]+){1,2}([.-][0-9A-Za-z.-]+)?$/){print $i; found=1; break}}
    }
    END{ if(!found){ exit 42 } }
  ' "${PKG_INIT}" 2>/dev/null || true
)"

if [[ -z "${current_version}" ]]; then
  echo "ERROR: __version__ string literal not found in ${PKG_INIT}." >&2
  exit 2
fi

if [[ "${current_version}" == "${VERSION}" ]]; then
  echo "ERROR: ${PKG_INIT} already has version ${VERSION}." >&2
  exit 2
fi

echo "Bumping version: ${current_version} -> ${VERSION}"

# ------------------------------ update file -----------------------------------
tmp="${PKG_INIT}.bak.$$"
cp -f "${PKG_INIT}" "${tmp}"

# Replace the __version__ line only
if ! awk -v ver="${VERSION}" '
  BEGIN{done=0}
  /^__version__[[:space:]]*=/{
    print "__version__ = \"" ver "\""; done=1; next
  }
  { print }
  END{
    if(!done){ exit 42 }
  }
' "${tmp}" > "${PKG_INIT}"; then
  rm -f "${tmp}"
  echo "ERROR: failed to update __version__ in ${PKG_INIT}." >&2
  exit 2
fi
rm -f "${tmp}"

# Verify write
new_version="$(
  python3 - <<'PY'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1])
m = re.search(r'^__version__\s*=\s*"([^"]+)"\s*$', p.read_text(encoding="utf-8"), re.M)
print(m.group(1) if m else "")
PY
  "${PKG_INIT}"
)"
if [[ "${new_version}" != "${VERSION}" ]]; then
  echo "ERROR: post-write check failed; got '${new_version}', expected '${VERSION}'." >&2
  exit 2
fi

# ------------------------------ commit/tag/push -------------------------------
git add "${PKG_INIT}"
git commit -m "chore(release): v${VERSION} — ${MESSAGE}"

# Push commit
git push "${REMOTE}" HEAD

# Annotated tag with brief body
tag_body=$(
  cat <<EOF
v${VERSION} — ${MESSAGE}
EOF
)
git tag -a "v${VERSION}" -m "${tag_body}"

# Push tag
git push "${REMOTE}" "v${VERSION}"

echo "Done: bumped to v${VERSION}, pushed commit and tag."
echo "Tip: trigger your CI/CD to build wheels/PEX/EXE for v${VERSION}."
