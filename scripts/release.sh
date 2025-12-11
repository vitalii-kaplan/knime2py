#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   VERSION=0.1.12 MESSAGE="Bug fix: remove K2P UI graph from output" ./scripts/release.sh
# or
#   ./scripts/release.sh 0.1.12 "Bug fix: remove K2P UI graph from output"

VERSION="${1:-${VERSION:-}}"
MESSAGE="${2:-${MESSAGE:-}}"

if [[ -z "${VERSION}" ]]; then
  echo "ERROR: VERSION is required (env var or first arg)." >&2
  exit 2
fi
: "${MESSAGE:=Release ${VERSION}}"

# Move to repo root, no matter where this script is called from
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

PYPROJECT="pyproject.toml"
if [[ ! -f "$PYPROJECT" ]]; then
  echo "ERROR: $PYPROJECT not found at $ROOT" >&2
  exit 2
fi

# Refuse to overwrite an existing tag
if git rev-parse -q --verify "refs/tags/v${VERSION}" >/dev/null; then
  echo "ERROR: tag v${VERSION} already exists." >&2
  exit 2
fi

# Bump version in [project] section only (portable: uses awk, not sed -i differences)
cp "$PYPROJECT" "${PYPROJECT}.bak"
awk -v ver="$VERSION" '
  BEGIN { inproj = 0 }
  /^\[project\]/ { inproj = 1 }
  /^\[/ && $0 !~ /^\[project\]/ { inproj = 0 }
  inproj && $1 == "version" && $2 == "=" {
    print "version = \"" ver "\""
    next
  }
  { print }
' "${PYPROJECT}.bak" > "${PYPROJECT}"
rm -f "${PYPROJECT}.bak"

# Commit the bump
git add "$PYPROJECT"
git commit -m "chore(release): v${VERSION} — ${MESSAGE}"

# Push the commit
git push

# Create and push annotated tag
git tag -a "v${VERSION}" -m "v${VERSION} — ${MESSAGE}"
git push origin "v${VERSION}"

echo "Tagged and pushed v${VERSION}. Your GitHub workflows should trigger now."
