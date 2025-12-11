git fetch --tags

# Get the most recent semver tag reachable from HEAD.
LAST_TAG=$(git describe --tags --abbrev=0 --match "v[0-9]*.[0-9]*.[0-9]*" 2>/dev/null) || {
  echo "No semver tag found on this branch."; exit 1;
}

# Commits since that tag (newest first)
git log --topo-order --date=short --pretty=format:'%ad %h %<(20,trunc)%an %s' "$LAST_TAG"..HEAD