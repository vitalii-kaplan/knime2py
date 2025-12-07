#!/usr/bin/env bash
set -euo pipefail

git tag -a v0.1.12 -m "v0.1.12
Bug fixed: Remove K2P UI graph from the output"

git push origin v0.1.12