#!/usr/bin/env bash
set -euo pipefail

git tag -a v0.1.9 -m "v0.1.9 Test generator added. Now for every workflow with data input and output a functional test can be created in several minutes."
git push origin v0.1.9