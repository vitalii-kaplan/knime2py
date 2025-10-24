#!/usr/bin/env bash
set -euo pipefail

brew install ollama
brew services start ollama
brew services list | grep ollama
curl http://localhost:11434/api/tags

ollama pull llama3

curl http://localhost:11434/api/tags   
ollama run llama3 "2+2?"

#brew services stop ollama