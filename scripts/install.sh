#Instaler of dependencies from pyproject.toml

# Core (end users, CI for just the CLI)
pip install -e .

# Dev workflow
pip install -e ".[dev]"

# RAG tooling 
#pip install -e ".[rag]"          

# Example notebooks that run ML
pip install -e ".[ml-examples]"