# Installation & Distribution Options

You can run **knime2py** in three ways.
## 1) Docker image (no local Python/pip needed)
````markdown

Pull and run the published image (GHCR):

```bash
docker pull ghcr.io/vitalii-kaplan/knime2py:latest
docker run --rm ghcr.io/vitalii-kaplan/knime2py:latest --help
````

Typical run (simple mounts):

```bash
docker run --rm \
  -v "$PWD/workflow":/wf:ro \
  -v "$PWD/out":/out \
  ghcr.io/vitalii-kaplan/knime2py:latest \
  /wf --out /out --workbook both
```

Preserve **host absolute paths** in generated code (mirror the path inside the container):

```bash
docker run --rm \
  -u "$(id -u):$(id -g)" \
  -v "$PWD":"$PWD" \
  -w "$PWD" \
  ghcr.io/vitalii-kaplan/knime2py:latest \
  "$PWD/workflow" --out "$PWD/out" --graph off
```

A helper script is available:
`k2p_docker.sh` — [https://github.com/vitalii-kaplan/knime2py/blob/main/k2p_docker.sh](https://github.com/vitalii-kaplan/knime2py/blob/main/k2p_docker.sh)

---

## 2) PEX single-file binaries (require Python **3.11** on the user’s machine)

Download OS-specific binaries from **Releases**:
[https://github.com/vitalii-kaplan/knime2py/releases](https://github.com/vitalii-kaplan/knime2py/releases)

**macOS / Linux**

```bash
python3 --version      # must be 3.11.x
chmod +x k2p-macos-<arch>.pex    # or: k2p-linux.pex
python3 k2p-macos-<arch>.pex --help

# Example
python3 k2p-macos-<arch>.pex /path/to/workflow --out /path/to/out --graph off
```

**Windows (PowerShell)**

```powershell
py -3.11 k2p-windows.pex --help
py -3.11 k2p-windows.pex C:\path\to\workflow --out C:\path\to\out --graph off
```

On first run, PEX materializes a managed virtualenv in `~/.pex` (or `%USERPROFILE%\.pex`); no network access is needed at runtime since dependencies are bundled.

---

## 3) Source (developer) install

```bash
python -m pip install --upgrade pip
pip install -e .
# optional: run tests
pytest -q
```