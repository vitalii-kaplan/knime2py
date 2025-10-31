# Quick start (CLI)

The console entrypoint is **`k2p`**. You can also use `python -m knime2py`.

Generate BOTH notebook and script (omit `--workbook`):
```bash
k2p /path/to/workflow.knime --out out_dir
```

Or pass a directory that contains exactly one `workflow.knime`:

```bash
k2p /path/to/knime_project_dir --out out_dir
```

Only notebook:

```bash
k2p /path/to/workflow.knime --out out_dir --workbook ipynb
```

Only script:

```bash
k2p /path/to/workflow.knime --out out_dir --workbook py
```

Disable graph files:

```bash
k2p /path/to/workflow.knime --out out_dir --graph off
```

Outputs are written to `out_dir/` with one set **per component**:

```
<base>__g01.json
<base>__g01.dot
<base>__g01_workbook.ipynb
<base>__g01_workbook.py
<base>__g02.json
…
```

`<base>` is the workflow directory name; `__gNN` is the component index.
