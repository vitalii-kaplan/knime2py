"""Web-service CLI contract checks for knime2py."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from knime2py.cli import run_cli


def _assert_single_error_line(err: str) -> None:
    lines = [ln for ln in err.strip().splitlines() if ln.strip()]
    assert len(lines) == 1


def _bundle_from_dir(src_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in src_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.relative_to(src_dir))


def _bundle_with_backslashes(workflow_dir: Path, zip_path: Path) -> None:
    workflow_text = (workflow_dir / "workflow.knime").read_text(encoding="utf-8")
    settings_text = (workflow_dir / "CSV Reader (#1)" / "settings.xml").read_text(encoding="utf-8")
    writer_text = (workflow_dir / "CSV Writer (#2)" / "settings.xml").read_text(encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("workflow.knime", workflow_text)
        zf.writestr(r"CSV Reader (#1)\settings.xml", settings_text)
        zf.writestr(r"CSV Writer (#2)\settings.xml", writer_text)


def test_cli_outputs_deterministic_files(tmp_path: Path, workflow):
    wf = workflow("KNIME_io_csv")
    out_dir = tmp_path / "out"

    code = run_cli([str(wf.parent), "--out", str(out_dir)])
    assert code == 0

    base = "KNIME_io_csv__g01"
    graph_json = out_dir / f"{base}.json"
    graph_dot = out_dir / f"{base}.dot"
    wb_py = out_dir / f"{base}_workbook.py"
    wb_ipynb = out_dir / f"{base}_workbook.ipynb"

    assert graph_json.exists()
    assert graph_dot.exists()
    assert wb_py.exists()
    assert wb_ipynb.exists()

    payload = json.loads(graph_json.read_text(encoding="utf-8"))
    assert payload.get("workflow_id") == base

    py_text = wb_py.read_text(encoding="utf-8")
    assert "knime2py" in py_text
    assert "context = {}" in py_text


def test_cli_in_zip_uses_stable_ids(tmp_path: Path, workflow):
    wf = workflow("KNIME_io_csv")
    zip_path = tmp_path / "bundle.zip"
    _bundle_from_dir(wf.parent, zip_path)

    out_dir = tmp_path / "out_zip"
    code = run_cli(["--in-zip", str(zip_path), "--out", str(out_dir)])
    assert code == 0

    base = "bundle__g01"
    assert (out_dir / f"{base}.json").exists()
    assert (out_dir / f"{base}_workbook.py").exists()


def test_cli_in_zip_backslash_paths(tmp_path: Path, workflow):
    wf = workflow("KNIME_io_csv")
    zip_path = tmp_path / "bundle_win.zip"
    _bundle_with_backslashes(wf.parent, zip_path)

    out_dir = tmp_path / "out_zip_win"
    code = run_cli(["--in-zip", str(zip_path), "--out", str(out_dir)])
    assert code == 0
    assert (out_dir / "bundle_win__g01_workbook.py").exists()


def test_cli_missing_settings_is_hard_error(tmp_path: Path, workflow, capsys):
    wf = workflow("KNIME_io_csv")
    work_dir = tmp_path / "missing_settings"
    work_dir.mkdir(parents=True, exist_ok=True)
    work_dir.joinpath("workflow.knime").write_text(
        wf.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out_missing"
    code = run_cli([str(work_dir), "--out", str(out_dir)])
    assert code == 5

    err = capsys.readouterr().err
    _assert_single_error_line(err)
    assert "\"code\": \"missing_settings\"" in err


def test_cli_zip_rejects_path_traversal(tmp_path: Path, workflow, capsys):
    wf = workflow("KNIME_io_csv")
    zip_path = tmp_path / "bad_traversal.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("workflow.knime", wf.read_text(encoding="utf-8"))
        zf.writestr("../escape.txt", "oops")

    out_dir = tmp_path / "out_bad"
    code = run_cli(["--in-zip", str(zip_path), "--out", str(out_dir)])
    assert code == 1
    err = capsys.readouterr().err
    _assert_single_error_line(err)
    assert "\"code\": \"general_failure\"" in err


def test_cli_zip_rejects_symlink_entry(tmp_path: Path, workflow, capsys):
    wf = workflow("KNIME_io_csv")
    zip_path = tmp_path / "bad_symlink.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("workflow.knime", wf.read_text(encoding="utf-8"))
        info = zipfile.ZipInfo("link.txt")
        info.external_attr = 0o120777 << 16
        zf.writestr(info, "target")

    out_dir = tmp_path / "out_bad_symlink"
    code = run_cli(["--in-zip", str(zip_path), "--out", str(out_dir)])
    assert code == 1
    err = capsys.readouterr().err
    _assert_single_error_line(err)
    assert "\"code\": \"general_failure\"" in err


def test_cli_settings_path_escape_is_rejected(tmp_path: Path, workflow, capsys):
    wf = workflow("KNIME_single_csv")
    work_dir = tmp_path / "escape_settings"
    work_dir.mkdir(parents=True, exist_ok=True)
    escape_dir = tmp_path / "escape"
    escape_dir.mkdir(parents=True, exist_ok=True)
    escape_dir.joinpath("settings.xml").write_text(
        "<config key='settings.xml'/>",
        encoding="utf-8",
    )

    workflow_text = wf.read_text(encoding="utf-8")
    workflow_text = workflow_text.replace(
        "CSV Reader (#1)/settings.xml",
        "../escape/settings.xml",
    )
    work_dir.joinpath("workflow.knime").write_text(workflow_text, encoding="utf-8")

    out_dir = tmp_path / "out_escape"
    code = run_cli([str(work_dir), "--out", str(out_dir)])
    assert code == 5
    err = capsys.readouterr().err
    _assert_single_error_line(err)
    assert "\"code\": \"missing_settings\"" in err


def test_cli_in_zip_requires_root_workflow(tmp_path: Path, workflow, capsys):
    wf = workflow("KNIME_io_csv")
    zip_path = tmp_path / "nested.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in wf.parent.rglob("*"):
            if file_path.is_file():
                arcname = Path("MyWorkflow") / file_path.relative_to(wf.parent)
                zf.write(file_path, arcname=arcname.as_posix())

    out_dir = tmp_path / "out_nested"
    code = run_cli(["--in-zip", str(zip_path), "--out", str(out_dir)])
    assert code == 2
    err = capsys.readouterr().err
    _assert_single_error_line(err)
    assert "\"code\": \"missing_workflow\"" in err


def test_cli_zip_rejects_high_compression_ratio(tmp_path: Path, workflow, capsys):
    wf = workflow("KNIME_io_csv")
    zip_path = tmp_path / "ratio.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("workflow.knime", wf.read_text(encoding="utf-8"))
        zf.writestr("big.txt", "A" * 20000)

    out_dir = tmp_path / "out_ratio"
    code = run_cli(["--in-zip", str(zip_path), "--out", str(out_dir)])
    assert code == 1
    err = capsys.readouterr().err
    _assert_single_error_line(err)
    assert "\"code\": \"general_failure\"" in err
