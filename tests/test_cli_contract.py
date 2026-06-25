import subprocess
import sys


def test_cli_does_not_write_debug_dict_by_default(tmp_path):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "_chat.txt").write_text(
        "[01/02/2024, 18:30:00] Alice: hello\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "chatparser.py",
            "--input-directory",
            str(export_dir),
            "--no-progress-bar",
            "--force-redo",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert not (export_dir / "_debug_dict.json").exists()


def test_cli_requires_input_directory():
    result = subprocess.run(
        [sys.executable, "chatparser.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "at least one --input-directory is required" in result.stderr


def test_cli_rejects_inaccessible_input_directory(tmp_path):
    missing = tmp_path / "missing"

    result = subprocess.run(
        [sys.executable, "chatparser.py", "--input-directory", str(missing)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "not a directory or is not accessible" in result.stderr
