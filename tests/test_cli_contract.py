import subprocess
import sys


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
