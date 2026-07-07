import shutil
import subprocess

import pytest


def test_frontend_bundle_is_valid_javascript():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to syntax-check static/app.js")

    result = subprocess.run(
        [node, "--check", "static/app.js"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
