from pathlib import Path


def test_example_output_does_not_contain_legacy_placeholder_transport():
    example = Path("docs/example-output.md").read_text(encoding="utf-8")

    assert "Placeholder Air" not in example
    assert "verified against historical avg" not in example
