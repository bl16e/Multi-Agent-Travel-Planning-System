import os
from pathlib import Path

import pytest


def test_live_tests_are_skipped_without_run_live_tests(pytester):
    pytester.makeini(
        """
        [pytest]
        asyncio_default_fixture_loop_scope = function
        """
    )
    pytester.makeconftest(Path(__file__).with_name("conftest.py").read_text(encoding="utf-8"))
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.live
        def test_live_case():
            assert False
        """
    )

    os.environ.pop("RUN_LIVE_TESTS", None)
    result = pytester.runpytest("-q")

    result.assert_outcomes(skipped=1)


def test_live_tests_can_be_enabled_from_dotenv(pytester):
    pytester.makeini(
        """
        [pytest]
        asyncio_default_fixture_loop_scope = function
        """
    )
    pytester.makeconftest(Path(__file__).with_name("conftest.py").read_text(encoding="utf-8"))
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.live
        def test_live_case():
            assert True
        """
    )

    os.environ.pop("RUN_LIVE_TESTS", None)
    dotenv_path = pytester.path.parent / ".env"
    dotenv_path.write_text("RUN_LIVE_TESTS=1\n", encoding="utf-8")
    try:
        result = pytester.runpytest("-q")
    finally:
        dotenv_path.unlink(missing_ok=True)

    result.assert_outcomes(passed=1)


def test_live_full_planning_flow_e2e_test_exists():
    test_file = Path("tests/test_live_e2e.py")

    assert test_file.exists()
    assert "test_live_full_planning_flow_through_plan_api" in test_file.read_text(encoding="utf-8")
