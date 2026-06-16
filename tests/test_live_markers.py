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
