from pathlib import Path

import pytest
from kedro.framework.context import KedroContext
from nba_analysis import utilities


@pytest.fixture
def project_context():
    return KedroContext(package_name="nba_analysis", project_path=Path.cwd())


class TestUtilities:
    def test_get_base_path(self):
        assert isinstance(utilities.get_base_path(), Path)
