from pathlib import Path

import pytest
from nba_analysis.pipelines.data_processing import basketball_reference


class TestBasketballReference:
    def test_get_full_season_game_log(self):
        df = basketball_reference.get_full_season_game_log(season=2019)
        assert df.shape[0] > 0

    def test_get_full_season_game_log(self):
        df = basketball_reference.get_remote_data(
            url_type="game_log", season=2019, month="february"
        )
        assert df.shape[0] > 0
