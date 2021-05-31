import pandas as pd
from . import basketball_reference


def download_season_data(season: int = 2020) -> pd.DataFrame:
    """
    Download data from basketball_reference for a given season
    """
    return basketball_reference.get_remote_data(
        url_type="season_summary_per_game", season=season
    )


def download_game_log_data(season: int = 2021, month: str = "december") -> pd.DataFrame:
    """
    Download game log data from basketball_reference for a given season
    """
    return basketball_reference.get_remote_data(
        url_type="game_log", season=season, month=month
    )


def process_season_data(*args) -> pd.DataFrame:
    """
    Takes multiple season data frames, cleans each and combines into single dataframe.
    """
    return pd.concat(
        map(
            lambda df: basketball_reference.process_df_season_summary(
                df=df, url_type="season_summary_per_game"
            ),
            [*args],
        ),
        axis=0,
    )
