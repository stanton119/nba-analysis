import pandas as pd
import numpy as np
from datetime import datetime

from functools import lru_cache


@lru_cache
def get_remote_data(
    url_type: str, player_id: str = None, season: int = None, month: str = None
):
    if url_type == "advanced":
        url = f"https://widgets.sports-reference.com/wg.fcgi?css=1&site=bbr&url=%2Fplayers%2F{player_id[0]}%2F{player_id}.html&div=div_advanced"
    elif url_type == "season_summary_per_game":
        url = f"https://widgets.sports-reference.com/wg.fcgi?css=1&site=bbr&url=%2Fleagues%2FNBA_{season}_per_game.html&div=div_per_game_stats"
    elif url_type == "season_summary_advanced":
        url = f"https://widgets.sports-reference.com/wg.fcgi?css=1&site=bbr&url=%2Fleagues%2FNBA_{season}_advanced.html&div=div_advanced_stats"
    elif url_type == "season_summary_shooting":
        url = f"https://widgets.sports-reference.com/wg.fcgi?css=1&site=bbr&url=%2Fleagues%2FNBA_{season}_shooting.html&div=div_shooting_stats"
    elif url_type == "season_summary_adj_shooting":
        url = f"https://widgets.sports-reference.com/wg.fcgi?css=1&site=bbr&url=%2Fleagues%2FNBA_{season}_adj_shooting.html&div=div_adj-shooting"
    elif url_type == "game_log":
        url = f"https://widgets.sports-reference.com/wg.fcgi?css=1&site=bbr&url=%2Fleagues%2FNBA_{season}_games-{month}.html&div=div_schedule"

    print(f"fetching...{url}")
    try:
        df = pd.read_html(url, flavor="bs4",)[0]
    except ValueError as e:
        if str(e) != "No tables found":
            raise
        else:
            df = None
    return df


def get_season_summary(season: int = 2020):
    df_per_game = get_remote_data(url_type="season_summary_per_game", season=season)
    df_advanced = get_remote_data(url_type="season_summary_advanced", season=season)
    # df_shooting = get_remote_data(url_type="season_summary_adj_shooting", season = season)

    df_per_game = process_df_season_summary(
        df=df_per_game, url_type="season_summary_per_game"
    )
    df_advanced = process_df_season_summary(
        df=df_advanced, url_type="season_summary_advanced"
    )
    # df_shooting = process_df_season_summary(df=df_shooting, url_type="season_summary_adj_shooting")

    merge_cols = [col for col in df_per_game.columns if col in df_advanced.columns]
    df_summary = df_per_game.merge(df_advanced, on=merge_cols, how="inner")
    return df_summary


def process_df_totals(df):
    # filter rows/columns and type
    df = df[["Season", "Age", "G", "MP"]]
    df = df.dropna(axis=1, thresh=1)
    df = df.loc[~df.isna().any(axis=1)]
    df["Season"] = pd.to_numeric(df["Season"].apply(lambda x: x[:4]))

    # keep first duplicate for changing team
    df = df.loc[~df["Age"].duplicated()]

    # reindex Age to fill gaps
    age_range = np.arange(df["Age"].min(), df["Age"].max() + 1)
    df = df.set_index("Age").reindex(age_range).reset_index()
    df["Season"] = df["Season"].interpolate()
    df = df.mask(df.isna(), 0)

    # seasonNo as number of seasons played
    df["SeasonNo"] = df["Age"] - df["Age"].min() + 1

    return df


def process_df_season_summary(df, url_type: str):
    """
    Remove player duplicates, take first row for totals
    Some rows not fully populated
    Some columns empty
    """
    # filter columns
    df = df.dropna(axis=1, thresh=1)

    # remove extra header rows
    df = df.loc[df["Rk"] != "Rk"]

    # keep first duplicate for changing team
    df = df.loc[~df[["Rk", "Player"]].duplicated()]

    # numeric types
    str_cols = ["Player", "Pos", "Tm"]
    for col in df.columns:
        if col not in str_cols:
            df[col] = pd.to_numeric(df[col])

    if url_type == "season_summary_per_game":
        # replace NaN with medians
        replace_vals = df.median(axis=0, skipna=True)
        for col in replace_vals.index:
            df[col] = df[col].mask(df[col].isna(), replace_vals[col])

    if url_type == "season_summary_advanced":
        # drop minutes played, prefer the MP per game
        df = df.drop(columns="MP")
        # remove NaN rows
        df = df.loc[df.notna().all(axis=1)]

    return df


def get_full_season_game_log(season: int = 2021) -> pd.DataFrame:
    """
    Download each month for a given season and concat
    """
    month_range = [
        "october",
        "november",
        "december",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
    ]
    month_data = [
        get_remote_data(url_type="game_log", season=season, month=month)
        for month in month_range
    ]

    season_data = pd.concat(month_data, axis=0).reset_index(drop=True)

    return season_data

def process_df_game_log(df):
    """
    Remove player duplicates, take first row for totals
    Some rows not fully populated
    Some columns empty
    """
    df.columns = [
        "date",
        "time",
        "away",
        "pts_away",
        "home",
        "pts_home",
        "drop",
        "ot",
        "attendence",
        "notes",
    ]
    df = df.drop(columns=["drop"])
    df["date"] = pd.to_datetime(df["date"])
    df["time"] = df["time"].str[:-1]
    df["time"] = df["time"].apply(
        lambda x: datetime.strptime(x, "%H:%M").time()
    )
    # pd.testing.assert_frame_equal(df, df.sort_values(["date", "time"]))

    df = df.loc[~df[['pts_home','pts_away']].isna().any(axis=1)]
    return df.reset_index(drop=True)