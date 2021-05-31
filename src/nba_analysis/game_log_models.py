import pandas as pd
import numpy as np

def get_training_data(df):
    """
    date to offset int from start of season - int for ordering for now
    team names to ints
    
    pts_home = home_attack - away_defense
    pts_away = away_attack - home_defense
    """
    # self.
    team_list = (
        pd.concat([df["away"], df["home"]])
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
        .rename("team_name")
    )
    df_out = pd.DataFrame()
    df_out["day_no"] = (df["date"] - df["date"].min()).dt.days
    df_out["home"] = df["home"].replace(
        pd.Series(data=team_list.index, index=team_list)
    )
    df_out["away"] = df["away"].replace(
        pd.Series(data=team_list.index, index=team_list)
    )

    df_out["pts_home"] = df["pts_home"]
    df_out["pts_away"] = df["pts_away"]
    return df_out, team_list


def encode_training_data(df):
    """
    two rows per game
    each team has attack/defense parameter
    attack/defense as sparse row per game
    model coefficients:
    [home_attack away_attack home_defense away_defense]

    training_data:
    # [pts_home ~ home_attack 0           0            away_defense]
    # [pts_away ~ 0           away_attack home_defense 0           ]

    [pts_home ~ home_attack away_defense home_advantage]
    [pts_away ~ away_attack home_defense 0             ]
    """

    temp = pd.get_dummies(df["home"]) * 0
    temp.sum()
    home_encode = pd.concat(
        [
            pd.get_dummies(df["home"]),
            pd.get_dummies(df["away"]),
            pd.get_dummies(df["home"]),
        ],
        axis=1,
    )
    away_encode = pd.concat(
        [
            pd.get_dummies(df["away"]),
            pd.get_dummies(df["home"]),
            temp,
        ],
        axis=1,
    )

    x = pd.concat([home_encode, away_encode], axis=0)
    y = pd.concat([df["pts_home"], df["pts_away"]], axis=0)
    batch_no = pd.concat([df["day_no"], df["day_no"]], axis=0)
    row_no = x.index.to_numpy()
    sort_id = np.argsort(row_no)

    x = x.iloc[sort_id].to_numpy()
    y = y.iloc[sort_id].to_numpy()
    batch_no = batch_no.iloc[sort_id].to_numpy()
    row_no = row_no[sort_id]

    return x, y, batch_no, row_no