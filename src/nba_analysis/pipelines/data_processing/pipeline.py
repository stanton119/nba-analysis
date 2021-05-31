"""
Two pipelines:
* full history
* update latest season
    * Only updates latest season year
"""


from functools import partial
import itertools
from kedro.pipeline import Pipeline, node

from nba_analysis.pipelines.data_processing import basketball_reference

from . import nodes


def create_pipeline(**kwargs):
    season_range = range(2018, 2021)
    download_nodes = [
        node(
            func=partial(nodes.download_season_data, season=season),
            inputs=[],
            outputs=f"season_data_{season}",
            name=f"download_season_data_{season}_node",
        )
        for season in season_range
    ]
    # month_range = ['october','november','december','january','february','march','april','may','june','july','august','september']
    # download_game_log_nodes = [
    #     node(
    #         func=partial(nodes.download_game_log_data, season=season, month=month),
    #         inputs=[],
    #         outputs=f"game_log_data_{season}_{month}",
    #         name=f"download_game_log_data_{season}_{month}_node",
    #     )
    #     for season, month in itertools.product(season_range,month_range)
    # ]
    download_game_log_nodes = [
        node(
            func=partial(
                basketball_reference.get_full_season_game_log, season=season
            ),
            inputs=[],
            outputs=f"game_log_data_{season}",
            name=f"download_game_log_data_{season}_node",
        )
        for season in season_range
    ]
    process_game_log_nodes = [
        node(
            func=basketball_reference.process_df_game_log,
            inputs=f"game_log_data_{season}",
            outputs=f"game_log_data_{season}_int",
            name=f"process_game_log_data_{season}_node",
        )
        for season in season_range
    ]

    return Pipeline(
        [
            *download_nodes,
            node(
                func=nodes.process_season_data,
                inputs=[f"season_data_{season}" for season in season_range],
                outputs="cleaned_season_data",
                name="process_season_data_node",
            ),
            *download_game_log_nodes,
            *process_game_log_nodes,
        ]
    )
