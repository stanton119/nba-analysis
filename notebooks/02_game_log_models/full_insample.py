# %% Load directly form the data catalogue

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from pathlib import Path
from nba_analysis import utilities
from nba_analysis.pipelines.data_processing import basketball_reference
from nba_analysis import game_log_models

metadata = bootstrap_project(utilities.get_base_path())
session = KedroSession.create(project_path=utilities.get_base_path(), env=None)
context = session.load_context()

catalog = context.catalog
catalog.list()
df = catalog.load("game_log_data_2020_int")
df

# %%
df, team_list = game_log_models.get_training_data(df)
df

# %%
x, y, batch_no, row_no = game_log_models.encode_training_data(df)

# %%
# full insample model
from sklearn.linear_model import Ridge
# ridge to resolve missing constraint
model = Ridge(alpha=1e-10)
model.fit(X=x, y=y)
y_est = model.predict(x)

model.coef_
# %%
import matplotlib.pyplot as plt
y_est
y
plt.plot(y)
plt.plot(y_est)

# %%
import pandas as pd
import numpy as np
coefs = np.reshape(model.coef_, [-1, len(team_list)]).transpose()
coefs = pd.DataFrame(data=coefs, index=team_list, columns=['attack','defense','home_advantage'])

coefs

# %% manual checking
# attack ability
df.groupby('home')['pts_home'].mean().sort_values()
df.groupby('away')['pts_away'].mean().sort_values()

# defense ability
df.groupby('home')['pts_away'].mean().sort_values()
df.groupby('away')['pts_home'].mean().sort_values()

# hAdv
(df.groupby('home')['pts_home'].mean() - df.groupby('away')['pts_away'].mean()).sort_values()

# %%
