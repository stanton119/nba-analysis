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

import pytorch_lightning as pl

class Lasso(pl.LightningModule):
    def __init__(
        self, loss_fn, n_inputs: int = 1, learning_rate=0.05, l1_lambda=0.05, l2_lambda=0.05
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.output_layer = torch.nn.Linear(n_inputs, 1)
        self.train_log = []

    def forward(self, x):
        outputs = self.output_layer(x)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def l1_reg(self):
        l1_norm = sum(p.abs().sum() for p in self.parameters())

        return self.l1_lambda * l1_norm

    def l2_reg(self):
        l2_norm = sum(p.pow(2).sum() for p in self.parameters())

        return self.l2_lambda * l2_norm

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y) + self.l1_reg() + self.l2_reg()
        
        self.log("loss", loss)
        self.train_log.append(loss.detach().numpy())
        return loss

class Dropout(pl.LightningModule):
    def __init__(
        self, loss_fn, n_inputs: int = 1, learning_rate=0.05, l1_lambda=0.05, l2_lambda=0.05, dropout_rate=0.5
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.output_layer = torch.nn.Linear(n_inputs, 1)
        self.dropout_layer = torch.nn.Dropout(self.dropout_rate)
        self.train_log = []

    def forward(self, x):
        outputs = self.output_layer(x)
        outputs = self.dropout_layer(outputs)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def l1_reg(self):
        l1_norm = sum(p.abs().sum() for p in self.parameters())

        return self.l1_lambda * l1_norm

    def l2_reg(self):
        l2_norm = sum(p.pow(2).sum() for p in self.parameters())

        return self.l2_lambda * l2_norm

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y) + self.l1_reg() + self.l2_reg()
        
        self.log("loss", loss)
        self.train_log.append(loss.detach().numpy())
        return loss


model = Dropout(
    loss_fn=torch.nn.MSELoss(),
    n_inputs=x_train.shape[1],
    l1_lambda=0.05,
    l2_lambda=0.5,
    dropout_rate=0.5,
    learning_rate=0.01,
)

# %%
trainer = pl.Trainer(max_epochs=200)#, progress_bar_refresh_rate=20)
# trainer = pl.Trainer(max_epochs=200, progress_bar_refresh_rate=20, auto_lr_find=True)
# trainer.tune(model, dataloader_train, dataloader_test)
trainer.fit(model, dataloader_train, dataloader_test)

w_model = np.append(
    model.output_layer.bias.detach().numpy()[0],
    model.output_layer.weight.detach().numpy(),
)

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
