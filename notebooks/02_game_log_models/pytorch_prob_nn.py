"""
Aletoric uncertainty on the score outputs

TODO:
Variational inference to extend to epistemic uncertainty
Sequential batches
"""
# %% Load directly form the data catalogue

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from pathlib import Path
from nba_analysis import utilities
from nba_analysis.pipelines.data_processing import basketball_reference
from nba_analysis import game_log_models
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

metadata = bootstrap_project(utilities.get_base_path())
session = KedroSession.create(project_path=utilities.get_base_path(), env=None)
context = session.load_context()

catalog = context.catalog
catalog.list()
df = catalog.load("game_log_data_2020_int")
df

# %%
df_train, team_list = game_log_models.get_training_data(df)
df_train

# %%
x, y, batch_no, row_no = game_log_models.encode_training_data(df_train)


# %%
import torch
import pytorch_lightning as pl


class DeepNormalModel(pl.LightningModule):
    def __init__(
        self,
        n_inputs: int = 1,
        learning_rate: float = 0.1,
        l2_lambda: float = 1e-5,
        constrain_weight: float = 1e-5,
        constrain_len: int = None,
        init_bias: float = 0,
    ):
        super().__init__()
        self.mean_linear = torch.nn.Linear(n_inputs, 1)
        self.mean_linear.bias.data.fill_(init_bias)
        self.scale_linear = torch.nn.Linear(n_inputs, 1)
        self.train_log = []
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.constrain_len = constrain_len
        self.constrain_weight = constrain_weight
        # self.constrain_matrix = constrain_matrix

    def forward(self, x):
        mean = self.mean_linear(x)
        scale = torch.nn.functional.softplus(self.scale_linear(x))

        return torch.distributions.Normal(mean, scale)

    def loss_fn_loglike(self, y_hat, y):
        negloglik = -y_hat.log_prob(y)
        return torch.mean(negloglik)

    def l2_reg(self):
        l2_norm = sum(p.pow(2).sum() for p in self.parameters())
        return self.l2_lambda * l2_norm

    def constrain_avg_weights(self):
        if self.constrain_len is None:
            return 0
        weight_sum = torch.reshape(
            self.mean_linear.weight, [-1, self.constrain_len]
        ).sum(dim=1)
        # only applies to attack/defense
        return self.constrain_weight * weight_sum[:2].abs().sum()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = (
            self.loss_fn_loglike(y_hat, y)
            + self.l2_reg()
            + self.constrain_avg_weights()
        )

        self.log("loss", loss)
        self.train_log.append(loss.detach().numpy())
        return loss

# %%
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

x_t = torch.Tensor(x)
y_t = torch.Tensor(y[:, np.newaxis])

dataset_train = TensorDataset(x_t, y_t)
dataloader_train = DataLoader(
    dataset_train, batch_size=x.shape[0]
)


# %%

model = DeepNormalModel(
    n_inputs=x.shape[1],
    learning_rate=0.5,
    l2_lambda=0.0,
    constrain_len=len(team_list),
    init_bias=y.mean(),
)



# %%
model(x_t)

# %%
trainer = pl.Trainer(max_epochs=200, auto_lr_find=True)
trainer.tune(model, dataloader_train)

trainer.fit(model, dataloader_train)

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(model.train_log)
ax.set_yscale('log')

y_est = model(x_t).mean.detach().numpy()
# %%
if 0:
    model = DeepNormalModel(
        n_inputs=x.shape[1],
        learning_rate=0.5,
        l2_lambda=0.0,
        constrain_len=len(team_list),
        init_bias=y.mean(),
    )

    trainer = pl.Trainer(max_epochs=1)

    for idx in row_no[:20]:
        print(idx)
        dataset_train = TensorDataset(x_t[row_no==idx,:], y_t[row_no==idx,:])
        dataloader_train = DataLoader(
            dataset_train, batch_size=int(sum(row_no==idx)), shuffle=False
        )
        trainer.fit(model, dataloader_train)

# %%
plt.plot(y)
plt.plot(y_est)

# %%
import pandas as pd
import numpy as np

coefs = np.reshape(
    model.mean_linear.weight.detach().numpy(), [-1, len(team_list)]
).transpose()
team_params = pd.DataFrame(
    data=coefs, index=team_list, columns=["attack", "defense", "home_advantage"]
)

conf = np.reshape(
    torch.nn.functional.softplus(
        model.scale_linear.bias + model.scale_linear.weight
    )
    .detach()
    .numpy(),
    [-1, len(team_list)],
).transpose()
team_conf_params = pd.DataFrame(
    data=conf,
    index=team_list,
    columns=["attack_std", "defense_std", "home_advantage_std"],
)

team_params = pd.concat([team_params, team_conf_params], axis=1)

bias = model.mean_linear.bias.detach().numpy()[0]
bias
team_params.mean()
team_params


# %% manual checking
# attack ability
temp_home = df[["home", "pts_home"]]
temp_home.columns = ["name", "pts"]
temp_away = df[["away", "pts_away"]]
temp_away.columns = ["name", "pts"]
pd.concat([temp_home, temp_away], axis=0).groupby("name")[
    "pts"
].mean().sort_values()
team_params["attack"].sort_values()

pd.concat([temp_home, temp_away], axis=0).groupby("name")[
    "pts"
].std().sort_values()
team_params["attack_std"].sort_values()

# defense ability
temp_home = df[["home", "pts_away"]]
temp_home.columns = ["name", "pts"]
temp_away = df[["away", "pts_home"]]
temp_away.columns = ["name", "pts"]
pd.concat([temp_home, temp_away], axis=0).groupby("name")[
    "pts"
].mean().sort_values()
team_params["defense"].sort_values()

# hAdv
(
    df.groupby("home")["pts_home"].mean()
    - df.groupby("away")["pts_away"].mean()
).sort_values()
team_params["home_advantage"].sort_values()
