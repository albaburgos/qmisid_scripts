from typing import Union
from numpy.typing import ArrayLike
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
from bokeh.io import show, output_notebook
from iwpc.visualise.bokeh_function_visualiser_1D import BokehFunctionVisualiser1D
from iwpc.data_modules.pandas_directory_data_module import PandasDirDataModule
from iwpc.data_modules.pandas_directory_data_module_builder import PandasDirDataModuleBuilder
from iwpc.models.utils import basic_model_factory
from iwpc.encodings.trivial_encoding import TrivialEncoding
from iwpc.encodings.continuous_periodic_encoding import ContinuousPeriodicEncoding
from tqdm import tqdm
from pathlib import Path
import os
import lightning as L
from iwpc.calculate_divergence import calculate_divergence
from lightning import Trainer


"""

def f_phi(phi1, phi2):
    term1 = F.softplus(phi1 + phi2)
    stacked = torch.stack([phi1, phi2], dim=-1)
    term2 = torch.logsumexp(stacked, dim=-1)
    return term1 - term2

class KappaNet(nn.Module): 
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, e):
        return self.net(e).squeeze(-1)

class KappaLightning(pl.LightningModule):
    def __init__(self,model, lr=1e-3):
        super().__init__()
        self.model = model
        self.save_hyperparameters()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr

    def forward(self, e):
        return self.model(e)

    def _pairwise_loss(self, e1, e2, labels):
        phi1, phi2 = self(e1), self(e2)
        f = f_phi(phi1, phi2)
        return self.criterion(f, labels.float())

    def training_step(self, batch, batch_idx):
        e1, e2, labels = batch # tuple of data + labels
        loss = self._pairwise_loss(e1, e2, labels) # calculate cross-entropy loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True) #log
        return loss
    
    def validation_step(self, batch, batch_idx):
        e1, e2, labels = batch
        loss = self._pairwise_loss(e1, e2, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self): #pass model parameters and the learning rate
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr) #hparams used to rank runs


def make_pairs(df, n_pairs):
        i = np.random.randint(0, len(df), size=n_pairs)
        j = np.random.randint(0, len(df), size=n_pairs)
        # ensure i != j
        j[i == j] = (j[i == j] + 1) % len(df)

        e1 = torch.tensor(df.iloc[i].values, dtype=torch.float32)
        e2 = torch.tensor(df.iloc[j].values, dtype=torch.float32)

        # Label: 1 = same charge (SC), 0 = opposite charge (OC)
        charge1 = df.iloc[i]["truth_el_charge"].values
        charge2 = df.iloc[j]["truth_el_charge"].values
        labels = torch.tensor((charge1 * charge2 > 0).astype(np.float32))

        return TensorDataset(e1, e2, labels)

def visualise_1d(model, data_batch, feature_cols, x_feature_index=0, n_points=1000):
    #Visualise the 1D effect of a single feature.
    input_data = data_batch.clone().detach()
    fixed_values = input_data[0].numpy()
    
    x_min, x_max = input_data[:, x_feature_index].min().item(), input_data[:, x_feature_index].max().item()
    x_vals = np.linspace(x_min, x_max, n_points)
    
    inputs = np.tile(fixed_values, (n_points, 1))
    inputs[:, x_feature_index] = x_vals

    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(inputs, dtype=torch.float32)).numpy()

    output_notebook()
    visualiser_1d = BokehFunctionVisualiser1D(
        function=lambda x: model(torch.tensor(x, dtype=torch.float32)).numpy(),
        input_scalars=feature_cols,
        output_scalar='output',
        initial_x_axis_scalar_ind=x_feature_index,
        use_points=True
    )
    visualiser_1d.xbins = x_vals
    visualiser_1d.last_output = y_pred
    visualiser_1d.last_scalar_output = y_pred
    visualiser_1d.setup_figure()
    visualiser_1d.setup()
    show(visualiser_1d.root)

    
    """


if __name__ == "__main__":
    path = "/Users/albaburgosmondejar/Desktop/shuhui_truth_dm"

    dm = PandasDirDataModule(
        dataset_dir=path,
        feature_cols=['truth_el_q_over_pt', 'truth_el_eta', 'truth_el_phi'],
        target_cols=['is_flipped'],
        split=0.8,
        dataloader_kwargs={"num_workers": 0, "batch_size": 2**15},
    )

    input_encoding  = TrivialEncoding(2) & ContinuousPeriodicEncoding()
    target_encoding = TrivialEncoding(1)
    model = basic_model_factory(input_encoding, target_encoding)  # must be FDivergenceEstimator

    result = new function
        module=model,
        data_module=dm,
        patience=20,
        name="flip-vs-not",
        trainer_kwargs=dict(
            accelerator="auto",
            devices="auto",
            max_epochs=200,     # adjust as you like
            log_every_n_steps=50,
        ),
    )

new function trains model using 





"""





if __name__ == "__main__":
    path = "/Users/albaburgosmondejar/Desktop/shuhui_truth_dm"
    
    dm = PandasDirDataModule(
        dataset_dir=path,
        feature_cols=['truth_el_q_over_pt', 'truth_el_eta', 'truth_el_phi'],  
        target_cols=['is_flipped'],
        split=0.8,
        dataloader_kwargs={"num_workers": 0, "batch_size": 2**15},
    )

    input_encoding = TrivialEncoding(2) & ContinuousPeriodicEncoding()
    model = basic_model_factory(
        input_encoding, 
        TrivialEncoding(1)
    )


    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    model2 = KappaLightning(model, lr=1e-3)
    logger = TensorBoardLogger(save_dir="lightning_logs", name="kappa")
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="gpu",
        devices=1,
        logger=logger,
        enable_checkpointing=True,
    )


    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    batch = next(iter(train_loader))  
    e1, e2, labels = make_pairs(batch, n_pairs=2**5)
    visualise_1d(model, e1, dm.feature_cols, x_feature_index=0)






    for batch in tqdm(train_loader):
        e1, e2, labels = make_pairs(batch, n_pairs=2**5)
        loss = pairwise_loss(model, e1, e2, labels)
        trainer = Trainer(
            max_epochs=5,              
            accelerator="gpu",
            devices="gpu",
            enable_checkpointing=True,      
        )

        logger = TensorBoardLogger(save_dir="lightning_logs", name="kappa")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        visualise_1d(model, e1, dm.feature_cols, x_feature_index=0)





setup_stdout_logging

"""