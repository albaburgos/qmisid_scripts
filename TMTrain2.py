import os
import math
import datetime
from pathlib import Path
from typing import Optional, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from iwpc.calculate_divergence import DivergenceResult
from iwpc.data_modules.pandas_directory_data_module import PandasDirDataModule
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
from iwpc.data_modules.pandas_directory_data_module import PandasDirDataModule
from iwpc.data_modules.pandas_directory_data_module_builder import PandasDirDataModuleBuilder
from iwpc.models.utils import basic_model_factory
from iwpc.encodings.trivial_encoding import TrivialEncoding
from iwpc.encodings.continuous_periodic_encoding import ContinuousPeriodicEncoding
from tqdm import tqdm


def f_phi(phi1: torch.Tensor, phi2: torch.Tensor) -> torch.Tensor:
    term1 = F.softplus(phi1 + phi2)
    stacked = torch.stack([phi1, phi2], dim=-1)
    term2 = torch.logsumexp(stacked, dim=-1)
    return term1 - term2

def _make_pairs(x, y):
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

        return torch.TensorDataset(e1, e2, labels)

class KappaLightning(L.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, e):
        return self.model(e)

    def _pairwise_loss(self, x, y):
        e1, e2, labels = self._make_pairs(x, y)
        phi1, phi2 = self(e1), self(e2)        
        f = f_phi(phi1, phi2)                   

        loss = F.binary_cross_entropy_with_logits(f, labels)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def calc_epsilon(
        model,
        datamodule,
        trainer_kwargs: Optional[Dict] = None,
        lr: float = 1e-3,   
    ):
            trainer_kwargs = trainer_kwargs or {}

            log_dir = "epsilon_logs"
            tb_logger = TensorBoardLogger(save_dir=log_dir, name="epsilon")

            checkpoint_callback = ModelCheckpoint(
                save_top_k=1,
                monitor="val_Df",
                mode="max",
            )

            trainer = L.Trainer(
                callbacks=[
                    checkpoint_callback,
                    EarlyStopping(monitor="val_Df", mode="max"),
                    LearningRateMonitor(logging_interval="epoch"),
                ],
                default_root_dir=log_dir,
                logger=tb_logger,
                **trainer_kwargs,
            )

            trainer.fit(model=model, datamodule=datamodule)

            best_ckpt_path = checkpoint_callback.best_model_path
            best_model = type(model).load_from_checkpoint(best_ckpt_path)

            results = trainer.validate(model=best_model, datamodule=datamodule, verbose=True)
            return results[0], best_ckpt_path


if __name__ == "__main__":
    path = "/Users/albaburgosmondejar/Desktop/shuhui_truth_dm"

    dm = PandasDirDataModule(
        dataset_dir=path,
        feature_cols=["truth_el_q_over_pt", "truth_el_eta", "truth_el_phi"],
        target_cols=["is_flipped"],  
        split=0.8,
        dataloader_kwargs={"num_workers": 4, "batch_size": 2**15},
    )

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    input_dim = len(dm.feature_cols)
    input_encoding  = TrivialEncoding(2) & ContinuousPeriodicEncoding()
    target_encoding = TrivialEncoding(1)
    model = basic_model_factory(input_encoding, target_encoding)  
    for batch in tqdm(train_loader):
        e1, e2, labels = _make_pairs(batch, n_pairs=2**5)
        _calc_epsilon(model, dm, dict(accelerator="gpu",devices="gpu",max_epochs=200,log_every_n_steps=50), lr=1e-3)
            


