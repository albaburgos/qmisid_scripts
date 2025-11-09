# This script applies Pytorch lightning to randomly generated data - just to check it works

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningModule, Trainer, seed_everything, Callback
from pytorch_lightning.callbacks import ModelCheckpoint

logger = TensorBoardLogger(save_dir="lightning_logs", name="kappa")

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

def f_phi(phi1, phi2):
    term1 = F.softplus(phi1 + phi2)
    stacked = torch.stack([phi1, phi2], dim=-1)
    term2 = torch.logsumexp(stacked, dim=-1)
    return term1 - term2

class KappaLightning(LightningModule):
    def __init__(self, input_dim, hidden_dim=64, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = KappaNet(input_dim=input_dim, hidden_dim=hidden_dim)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, e): #  run model on imput tensor, what happens to the output tensor
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

class LossHistory(Callback):
    def __init__(self):
        self.train_losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.train_losses.append(float(loss))

device = "cpu"
seed_everything(42) #reproducibility

logger = TensorBoardLogger(save_dir="lightning_logs", name="kappa")

D, N = 10, 32 # N = batch size = number of samples, D = feature dim
e1 = torch.randn(N, D) #first embedding
e2 = torch.randn(N, D) #second embedding
labels = torch.randint(0, 2, (N,)) 

full_ds = TensorDataset(e1, e2, labels)
n_val = max(1, int(0.2 * N))
n_train = N - n_val
train_ds, val_ds = random_split(full_ds, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=n_train, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=n_val, shuffle=False)

model = KappaLightning(input_dim=D, hidden_dim=64, lr=1e-3)
loss_cb = LossHistory()

ckpt_cb = ModelCheckpoint(
    monitor="val_loss",    
    mode="min",
    save_top_k=1,         
    save_last=True,     
    filename="kappa-{epoch:03d}-{val_loss:.4f}"
)

trainer = Trainer(
    max_epochs=1000,              
    accelerator="auto",
    devices="auto",
    enable_checkpointing=True,   
    logger=logger,               
    callbacks=[loss_cb, ckpt_cb],
)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

plt.figure(figsize=(8, 5))
plt.plot(loss_cb.train_losses, label='Training Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('KappaNet Training Loss Per Epoch (Lightning)')
plt.legend()
plt.grid(True)
plt.show()