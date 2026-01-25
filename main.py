import pathlib
import torch

from src.config import cfg
from src.data import build_dataset, ECGDataset
from src.models import ECGGenerator, ECGDiscriminator
from src.train import train_wgan

torch.manual_seed(cfg.seed)
X_train = build_dataset(cfg.records, window_sec=2.0)
train_ds = ECGDataset(X_train)

train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=cfg.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=2,
    pin_memory=True
)

G = ECGGenerator(z_dim=cfg.z_dim)
D = ECGDiscriminator()

train_wgan(
    G=G,
    D=D,
    loader=train_loader,
    device=cfg.device,
    epochs=cfg.epochs,
    z_dim=cfg.z_dim,
    n_critic=5,
    gp_lambda=10
)


MODEL_PATH = pathlib.Path("weights")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

torch.save(G.state_dict(), MODEL_PATH / "ecg_generator.pt")
torch.save(D.state_dict(), MODEL_PATH / "ecg_discriminator.pt")
