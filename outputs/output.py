from pathlib import Path
import torch
import matplotlib.pyplot as plt

from models import ECGGenerator, ECGDiscriminator

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"

G = ECGGenerator()
D = ECGDiscriminator()

G.load_state_dict(torch.load(WEIGHTS / "ecg_wgan_generator.pt", map_location="cpu"))
D.load_state_dict(torch.load(WEIGHTS / "ecg_wgan_discriminator.pt", map_location="cpu"))

G.eval()
D.eval()

z_dim = 100
z = torch.randn(1, z_dim)

with torch.inference_mode():
    fake_ecg = G(z).squeeze().numpy()
    d_output = D(G(z)).item()

print(f"Discriminator output for generated ECG: {d_output:.4f}")
plt.plot(fake_ecg)
plt.title("Generated ECG Signal")
plt.show()