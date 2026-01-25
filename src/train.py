import torch
from loss import gradient_penalty, temporal_smoothness_loss, spectral_loss

def train_wgan(
    G, D, loader, device,
    epochs=100,
    z_dim=100,
    n_critic=5,
    gp_lambda=10,
    lambda_smooth=0.2,
    lambda_spec=0.05
):
    G.to(device)
    D.to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))

    for epoch in range(epochs):
        for real in loader:
            real = real.to(device)
            bs = real.size(0)

            z = torch.randn(bs, z_dim, device=device)
            fake = G(z).detach()

            for _ in range(n_critic):
                d_real = D(real).mean()
                d_fake = D(fake).mean()
                gp = gradient_penalty(D, real, fake, device)

                d_loss = d_fake - d_real + gp_lambda * gp

                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()

            z = torch.randn(bs, z_dim, device=device)
            fake = G(z)

            g_loss = -D(fake).mean()
            g_loss += lambda_smooth * temporal_smoothness_loss(fake)
            g_loss += lambda_spec * spectral_loss(fake, real)

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"D: {d_loss.item():.4f} | G: {g_loss.item():.4f}"
        )
