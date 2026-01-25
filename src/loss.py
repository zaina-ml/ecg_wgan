import torch

def gradient_penalty(D, real, fake, device):
    alpha = torch.rand(real.size(0), 1, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    d_interp = D(interp)

    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    grads = grads.view(grads.size(0), -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


def temporal_smoothness_loss(fake):
    d1 = fake[:, :, 1:] - fake[:, :, :-1]
    d2 = d1[:, :, 1:] - d1[:, :, :-1]
    return torch.mean(torch.abs(d2))


def spectral_loss(fake, real):
    fake_fft = torch.fft.rfft(fake, dim=-1).abs()
    real_fft = torch.fft.rfft(real, dim=-1).abs()
    return torch.mean((fake_fft - real_fft) ** 2)