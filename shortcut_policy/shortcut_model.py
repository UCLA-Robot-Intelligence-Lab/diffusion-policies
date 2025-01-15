import torch


class ShortcutModel:
    def __init__(self, model=None, num_steps=1000, device="cuda"):
        self.model = model
        self.num_steps = num_steps
        self.device = device

    def get_train_tuple(self, z0=None, z1=None):
        B = z0.shape[0]

        # t ~ U[0, 1]
        t = torch.rand((B, 1, 1), device=z0.device)
        z_t = (1.0 - t) * z0 + t * z1
        target = z1 - z0
        distance = torch.zeros((B,), device=z0.device)

        return z_t, t.squeeze(), target, distance

    def get_shortcut_train_tuple(self, z0=None, z1=None, **model_kwargs):
        B = z0.shape[0]

        log_distance = torch.randint(low=0, high=7, size=(B,))
        distance = torch.pow(2, -1 * log_distance).to(z0.device)

        t = torch.rand((B, 1, 1), device=z0.device)
        z_t = (1.0 - t) * z0 + t * z1

        # First half step
        s_t = self.model(z_t, t.squeeze(), distance=distance, **model_kwargs)
        z_tpd = z_t + s_t * distance.view(-1, 1, 1)
        tpd = t.squeeze() + distance

        # Second half step
        s_tpd = self.model(z_tpd, tpd, distance=distance, **model_kwargs)

        # Average the two predicted directions
        target = (s_t.detach().clone() + s_tpd.detach().clone()) / 2.0

        return z_t, t.squeeze(), target, distance * 2

    @torch.no_grad()
    def sample_ode_shortcut(self, z0=None, num_steps=None, **model_kwargs):
        if num_steps is None:
            num_steps = self.num_steps

        dt = 1.0 / num_steps
        traj = []
        z = z0.clone()

        traj.append(z.clone())

        for i in range(num_steps):
            t = torch.ones((z.shape[0],), device=self.device) * (i / num_steps)
            pred = self.model(z, t, distance=dt, **model_kwargs)
            z = z + pred * dt
            traj.append(z.clone())

        return traj
