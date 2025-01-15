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

        # We'll treat distance=0 as the small-step flow-matching case.
        distance = torch.zeros((B,), device=z0.device)

        return z_t, t.squeeze(), target, distance

    def get_shortcut_train_tuple(self, z0=None, z1=None, **model_kwargs):
        B = z0.shape[0]

        log_distance = torch.randint(low=0, high=7, size=(B,))
        distance = torch.pow(2, -1 * log_distance).to(z0.device)

        # --- CHANGED HERE: clamp so t + distance <= 1
        # i.e. sample t up to (1 - distance).
        # We do this by: t_ub = 1 - distance, then t = uniform(0, t_ub)
        # Reshape distance to broadcast properly for multiplication.
        distance_reshaped = distance.view(B, 1, 1)  # [B, 1, 1]
        t_ub = 1.0 - distance_reshaped
        t_rand = torch.rand((B, 1, 1), device=z0.device)
        t = t_rand * t_ub  # ensures t + distance <= 1

        z_t = (1.0 - t) * z0 + t * z1

        # First half step
        s_t = self.model(z_t, t.squeeze(), distance=distance, **model_kwargs)
        z_tpd = z_t + s_t * distance_reshaped
        tpd = t.squeeze() + distance  # [B]

        # Second half step
        s_tpd = self.model(z_tpd, tpd, distance=distance, **model_kwargs)

        # Average the two predicted directions
        target = (s_t.detach().clone() + s_tpd.detach().clone()) / 2.0

        # Our new single-step target uses step size = 2 * distance
        return z_t, t.squeeze(), target, distance * 2

    @torch.no_grad()
    def sample_ode_shortcut(self, z0=None, num_steps=None, **model_kwargs):
        """
        This method does standard small-step (num_steps) Euler integration.
        We'll leave it unchanged, but it does not show the big-step advantage.
        """
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

    # --- CHANGED HERE: Added a new 2-step sampling method
    @torch.no_grad()
    def sample_2step_shortcut(self, z0=None, **model_kwargs):
        """
        This performs exactly 2 steps over [0,1], each of size 0.5.
        That way we can test the "shortcut" behavior directly.
        """
        B = z0.shape[0]
        traj = []
        z = z0.clone()
        traj.append(z.clone())  # initial

        # Step 1: from t=0 to t=0.5
        step_size = 0.5
        t1 = torch.zeros((B,), device=self.device)  # all zeros
        pred1 = self.model(z, t1, distance=step_size, **model_kwargs)
        z = z + pred1 * step_size
        traj.append(z.clone())

        # Step 2: from t=0.5 to t=1.0
        t2 = torch.ones((B,), device=self.device) * 0.5
        pred2 = self.model(z, t2, distance=step_size, **model_kwargs)
        z = z + pred2 * step_size
        traj.append(z.clone())

        return traj
