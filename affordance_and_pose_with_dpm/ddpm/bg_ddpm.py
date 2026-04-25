import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, List, Tuple, Optional


class JointAffordancePose(nn.Module):
    def __init__(
        self,
        point_encoder: nn.Module,
        text_encoder: nn.Module,
        context_net: nn.Module,
        mask_denoiser: nn.Module,
        pose_denoiser: nn.Module,
        betas: torch.Tensor,
        alpha: torch.Tensor,
        alpha_bar: torch.Tensor
    ):
        """
        Args:
            point_encoder: maps (B, N, 3) -> (B, N, C_p)
            text_encoder:  maps List[str] -> (B, L, C_t)
            context_net:   maps (points, f_pts, f_txt) -> (B, N, C_c)
            mask_denoiser: maps (y_t, points, ctx, t) -> (B, N, 1) logits
            pose_denoiser: maps (z_t, points, ctx, mask, t, conf) -> (B, 7, 1)
            betas:         tensor of length T, the beta schedule
            alpha, alpha_bar: forward diffusion schedules
        """
        super().__init__()
        # freeze encoders
        for net in (point_encoder, text_encoder, context_net):
            for p in net.parameters():
                p.requires_grad = False

        self.point_encoder = point_encoder
        self.text_encoder  = text_encoder
        self.context_net   = context_net
        self.mask_denoiser = mask_denoiser
        self.pose_denoiser = pose_denoiser

        # compute schedules
        alpha_bar_prev = torch.cat([alpha_bar.new_ones(1), alpha_bar[:-1]])
        sigma = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar) * betas)

        # register buffers
        for name, buf in (
            ("beta", betas),
            ("alpha", alpha),
            ("alpha_bar", alpha_bar),
            ("alpha_bar_prev", alpha_bar_prev),
            ("sigma", sigma),
        ):
            self.register_buffer(name, buf)

    def train(self, mode: bool = True):
        super().train(mode)
        # keep encoders in eval
        for net in (self.point_encoder, self.text_encoder, self.context_net):
            net.eval()
        return self

    def encode_ctx(self, points: torch.Tensor, desc: List[str]) -> torch.Tensor:
        """
        Build per-point context features.
        """
        with torch.no_grad():
            f_p = self.point_encoder(points)           # (B, N, C_p)
            f_t = self.text_encoder(desc)               # (B, L, C_t)
            return self.context_net(points, f_p, f_t)   # (B, N, C_c)

    @staticmethod
    def xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a + b) % 2

    def q_sample_bernoulli(self, y0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Eq.3 forward noising: y_t ~ Bernoulli(alpha_bar[t] * y0 + (1-alpha_bar[t])/2)
        """
        pa = self.alpha_bar[t.long()].view(-1, 1, 1)
        p = pa * y0 + (1 - pa) / 2
        y_t = torch.bernoulli(p)
        return y_t, self.xor(y_t, y0)

    def theta_posterior(
        self,
        y_t: torch.Tensor,
        y0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Discrete posterior
        q(y_{t-1}|y_t, y0) = Bernoulli(theta_post)
        """
        t_idx = t.long()
        alpha_t = self.alpha[t_idx].view(-1, 1, 1)
        alpha_bar_prev = torch.cat(
            [torch.ones(1, device=y_t.device), self.alpha_bar[:-1]], dim=0
        )[t_idx].view(-1, 1, 1)

        # forward kernel two channels
        p_t0 = (1 - alpha_t) / 2.0
        p_t1 = alpha_t + (1 - alpha_t) / 2.0
        pt = torch.stack(
            [p_t0.expand_as(y_t), p_t1.expand_as(y_t)], dim=-1
        )  # (B,N,2)

        # y0 prior two channels
        p0_0 = alpha_bar_prev * (1 - y0) + (1 - alpha_bar_prev) / 2.0
        p0_1 = alpha_bar_prev * y0 + (1 - alpha_bar_prev) / 2.0
        p0 = torch.stack([p0_0, p0_1], dim=-1)  # (B,N,2)

        logits = (pt + 1e-8).log() + (p0 + 1e-8).log()
        probs = logits.softmax(-1)               # normalize
        return probs[..., 1:2].squeeze(-1)                   # P(y_{t-1}=1), shape (B,N,1)


    def loss_mask(self, x0: torch.Tensor, points: torch.Tensor, ctx: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        BCE on eps + KL true vs. pred posterior
        """
        y_t, eps = self.q_sample_bernoulli(x0, t)
        logits = self.mask_denoiser(y_t, points, ctx, t.float())
        bce = F.binary_cross_entropy_with_logits(logits.squeeze(-1), eps.squeeze(-1))

        pa = self.alpha_bar[t.long()].view(-1,1,1)
        y0_hat = ((y_t - (1-pa)/2) / pa).clamp(0,1)
        t_true = self.theta_posterior(y_t, x0, t).clamp(1e-6,1-1e-6)
        t_pred = self.theta_posterior(y_t, y0_hat, t).clamp(1e-6,1-1e-6)

        kl = (
            t_true * (t_true.log() - t_pred.log()) +
            (1-t_true) * ((1-t_true).log() - (1-t_pred).log())
        ).mean()

        return bce + kl

    @torch.no_grad()
    def sample_affordance_ddpm(self, T: int, points: torch.Tensor, desc: List[str], guidance: float = 1.0):
        """
        DDPM sampling for masks: y_T ~ Bernoulli(0.5) -> y0
        """
        B, N = points.shape[:2]
        ctx_c = self.encode_ctx(points, desc)
        ctx_u = self.encode_ctx(points, [""]*B)
        y     = torch.bernoulli(0.5*torch.ones(B, N, 1, device=points.device))

        final_probs = None
        for t in range(T-1, -1, -1):
            tf   = torch.full((B,), t, device=points.device).float()
            logc = self.mask_denoiser(y, points, ctx_c, tf)
            logu = self.mask_denoiser(y, points, ctx_u, tf)
            logit = logu + guidance * (logc - logu)

            eps = torch.sigmoid(logit) 

            y0_est = torch.abs(y - eps).clamp(0, 1)
            mu = self.theta_posterior(y, y0_est, tf)

            y = torch.bernoulli(mu) if t > 0 else (mu > 0.5).float()

            final_probs = mu

        return ctx_c, ctx_u, y, final_probs

    @torch.no_grad()
    def sample_affordance_ddim(
        self,
        T: int,
        points: torch.Tensor,
        desc: List[str],
        guidance: float = 1.0,
        num_steps: int = 100
    ):
        """
        Reduced-step DDIM sampling for masks: y_T ~ Bernoulli(0.5) -> y0
        """

        
        B, N = points.shape[:2]
        ctx_c = self.encode_ctx(points, desc)
        ctx_u = self.encode_ctx(points, [""]*B)
        y     = torch.bernoulli(0.5*torch.ones(B, N, 1, device=points.device))

        seq = torch.linspace(0, T-1, steps=num_steps+1, device=points.device) \
                 .long().unique_consecutive().tolist()[::-1]
        ab_list = self.alpha_bar[seq]
        sig_list = self.sigma[seq]
        prev_list = [1.0] + ab_list[:-1].tolist()

        final_probs = 0
        for i, t in enumerate(seq[:-1]):
            prev_ab = prev_list[i]
            ab_t    = ab_list[i]
            sig_t   = sig_list[i]

            tf = torch.full((B,), t, device=points.device).float()
            lc = self.mask_denoiser(y, points, ctx_c, tf)
            lu = self.mask_denoiser(y, points, ctx_u, tf)
            eps_hat = torch.sigmoid(lu + guidance * (lc - lu))

            diff = torch.abs(y - eps_hat)
            mu = sig_t * y + (prev_ab - sig_t * ab_t) * diff + ((1-prev_ab) - (1-ab_t)*sig_t)/2 
            # print(mu)
            mu = torch.clamp(mu, 0.0, 1.0)


            y = torch.bernoulli(mu) if i < len(seq)-2 else (mu > 0.5).float()
            final_probs = mu

        return ctx_c, ctx_u, y, final_probs

    def q_sample_gaussian(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gaussian forward noising for pose branch: x_t = sqrt(alpha_bar[t]) * x0 + sqrt(1-alpha_bar[t]) * eps
        """
        idx = t.long()
        eps = torch.randn_like(x0)
        sa = self.alpha_bar[idx].sqrt().view(-1,1,1)
        sb = (1-self.alpha_bar[idx]).sqrt().view(-1,1,1)
        return sa * x0 + sb * eps, eps

    def loss_pose(
        self,
        x0: torch.Tensor,
        points: torch.Tensor,
        ctx: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
        conf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Gaussian DDPM loss for pose branch.
        """
        x_t, eps = self.q_sample_gaussian(x0, t)
        pred     = self.pose_denoiser(x_t, points, ctx, mask, t.float(), conf)
        return F.mse_loss(pred.view_as(eps), eps)

    @torch.no_grad()
    def sample_pose_ddpm(
        self,
        T: int,
        points: torch.Tensor,
        ctx_c: torch.Tensor,
        ctx_u: torch.Tensor,
        mask: torch.Tensor,
        conf: Optional[torch.Tensor] = None,
        guidance: float = 1.0
    ) -> torch.Tensor:
        """
        Reverse DDPM sampling of a 7-D pose latent.
        """
        B = points.size(0)
        device = points.device
        x = torch.randn(B, 7, 1, device=device)

        for i in reversed(range(T)):
            t_idx  = torch.full((B,), i, dtype=torch.long, device=device)
            t_cond = t_idx.float()
            eps_c = self.pose_denoiser(x, points, ctx_c, mask, t_cond, conf)
            eps_u = self.pose_denoiser(x, points, ctx_u, mask, t_cond, conf)
            eps = eps_u + guidance * (eps_c - eps_u)


            alpha_t =  self.alpha[t_idx].view(-1, 1, 1)           # (B,1,1)
            ab_t    =  self.alpha_bar[t_idx].view(-1, 1, 1)      # (B,1,1)
            sqrt_alpha_t            = alpha_t.sqrt()            # (B,1,1)
            one_minus_alpha_t       = (1 - alpha_t)             # (B,1,1)
            sqrt_one_minus_ab_t     = (1 - ab_t).sqrt()         # (B,1,1)

            mean_p = (1.0 / sqrt_alpha_t) * (x - (one_minus_alpha_t / sqrt_one_minus_ab_t) * eps)

            if i > 0:
                noise_p = torch.randn_like(x)
                sigma_t  = self.beta[t_idx].view(-1,1,1).sqrt()        # (B,1,1)
                x = mean_p + sigma_t * noise_p
            else:
                x = mean_p

        x0    = x.view(B, 7)
        trans = x0[:, :3]
        quat  = x0[:, 3:]
        quat  = quat / quat.norm(dim=1, keepdim=True)
        return torch.cat([trans, quat], dim=1).unsqueeze(-1)

        
    import torch
    from typing import Optional

    @torch.no_grad()
    def sample_pose_ddim(
        self,
        T: int,
        points: torch.Tensor,
        ctx_c: torch.Tensor,
        ctx_u: torch.Tensor,
        mask: torch.Tensor,
        conf: Optional[torch.Tensor] = None,
        guidance: float = 1.0,
        num_steps: int = 100
    ) -> torch.Tensor:
        """
        Reduced-step DDIM sampling of a 7-D pose latent.
        """
        B = points.size(0)
        device = points.device
        x = torch.randn(B, 7, 1, device=points.device)
        guidance = float(guidance)


        seq = torch.linspace(0, T-1, steps=num_steps+1, device=points.device) \
                 .long().unique_consecutive().tolist()[::-1]
        
        # print(f"self.alpha_bar length: {len(self.alpha_bar)}")
        # print(f"seq: {seq}, max seq index: {max(seq)}")

        ab = self.alpha_bar[seq]


        if conf is not None and conf.numel() > 0:
            conf = conf.to(device)

        for i, t in enumerate(seq[:-1]):
            tf   = torch.full((B,), t, device=points.device).float()
            ec   = self.pose_denoiser(x, points, ctx_c, mask, tf, conf)
            eu   = self.pose_denoiser(x, points, ctx_u, mask, tf, conf)
            e_hat = eu + guidance * (ec - eu)

            ab_t, ab_s = ab[i].view(-1, 1, 1), ab[i+1].view(-1, 1, 1)
            sa_t = ab_t.sqrt()
            sb_t = (1-ab_t).sqrt()
            x0_pred = (x - sb_t * e_hat) / sa_t
            x = ab_s.sqrt() * x0_pred + (1-ab_s).sqrt() * e_hat

        x0    = x.view(B, 7)
        trans = x0[:, :3]
        quat  = x0[:, 3:]
        quat  = quat / quat.norm(dim=1, keepdim=True)
        return torch.cat([trans, quat], dim=1).unsqueeze(-1)