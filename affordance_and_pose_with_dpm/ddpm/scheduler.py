import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod

class BaseSchedule(ABC):
    """Base class."""
    def __init__(self, T, device):
        self.T = T
        self.device = device

    @abstractmethod
    def compute_betas(self):
        pass

    @property
    def betas(self):
        return self.compute_betas()

    @property
    def alphas(self):
        return 1.0 - self.betas

    @property
    def alpha_bar(self):
        return torch.cumprod(self.alphas, dim=0)

class LinearSchedule(BaseSchedule):
    """Linear schedule."""
    def __init__(self, T, device, beta_start, beta_end):
        super().__init__(T, device)
        self.beta_start = beta_start
        self.beta_end = beta_end

    def compute_betas(self):
        return torch.linspace(self.beta_start, self.beta_end, self.T, device=self.device) 
        

class CosineSchedule(BaseSchedule):
    """Cosine schedule."""
    def __init__(self, T, device, s=0.008):
        super().__init__(T, device)
        self.s = s

    def compute_betas(self):
        steps = torch.linspace(0, self.T, self.T + 1, device=self.device) / self.T
        alphas_bar_cont = torch.cos(((steps + self.s) / (1 + self.s)) * (torch.pi / 2)) ** 2
        betas = []
        for t in range(self.T):
            beta_t = min(1 - alphas_bar_cont[t + 1].item() / alphas_bar_cont[t].item(), 0.999)
            betas.append(beta_t)
        return torch.tensor(betas, device=self.device)

def create_schedule(schedule_type, T, device, **kwargs):
    """Create schedule."""
    schedule_type = schedule_type.lower()
    if schedule_type == "linear":
        return LinearSchedule(T, device, kwargs.get("beta_start"), kwargs.get("beta_end"))
    elif schedule_type == "cosine":
        return CosineSchedule(T, device, kwargs.get("s", 0.008))
    else:
        raise ValueError("Unsupported schedule type.")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = 1000

    linear_schedule = create_schedule("linear", T, device, beta_start=0.0001, beta_end=0.02)
    cosine_schedule = create_schedule("cosine", T, device, s=0.008)

    alpha_bar_linear = linear_schedule.alpha_bar.cpu().numpy()
    alpha_bar_cosine = cosine_schedule.alpha_bar.cpu().numpy()
    t = np.linspace(0, 1, T)
    plt.figure(figsize=(8, 6))
    plt.plot(t/T, alpha_bar_linear, label="Linear")
    plt.plot(t/T, alpha_bar_cosine, label="Cosine")
    plt.xlabel("Normalized time")
    plt.ylabel("Alpha_bar")
    plt.title("Alpha_bar vs. Time")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
