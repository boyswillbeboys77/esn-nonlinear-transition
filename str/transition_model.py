import numpy as np
import torch
import torch.nn as nn

class TransitionMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class NonlinearTransitionModel:
    def __init__(self, mlp, xmu, xstd, ymu, ystd):
        self.mlp = mlp
        self.xmu = xmu
        self.xstd = xstd
        self.ymu = ymu
        self.ystd = ystd

    def predict(self, x_col, u_scalar):
        x = np.asarray(x_col).reshape(-1)
        inp = np.concatenate([x, [u_scalar]]).astype(np.float32)
        inp_n = (inp - self.xmu) / self.xstd

        with torch.no_grad():
            yhat_n = self.mlp(torch.from_numpy(inp_n[None, :])).numpy().reshape(-1)

        yhat = yhat_n * self.ystd + self.ymu
        return yhat.reshape(-1, 1)