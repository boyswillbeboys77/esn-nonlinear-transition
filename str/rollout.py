import numpy as np
from .readout import apply_wout_single

def rollout_with_reset(Xobs, idx_obs, model, Wout1, T):
    obs_map = {int(t): Xobs[i] for i, t in enumerate(idx_obs)}
    K = Xobs.shape[1]
    Xhat = np.zeros((T, K))

    x = obs_map.get(0, np.zeros(K)).reshape(-1, 1)
    Xhat[0] = x.ravel()

    for t in range(T-1):
        u_hat = apply_wout_single(Wout1, x.ravel())
        x = model.predict(x, u_hat)

        if (t+1) in obs_map:
            x = obs_map[t+1].reshape(-1, 1)

        Xhat[t+1] = x.ravel()

    return Xhat