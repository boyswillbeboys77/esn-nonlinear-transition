import numpy as np

def make_random_pl_2dip(t, rng,
                        center_range=(0.5, 0.52),
                        split_range=(0.1, 0.4),
                        gamma_range=(0.053, 0.053),
                        depth_range=(0.035, 0.045),
                        baseline_range=(1.0, 1.0),
                        noise_std=0.0):

    f0 = rng.uniform(*center_range)
    split = rng.uniform(*split_range)
    f1 = f0 - split / 2
    f2 = f0 + split / 2

    gamma = rng.uniform(*gamma_range)
    depth = rng.uniform(*depth_range)
    baseline = rng.uniform(*baseline_range)

    def lorentz(x, x0, g):
        return g**2 / ((x - x0)**2 + g**2)

    y = baseline \
        - depth * lorentz(t, f1, gamma) \
        - depth * lorentz(t, f2, gamma)

    if noise_std > 0:
        y += rng.normal(0, noise_std, size=len(t))

    return np.clip(y, 0.01, 0.99)