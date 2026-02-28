import numpy as np

from src.esn import *
from src.signal import *
from src.transition_model import *
from src.readout import *
from src.rollout import *
from src.metrics import *
from src.visualization import *

def main():
    T = 220
    washout = 20
    M = 50

    t = np.linspace(0,1,T)
    rng = np.random.default_rng(0)

    Win, Wres = build_true_esn(M, 0.95, seed=1)

    y = make_random_pl_2dip(t, rng)
    X = run_true_esn_states(y, Win, Wres)

    print("Example run complete")

if __name__ == "__main__":
    main()