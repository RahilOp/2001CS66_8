import numpy as np
import random
from FS.functionHO import Fun


def jfs(feat, label, opts):
    # Parameters
    pi = opts.get("pi", 0.85)  # probability of individual learning
    pr = opts.get("pr", 0.1)  # probability of exploration learning
    N = opts.get("N")
    max_Iter = opts.get("T")

    # Objective function
    fun = Fun

    # Number of dimensions
    dim = feat.shape[1]

    # Initial
    X = jInitialPopulation(N, dim)

    # Fitness
    fit = np.zeros(N)
    fitSKD = float("inf")
    for i in range(N):
        fit[i] = fun(feat, label, X[i], opts)
        # Update SKD/gbest
        if fit[i] < fitSKD:
            fitSKD = fit[i]
            SKD = X[i].copy()

    # Get IKD/pbest
    fitIKD = fit.copy()
    IKD = X.copy()
    # Pre
    curve = np.zeros(max_Iter)
    curve[0] = fitSKD
    t = 1  # 1-based indexing as in Matlab, so starting at 1

    # Generations
    while t < max_Iter:
        for i in range(N):
            # Update solution (8)
            for d in range(dim):
                # Random probability in [0,1]
                r = random.random()
                if 0 <= r < pr:
                    # Random exploration learning operator (7)
                    if random.random() < 0.5:
                        X[i][d] = 0
                    else:
                        X[i][d] = 1
                elif pr <= r < pi:
                    X[i][d] = IKD[i][d]
                else:
                    X[i][d] = SKD[d]

        # Fitness
        for i in range(N):
            # Fitness
            fit[i] = fun(feat, label, X[i], opts)
            # Update IKD/pbest
            if fit[i] < fitIKD[i]:
                fitIKD[i] = fit[i]
                IKD[i] = X[i].copy()
            # Update SKD/gbest
            if fitIKD[i] < fitSKD:
                fitSKD = fitIKD[i]
                SKD = IKD[i].copy()

        curve[t] = fitSKD
        print(f"\nGeneration {t + 1} Best (HLO) = {curve[t]}")
        t += 1

    # Select features based on selected index
    pos = np.arange(dim)
    Sf = pos[SKD == 1]  # This will return 0-based indices
    sFeat = feat[:, Sf]

    # Store results
    HLO = {
        "sf": Sf,
        "ff": sFeat,
        "nf": len(Sf),
        "c": curve,
        "f": feat,
        "l": label,
    }

    return HLO


def jInitialPopulation(N, dim):
    X = np.zeros((N, dim))
    for i in range(N):
        for d in range(dim):
            if random.random() > 0.5:
                X[i][d] = 1
    return X
