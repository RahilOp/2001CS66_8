import numpy as np
import random
from FS.functionHO import Fun


def jfs(feat, label, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    M = 2  # movement parameter
    f = 3  # control parameter
    l = 2  # control parameter

    # Option parameters
    if "T" in opts:
        max_iter = opts["T"]
    if "N" in opts:
        N = opts["N"]
    if "M" in opts:
        M = opts["M"]
    if "f" in opts:
        f = opts["f"]
    if "l" in opts:
        l = opts["l"]
    if "thres" in opts:
        thres = opts["thres"]

    # Objective function
    fun = Fun

    # Number of dimensions
    dim = feat.shape[1]

    # Initial
    X = np.zeros((N, dim))
    for i in range(N):
        for d in range(dim):
            X[i, d] = lb + (ub - lb) * random.random()

    # Pre
    fit = np.zeros(N)
    fitG = float("inf")

    curve = np.full(max_iter, float("inf"))
    t = 1

    # Iterations
    while t <= max_iter:
        for i in range(N):
            # Fitness
            fit[i] = fun(feat, label, (X[i, :] > thres), opts)
            # Best solution
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]

        # Generate radius in [0,1]
        R = random.random()
        # Time (7)
        if R > 1:
            T0 = 0
        else:
            T0 = 1
        # Temperature profile (7)
        T = T0 - (max_iter / (t - max_iter))
        for i in range(N):
            for d in range(dim):
                # P_grid (10)
                P_grid = abs(Xgb[d] - X[i, d])
                # Vector A (9)
                A = (M * (T + P_grid) * random.random()) - T
                # Vector C (11)
                C = random.random()
                # Compute function S (12)
                S = np.sqrt(f * np.exp(t / l) - np.exp(-t)) ** 2
                # Distance (8)
                Dep = abs(S * Xgb[d] - C * X[i, d])
                # Position update (13)
                X[i, d] = Xgb[d] - A * Dep
            # Boundary
            X[i, :] = np.clip(X[i, :], lb, ub)

        curve[t - 1] = fitG
        print(f"\nIteration {t} Best (EPO)= {curve[t - 1]}")
        t += 1

    # Select features
    pos = np.arange(dim)
    sf = pos[Xgb > thres]
    s_feat = feat[:, sf]

    # Store results
    epo = {"sf": sf, "ff": s_feat, "nf": len(sf), "c": curve, "f": feat, "l": label}

    return epo
