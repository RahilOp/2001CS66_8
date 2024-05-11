import numpy as np
from FS.functionHO import Fun
import math


def jfs(feat, label, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    b = 1  # constant

    max_Iter = opts.get("T", 100)
    N = opts.get("N", 100)
    b = opts.get("b", b)
    thres = opts.get("thres", thres)

    # Objective function
    fun = Fun

    # Number of dimensions
    dim = feat.shape[1]

    # Initial population
    X = np.random.uniform(lb, ub, size=(N, dim))

    # Pre-allocation
    fit = np.zeros(N)
    fitG = np.inf

    # Tracking best fitness value
    curve = np.zeros(max_Iter)

    t = 0
    while t < max_Iter:
        # Calculate fitness and global best
        for i in range(N):
            fit[i] = fun(feat, label, X[i, :] > thres, opts)
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]

        # Initial flame and fitness
        if t == 0:
            idx = np.argsort(fit)
            flame = X[idx, :]
            fitF = fit[idx]
        else:
            # Merge population
            XX = np.vstack((flame, X))
            FF = np.hstack((fitF, fit))

            idx = np.argsort(FF)
            flame = XX[idx[:N], :]
            fitF = FF[idx[:N]]

        # Update number of flames
        flame_no = round(N - t * ((N - 1) / max_Iter))

        # Convergence constant, decreases linearly from -1 to -2
        r = -1 + t * (-1 / max_Iter)

        # Update moths
        for i in range(N):
            for d in range(dim):
                # Calculate T
                T = (r - 1) * np.random.rand() + 1

                # Distance between flame and moth
                dist = np.abs(flame[i, d] - X[i, d])

                # Update moth position
                if i < flame_no:
                    X[i, d] = (
                        dist * math.exp(b * T) * math.cos(2 * np.pi * T) + flame[i, d]
                    )
                else:
                    X[i, d] = (
                        dist * math.exp(b * T) * math.cos(2 * np.pi * T)
                        + flame[flame_no, d]
                    )

            # Apply boundary conditions
            X[i] = np.clip(X[i], lb, ub)

        # Store the best fitness value
        curve[t] = fitG
        print(f"\nIteration {t + 1} Best (MFO) = {curve[t]}")
        t += 1

    # Select features
    Pos = np.arange(dim)
    Sf = Pos[Xgb > thres].tolist()  # Convert 1-based indexing to 0-based indexing
    sFeat = feat[:, Sf]

    # Store results
    MFO = {
        "sf": Sf,  # Selected features (0-based index)
        "ff": sFeat,  # Selected feature subset
        "nf": len(Sf),  # Number of selected features
        "c": curve,  # Best fitness value curve
        "f": feat,  # Features
        "l": label,  # Labels
    }

    return MFO
