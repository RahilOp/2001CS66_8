import numpy as np
from FS.functionHO import Fun


def jfs(feat, label, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    p = 6  # control TDR
    Wmax = 1  # maximum WEP
    Wmin = 0.2  # minimum WEP
    type = 1

    if "T" in opts:
        max_iter = opts["T"]
    if "N" in opts:
        N = opts["N"]
    if "p" in opts:
        p = opts["p"]
    if "Wmin" in opts:
        Wmin = opts["Wmin"]
    if "Wmax" in opts:
        Wmax = opts["Wmax"]
    if "ty" in opts:
        type = opts["ty"]
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
            X[i, d] = lb + (ub - lb) * np.random.rand()

    # Pre
    fit = np.zeros(N)
    fitG = float("inf")

    curve = np.inf
    t = 1

    # Iterations
    while t <= max_iter:
        # Calculate inflation rate
        for i in range(N):
            fit[i] = fun(feat, label, X[i, :] > thres, opts)
            # Best universe
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]

        # Sort universe from best to worst
        idx = np.argsort(fit)
        X_SU = X[idx, :]
        fitSU = fit[idx]

        # Elitism (first 1 is elite)
        X[0, :] = X_SU[0, :]

        # Either 1-norm or 2-norm
        if type == 1:
            # Normalize inflation rate using 2-norm
            NI = fitSU / np.sqrt(np.sum(fitSU**2))
        elif type == 2:
            # Normalize inflation rate using 1-norm
            NI = fitSU / np.sum(fitSU)

        # Normalize inverse inflation rate using 1-norm
        inv_fitSU = 1 / (1 + fitSU)
        inv_NI = inv_fitSU / np.sum(inv_fitSU)

        # Wormhole Existence Probability (3.3), increases from 0.2 to 1
        WEP = Wmin + t * ((Wmax - Wmin) / max_iter)

        # Traveling Distance Rate (3.4), decreases from 0.6 to 0
        TDR = 1 - ((t ** (1 / p)) / (max_iter ** (1 / p)))

        # Start with 2 since the first is elite
        for i in range(1, N):
            # Define black hole
            idx_BH = i
            for d in range(dim):
                # White/black hole tunnels & exchange object of universes (3.1)
                r1 = np.random.rand()
                if r1 < NI[i]:
                    # Randomly select k with roulette wheel
                    idx_WH = j_roulette_wheel_selection(inv_NI)
                    # Position update
                    X[idx_BH, d] = X_SU[idx_WH, d]

                # Local changes for universes (3.2)
                r2 = np.random.rand()
                if r2 < WEP:
                    r3 = np.random.rand()
                    r4 = np.random.rand()
                    if r3 < 0.5:
                        X[i, d] = Xgb[d] + TDR * ((ub - lb) * r4 + lb)
                    else:
                        X[i, d] = Xgb[d] - TDR * ((ub - lb) * r4 + lb)
                else:
                    X[i, d] = X[i, d]

            # Boundary
            X[i, :] = np.clip(X[i, :], lb, ub)

        curve = fitG
        print(f"Iteration {t} Best (MVO) = {curve}")

        t += 1

    # Select features
    Pos = np.arange(dim)
    Sf = Pos[(Xgb > thres)]
    sFeat = feat[:, Sf]

    # Store results
    MVO = {"sf": Sf, "ff": sFeat, "nf": len(Sf), "c": curve, "f": feat, "l": label}

    return MVO


def j_roulette_wheel_selection(prob):
    # Cumulative summation
    cumsum = np.cumsum(prob)
    # Random one value, most probable value [0 ~ 1]
    random_value = np.random.rand()
    # Roulette wheel
    for i in range(len(cumsum)):
        if cumsum[i] > random_value:
            return i
    return len(cumsum) - 1  # Return the last index if no selection was made
