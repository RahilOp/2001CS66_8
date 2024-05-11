import numpy as np
from FS.functionHO import Fun


def jfs(feat, label, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    Pmut = 0.06  # mutation probability

    if "N" in opts:
        N = opts["N"]
    if "T" in opts:
        max_Iter = opts["T"]
    if "Pmut" in opts:
        Pmut = opts["Pmut"]
    if "thres" in opts:
        thres = opts["thres"]

    # Double population size: Main = Poor + Rich (1)
    N *= 2

    # Objective function
    fun = Fun

    # Number of dimensions
    dim = feat.shape[1]

    # Initial population
    X = np.random.uniform(lb, ub, size=(N, dim))

    # Initial fitness evaluation
    fit = np.zeros(N)
    fitG = np.inf

    for i in range(N):
        fit[i] = fun(feat, label, X[i, :] > thres, opts)
        # Best update
        if fit[i] < fitG:
            fitG = fit[i]
            Xgb = X[i, :]

    # Sort poor and rich (2)
    idx = np.argsort(fit)
    fit = fit[idx]
    X = X[idx, :]

    # Pre-allocation
    XRnew = np.zeros((N // 2, dim))
    XPnew = np.zeros((N // 2, dim))
    fitRnew = np.zeros(N // 2)
    fitPnew = np.zeros(N // 2)

    curve = np.zeros(max_Iter)
    curve[0] = fitG

    # Iteration loop
    t = 1
    while t < max_Iter:
        # Divide poor and rich
        XR = X[: N // 2, :]
        fitR = fit[: N // 2]
        XP = X[N // 2 :, :]
        fitP = fit[N // 2 :]

        # Select best rich individual
        idxR = np.argmin(fitR)
        XR_best = XR[idxR, :]

        # Select best poor individual
        idxP = np.argmin(fitP)
        XP_best = XP[idxP, :]

        # Compute mean of rich
        XR_mean = np.mean(XR, axis=0)

        # Compute worst of rich
        idxW = np.argmax(fitR)
        XR_worst = XR[idxW, :]

        # [Rich population]
        for i in range(N // 2):
            for d in range(dim):
                # Generate new rich (3)
                XRnew[i, d] = XR[i, d] + np.random.rand() * (XR[i, d] - XP_best[d])
                # Mutation (6)
                if np.random.rand() < Pmut:
                    # Normal random number with mean = 0 and sd = 1
                    G = np.random.randn()
                    # Mutation
                    XRnew[i, d] += G

            # Apply boundary conditions
            XRnew[i] = np.clip(XRnew[i], lb, ub)

            # Fitness of new rich
            fitRnew[i] = fun(feat, label, XRnew[i] > thres, opts)

        # [Poor population]
        for i in range(N // 2):
            for d in range(dim):
                # Calculate pattern (5)
                pattern = (XR_best[d] + XR_mean[d] + XR_worst[d]) / 3
                # Generate new poor (4)
                XPnew[i, d] = XP[i, d] + (np.random.rand() * pattern - XP[i, d])
                # Mutation (7)
                if np.random.rand() < Pmut:
                    # Normal random number with mean = 0 and sd = 1
                    G = np.random.randn()
                    # Mutation
                    XPnew[i, d] += G

            # Apply boundary conditions
            XPnew[i] = np.clip(XPnew[i], lb, ub)

            # Fitness of new poor
            fitPnew[i] = fun(feat, label, XPnew[i] > thres, opts)

        # Merge all four groups
        X = np.vstack((XR, XP, XRnew, XPnew))
        fit = np.concatenate((fitR, fitP, fitRnew, fitPnew))

        # Select the best N individuals
        idx = np.argsort(fit)
        fit = fit[idx[:N]]
        X = X[idx[:N], :]

        # Best update
        if fit[0] < fitG:
            fitG = fit[0]
            Xgb = X[0, :]

        curve[t] = fitG
        print(f"\nIteration {t + 1} Best (PRO)= {curve[t]}")
        t += 1

    # Select features based on selected index
    Pos = np.arange(dim)
    Sf = Pos[Xgb > thres].tolist()  # Convert 1-based indexing to 0-based indexing
    sFeat = feat[:, Sf]

    # Store results
    PRO = {
        "sf": Sf,  # Selected features (0-based index)
        "ff": sFeat,  # Selected feature subset
        "nf": len(Sf),  # Number of selected features
        "c": curve,  # Best fitness value curve
        "f": feat,  # Features
        "l": label,  # Labels
    }

    return PRO
