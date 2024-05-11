import numpy as np
from FS.functionHO import Fun


def jfs(feat, label, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5

    if "N" in opts:
        N = opts["N"]
    if "T" in opts:
        max_iter = opts["T"]
    if "thres" in opts:
        thres = opts["thres"]

    # Objective function
    fun = Fun

    # Number of dimensions
    dim = feat.shape[1]
    # Initial (26)
    X = np.zeros((N, dim))
    for i in range(N):
        for d in range(dim):
            X[i, d] = lb + (ub - lb) * np.random.rand()

    # Fitness
    fit = np.zeros(N)
    fitG = np.inf
    for i in range(N):
        fit[i] = fun(feat, label, (X[i, :] > thres), opts)
        # Best
        if fit[i] < fitG:
            fitG = fit[i]
            Xb = X[i, :]

    # Pre
    V = np.zeros((N, dim))

    curve = []
    curve.append(fitG)
    t = 1
    # Iteration
    while t < max_iter:
        # Compute mean position (22)
        M = np.mean(X, axis=0)
        for i in range(N):
            alpha = np.random.rand()
            # [Local exploitation]
            if alpha > 0.5:
                # Random numbers
                a = np.random.rand()
                b = np.random.rand()
                for d in range(dim):
                    # Compute mean (19)
                    mu = (1 / 3) * (X[i, d] + Xb[d] + M[d])
                    # Compute standard deviation (20)
                    delta = np.sqrt(
                        (1 / 3)
                        * ((X[i, d] - mu) ** 2 + (Xb[d] - mu) ** 2 + (M[d] - mu) ** 2)
                    )
                    # Compute eta (21)
                    lambda1 = np.random.rand()
                    lambda2 = np.random.rand()
                    if a <= b:
                        eta = np.sqrt(-1 * np.log(lambda1)) * np.cos(
                            2 * np.pi * lambda2
                        )
                    else:
                        eta = np.sqrt(-1 * np.log(lambda1)) * np.cos(
                            2 * np.pi * lambda2 + np.pi
                        )
                    # Generate normal distribution (18)
                    V[i, d] = mu + delta * eta
            # [Global Exploitation]
            else:
                # Random three vectors but not i
                RN = np.random.permutation(N)
                RN = RN[RN != i]
                p1, p2, p3 = RN[:3]
                # Random beta
                beta = np.random.rand()
                # Normal random number: zero mean & unit variance
                lambda3 = np.random.randn()
                lambda4 = np.random.randn()
                # Get v1 (24)
                if fit[i] < fit[p1]:
                    v1 = X[i, :] - X[p1, :]
                else:
                    v1 = X[p1, :] - X[i, :]
                # Get v2 (25)
                if fit[p2] < fit[p3]:
                    v2 = X[p2, :] - X[p3, :]
                else:
                    v2 = X[p3, :] - X[p2, :]
                # Generate new position (23)
                for d in range(dim):
                    V[i, d] = (
                        X[i, d]
                        + beta * abs(lambda3) * v1[d]
                        + (1 - beta) * abs(lambda4) * v2[d]
                    )

            # Boundary
            V[i, :] = np.clip(V[i, :], lb, ub)

        # Fitness
        for i in range(N):
            fitV = fun(feat, label, V[i, :] > thres, opts)
            # Greedy selection (27)
            if fitV < fit[i]:
                fit[i] = fitV
                X[i, :] = V[i, :]
            # Best
            if fit[i] < fitG:
                fitG = fit[i]
                Xb = X[i, :]

        # Save
        curve.append(fitG)
        print(f"\nIteration {t} Best (GNDO)= {curve[-1]:.6f}")
        t += 1

    # Select features based on selected index
    pos = np.arange(dim)
    Sf = pos[Xb > thres]
    sFeat = feat[:, Sf]
    # Store results
    GNDO = {"sf": Sf, "ff": sFeat, "nf": len(Sf), "c": curve, "f": feat, "l": label}

    print(GNDO)
    return GNDO
