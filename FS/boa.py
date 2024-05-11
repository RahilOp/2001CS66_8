import numpy as np
import random
from FS.functionHO import Fun


def jfs(feat, label, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    c = 0.01  # modular modality
    p = 0.8  # switch probability

    # Option parameters
    if "T" in opts:
        max_iter = opts["T"]
    if "N" in opts:
        N = opts["N"]
    if "c" in opts:
        c = opts["c"]
    if "p" in opts:
        p = opts["p"]
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
    Xnew = np.zeros((N, dim))
    fitG = float("inf")
    fit = np.zeros(N)

    curve = np.full(max_iter, float("inf"))
    t = 1

    # Iterations
    while t <= max_iter:
        # Fitness
        for i in range(N):
            fit[i] = fun(feat, label, (X[i, :] > thres), opts)
            # Global update
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]

        # Power component, increase from 0.1 to 0.3
        a = 0.1 + 0.2 * (t / max_iter)
        for i in range(N):
            # Compute fragrance (1)
            fragrance = c * (fit[i] ** a)
            # Random number in [0,1]
            r = random.random()
            if r < p:
                r1 = random.random()
                for d in range(dim):
                    # Move toward best butterfly (2)
                    Xnew[i, d] = X[i, d] + ((r1**2) * Xgb[d] - X[i, d]) * fragrance
            else:
                # Randomly select two butterflies
                R = np.random.permutation(N)
                J = R[0]
                K = R[1]
                r2 = random.random()
                for d in range(dim):
                    # Move randomly (3)
                    Xnew[i, d] = X[i, d] + ((r2**2) * X[J, d] - X[K, d]) * fragrance

            # Boundary
            Xnew[i] = np.clip(Xnew[i], lb, ub)

        # Replace
        X = Xnew
        # Save
        curve[t - 1] = fitG
        print(f"\nIteration {t} Best (BOA)= {curve[t - 1]}")
        t += 1

    # Select features
    pos = np.arange(dim)
    sf = pos[Xgb > thres]
    s_feat = feat[:, sf]

    # Store results
    boa = {"sf": sf, "ff": s_feat, "nf": len(sf), "c": curve, "f": feat, "l": label}

    return boa
