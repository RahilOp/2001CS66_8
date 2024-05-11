import numpy as np
from FS.functionHO import Fun


def jfs(feat, label, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    ST = 0.1  # switch probability

    if "T" in opts:
        max_Iter = opts["T"]
    if "N" in opts:
        N = opts["N"]
    if "ST" in opts:
        ST = opts["ST"]
    if "thres" in opts:
        thres = opts["thres"]

    # Objective function
    fun = Fun

    # Number of dimensions
    dim = feat.shape[1]

    # Initial population
    X = np.zeros((N, dim))
    for i in range(N):
        for d in range(dim):
            X[i, d] = lb + (ub - lb) * np.random.rand()

    # Fitness
    fit = np.zeros(N)
    for i in range(N):
        fit[i] = fun(feat, label, X[i, :] > thres, opts)

    # Best solution
    idx = np.argmin(fit)
    fitG = fit[idx]
    Xgb = X[idx, :]

    # Maximum and minimum number of seeds
    Smax = round(0.25 * N)
    Smin = round(0.1 * N)

    # Pre-iteration
    curve = np.zeros(max_Iter)
    curve[0] = fitG
    t = 2

    # Iteration loop
    while t <= max_Iter:
        for i in range(N):
            # Random number of seeds
            num_seed = round(Smin + np.random.rand() * (Smax - Smin))
            Xnew = np.zeros((num_seed, dim))

            for j in range(num_seed):
                # Randomly select a tree, but not i
                RN = np.random.permutation(N)
                RN = RN[RN != i]
                r = RN[0]

                for d in range(dim):
                    # Alpha in [-1,1]
                    alpha = -1 + 2 * np.random.rand()

                    if np.random.rand() < ST:
                        # Generate seed
                        Xnew[j, d] = X[i, d] + alpha * (Xgb[d] - X[r, d])
                    else:
                        # Generate seed
                        Xnew[j, d] = X[i, d] + alpha * (X[i, d] - X[r, d])

                # Boundary constraints
                Xnew[j, :] = np.clip(Xnew[j, :], lb, ub)

            # Fitness for new seeds
            for j in range(num_seed):
                Fnew = fun(feat, label, Xnew[j, :] > thres, opts)

                # Greedy selection
                if Fnew < fit[i]:
                    fit[i] = Fnew
                    X[i, :] = Xnew[j, :]

        # Best solution
        idx = np.argmin(fit)
        fitG_new = fit[idx]
        Xgb_new = X[idx, :]

        # Best update
        if fitG_new < fitG:
            fitG = fitG_new
            Xgb = Xgb_new

        # Store results in curve
        curve[t - 1] = fitG
        print(f"\nIteration {t} Best (TSA)= {curve[t - 1]:.6f}")
        t += 1

    # Convert 1-based index to 0-based index for feature selection
    Pos = np.arange(1, dim + 1)
    Sf = np.where(Xgb > thres)[0]  # Convert to 0-based index

    # Select the subset of features based on selected indices
    sFeat = feat[:, Sf]

    # Store results in a dictionary
    TSA = {"sf": Sf, "ff": sFeat, "nf": len(Sf), "c": curve, "f": feat, "l": label}

    return TSA
