import numpy as np
from FS.functionHO import Fun


def jfs(feat, label, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    max_limit = 5  # Maximum limits allowed

    if "N" in opts:
        N = opts["N"]
    if "T" in opts:
        max_iter = opts["T"]
    if "max" in opts:
        max_limit = opts["max"]
    if "thres" in opts:
        thres = opts["thres"]

    # Objective function
    fun = Fun

    # Number of dimensions
    dim = feat.shape[1]

    # Divide into employ and onlooker bees
    N = N // 2

    # Initial
    X = np.random.uniform(lb, ub, size=(N, dim))

    # Fitness (9)
    fit = np.zeros(N)
    fitG = np.inf
    for i in range(N):
        fit[i] = fun(feat, label, X[i, :], opts)
        # Best food source
        if fit[i] < fitG:
            fitG = fit[i]
            Xgb = X[i, :]

    # Pre
    limit = np.zeros(N)
    V = np.zeros((N, dim))

    curve = np.zeros(max_iter)
    curve[0] = fitG
    t = 1

    # Iteration
    while t < max_iter:
        # {1} Employed bee phase
        for i in range(N):
            # Choose k randomly, but not equal to i
            k = np.setdiff1d(np.arange(N), i)
            k = np.random.choice(k)
            for d in range(dim):
                # Phi in [-1, 1]
                phi = -1 + 2 * np.random.rand()
                # Position update (6)
                V[i, d] = X[i, d] + phi * (X[i, d] - X[k, d])
            # Boundary
            V[i, :] = np.clip(V[i, :], lb, ub)

        # Fitness
        for i in range(N):
            # Fitness
            Fnew = fun(feat, label, V[i, :], opts)
            # Compare neighbor bee
            if Fnew <= fit[i]:
                # Update bee & reset limit counter
                X[i, :] = V[i, :]
                fit[i] = Fnew
                limit[i] = 0
            else:
                # Update limit counter
                limit[i] += 1

        # Minimization problem (5)
        Ifit = 1 / (1 + fit)
        # Convert probability (7)
        prob = Ifit / Ifit.sum()

        # {2} Onlooker bee phase
        m = 0
        while m < N:
            for i in range(N):
                if np.random.rand() < prob[i]:
                    # Choose k randomly, but not equal to i
                    k = np.setdiff1d(np.arange(N), i)
                    k = np.random.choice(k)
                    for d in range(dim):
                        # Phi in [-1,1]
                        phi = -1 + 2 * np.random.rand()
                        # Position update (6)
                        V[i, d] = X[i, d] + phi * (X[i, d] - X[k, d])
                    # Boundary
                    V[i, :] = np.clip(V[i, :], lb, ub)
                    # Fitness
                    Fnew = fun(feat, label, V[i, :], opts)
                    # Greedy selection
                    if Fnew <= fit[i]:
                        X[i, :] = V[i, :]
                        fit[i] = Fnew
                        limit[i] = 0
                        # Re-compute new probability (5, 7)
                        Ifit = 1 / (1 + fit)
                        prob = Ifit / Ifit.sum()
                    else:
                        limit[i] += 1
                    m += 1
            # Reset i if it reaches N
            if i >= N:
                i = 0

        # {3} Scout bee phase
        for i in range(N):
            if limit[i] >= max_limit:
                X[i, :] = np.random.uniform(lb, ub, size=dim)
                # Reset limit
                limit[i] = 0
                # Fitness
                fit[i] = fun(feat, label, X[i, :], opts)
            # Best food source
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]

        curve[t] = fitG
        print(f"\nIteration {t + 1} Best (ABC)= {curve[t]}")
        t += 1

    # Select features based on selected index
    Pos = np.arange(1, dim + 1)
    Sf = Pos[(Xgb > thres) == 1] - 1
    sFeat = feat[:, Sf]

    # Store results
    ABC = {"sf": Sf, "ff": sFeat, "nf": len(Sf), "c": curve, "f": feat, "l": label}

    return ABC
