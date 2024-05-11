import numpy as np
from FS.functionHO import Fun


def jfs(feat, label, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    tau = 0.8  # constant
    sl = 0.035  # step length
    phi = 0.001  # constant
    lambda_const = 0.75  # constant

    max_Iter = opts.get("T", 100)
    N = opts.get("N", 30)
    tau = opts.get("tau", tau)
    sl = opts.get("sl", sl)
    phi = opts.get("phi", phi)
    lambda_const = opts.get("lambda", lambda_const)
    thres = opts.get("thres", thres)

    # Objective function
    fun = Fun

    # Number of dimensions
    dim = feat.shape[1]
    # Initial
    X = np.random.uniform(lb, ub, size=(N, dim))
    # Fitness
    fit = np.zeros(N)
    fitG = float("inf")

    for i in range(1, N + 1):
        fit[i - 1] = fun(feat, label, X[i - 1, :] > thres, opts)
        # Best update
        if fit[i - 1] < fitG:
            fitG = fit[i - 1]
            Xgb = X[i - 1, :]

    # Pre
    curve = np.zeros(max_Iter)
    curve[0] = fitG
    t = 2

    # Iterations
    while t <= max_Iter:
        # Rank solutions based on fitness
        idx = np.argsort(fit)
        X = X[idx, :]
        fit = fit[idx]

        # {1} Target point determination
        w = np.zeros(N)
        Xtar = np.zeros(dim)
        for i in range(1, N + 1):
            # Assign weight based on rank
            w[i - 1] = i ** (-tau)
            # Create target
            Xtar += X[i - 1, :] * w[i - 1]

        # Boundary for target
        Xtar = np.clip(Xtar, lb, ub)

        # Fitness of the target point
        fitT = fun(feat, label, Xtar > thres, opts)
        # Best update
        if fitT < fitG:
            fitG = fitT
            Xgb = Xtar

        # {2} Compute search direction
        gap = np.zeros((N, dim))
        direct = np.zeros((N, dim))

        for i in range(1, N + 1):
            if fit[i - 1] >= fitT:
                for d in range(dim):
                    # Compute gap and direction
                    gap[i - 1, d] = Xtar[d] - X[i - 1, d]
                    direct[i - 1, d] = np.sign(gap[i - 1, d])
            else:
                if np.random.rand() < np.exp(fit[i - 1] - fitT):
                    for d in range(dim):
                        # Compute gap and direction
                        gap[i - 1, d] = Xtar[d] - X[i - 1, d]
                        direct[i - 1, d] = np.sign(gap[i - 1, d])
                else:
                    for d in range(dim):
                        # Compute random direction
                        direct[i - 1, d] = np.sign(np.random.rand() - 0.5)

        # Compute step sizing function
        if np.random.rand() <= lambda_const:
            sl = sl - np.exp(t / (t - 1)) * phi * sl
        else:
            sl = sl + np.exp(t / (t - 1)) * phi * sl

        # {3} Neighbor generation
        for i in range(1, N + 1):
            for d in range(dim):
                # Update
                X[i - 1, d] = X[i - 1, d] + sl * direct[i - 1, d] * abs(X[i - 1, d])

            # Boundary
            X[i - 1, :] = np.clip(X[i - 1, :], lb, ub)

        # Fitness
        for i in range(1, N + 1):
            fit[i - 1] = fun(feat, label, X[i - 1, :] > thres, opts)
            # Best update
            if fit[i - 1] < fitG:
                fitG = fit[i - 1]
                Xgb = X[i - 1, :]

        curve[t - 1] = fitG
        print(f"\nIteration {t} Best (WSA)= {curve[t - 1]}")
        t += 1

    # Select features
    pos = np.arange(dim) + 1
    sf = pos[Xgb > thres]
    sfeat = feat[:, sf - 1]
    # Store results
    WSA = {"sf": sf, "ff": sfeat, "nf": len(sf), "c": curve, "f": feat, "l": label}

    return WSA
