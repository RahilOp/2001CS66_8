import numpy as np
from FS.functionHO import Fun


def jfs(feat, label, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    step_e = 0.05  # Control number of sunspot
    ratio = 0.2  # Control step
    type_ = 1  # Type 1 or 2

    if "T" in opts:
        max_iter = opts["T"]
    if "N" in opts:
        N = opts["N"]
    if "ratio" in opts:
        ratio = opts["ratio"]
    if "stepe" in opts:
        step_e = opts["stepe"]
    if "ty" in opts:
        type_ = opts["ty"]
    if "thres" in opts:
        thres = opts["thres"]

    # Objective function
    fun = Fun

    # Number of dimensions
    dim = feat.shape[1]

    # Initial
    X = np.random.uniform(lb, ub, size=(N, dim))

    # Fitness
    fit = np.zeros(N)
    fitG = np.inf
    for i in range(N):
        fit[i] = fun(feat, label, X[i, :], opts)
        # Global update
        if fit[i] < fitG:
            fitG = fit[i]
            Xgb = X[i, :]

    # Pre
    Xnew = np.zeros((N, dim))

    curve = np.zeros(max_iter)
    curve[0] = fitG
    t = 1

    # Iteration
    while t < max_iter:
        # Sort butterfly
        idx = np.argsort(fit)
        fit = fit[idx]
        X = X[idx, :]

        # Proportion of sunspot butterfly decreasing from 0.9 to ratio
        num_sun = round(N * (0.9 - (0.9 - ratio) * (t / max_iter)))

        # Define a, linearly decrease from 2 to 0
        a = 2 - 2 * (t / max_iter)

        # Step update (5)
        step = 1 - (1 - step_e) * (t / max_iter)

        # {1} Some butterflies with better fitness: Sunspot butterfly
        for i in range(num_sun):
            # Randomly select a butterfly k, but not equal to i
            k = np.random.choice(np.delete(np.arange(N), i))
            if type_ == 1:
                # [Version 1]
                # Randomly select a dimension
                J = np.random.randint(dim)
                # Random number in [-1,1]
                r1 = -1 + 2 * np.random.rand()
                # Position update (1)
                Xnew[i, :] = X[i, :]
                Xnew[i, J] = X[i, J] + (X[i, J] - X[k, J]) * r1
            elif type_ == 2:
                # [Version 2]
                # Distance
                dist = np.linalg.norm(X[k, :] - X[i, :])
                r2 = np.random.rand()
                for d in range(dim):
                    # Position update (2)
                    Xnew[i, d] = (
                        X[i, d] + ((X[k, d] - X[i, d]) / dist) * (ub - lb) * step * r2
                    )
            # Boundary
            Xnew[i, :] = np.clip(Xnew[i, :], lb, ub)

        # Fitness
        for i in range(num_sun):
            # Fitness
            Fnew = fun(feat, label, Xnew[i, :], opts)
            # Greedy selection
            if Fnew < fit[i]:
                fit[i] = Fnew
                X[i, :] = Xnew[i, :]
            # Global update
            if Fnew < fitG:
                fitG = Fnew
                Xgb = X[i, :]

        # {2} Some butterflies: Canopy butterfly
        for i in range(num_sun, N):
            # Randomly select a sunspot butterfly
            k = np.random.randint(num_sun)
            if type_ == 1:
                # [Version 1]
                # Randomly select a dimension
                J = np.random.randint(dim)
                # Random number in [-1,1]
                r1 = -1 + 2 * np.random.rand()
                # Position update (1)
                Xnew[i, :] = X[i, :]
                Xnew[i, J] = X[i, J] + (X[i, J] - X[k, J]) * r1
            elif type_ == 2:
                # [Version 2]
                # Distance
                dist = np.linalg.norm(X[k, :] - X[i, :])
                r2 = np.random.rand()
                for d in range(dim):
                    # Position update (2)
                    Xnew[i, d] = (
                        X[i, d] + ((X[k, d] - X[i, d]) / dist) * (ub - lb) * step * r2
                    )
            # Boundary
            Xnew[i, :] = np.clip(Xnew[i, :], lb, ub)

        # Fitness
        for i in range(num_sun, N):
            # Fitness
            Fnew = fun(feat, label, Xnew[i, :], opts)
            # Greedy selection
            if Fnew < fit[i]:
                fit[i] = Fnew
                X[i, :] = Xnew[i, :]
            else:
                # Randomly select a butterfly
                k = np.random.randint(N)
                # Fly to new location
                r3 = np.random.rand()
                r4 = np.random.rand()
                for d in range(dim):
                    # Compute D (4)
                    Dx = abs(2 * r3 * X[k, d] - X[i, d])
                    # Position update (3)
                    X[i, d] = X[k, d] - 2 * a * r4 - a * Dx
                # Boundary
                X[i, :] = np.clip(X[i, :], lb, ub)
                # Fitness
                fit[i] = fun(feat, label, X[i, :], opts)
            # Global update
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]

        curve[t] = fitG
        if type_ == 1:
            print(f"\nIteration {t + 1} Best (ABO 1)= {curve[t]}")
        elif type_ == 2:
            print(f"\nIteration {t + 1} Best (ABO 2)= {curve[t]}")
        t += 1

    # Select features
    pos = np.arange(1, dim + 1)
    Sf = pos[(Xgb > thres) == 1] - 1
    sFeat = feat[:, Sf]

    # Store results
    ABO = {"sf": Sf, "ff": sFeat, "nf": len(Sf), "c": curve, "f": feat, "l": label}

    return ABO
