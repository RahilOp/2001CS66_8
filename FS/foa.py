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
        max_Iter = opts["T"]
    if "thres" in opts:
        thres = opts["thres"]

    # Objective function
    fun = Fun
    # Number of dimensions
    dim = feat.shape[1]
    # Initial population
    X = np.zeros((N, dim))
    Y = np.zeros((N, dim))
    for i in range(1, N + 1):
        for d in range(1, dim + 1):
            X[i - 1, d - 1] = lb + (ub - lb) * np.random.rand()
            Y[i - 1, d - 1] = lb + (ub - lb) * np.random.rand()

    # Compute initial solutions
    S = np.zeros((N, dim))
    for i in range(1, N + 1):
        for d in range(1, dim + 1):
            # Distance between X and Y axes
            dist = np.sqrt(X[i - 1, d - 1] ** 2 + Y[i - 1, d - 1] ** 2)
            # Solution
            S[i - 1, d - 1] = 1 / dist

        # Apply boundary conditions
        S[i - 1] = np.clip(S[i - 1], lb, ub)

    # Initialize fitness and best fitness
    fit = np.zeros(N)
    fitG = float("inf")
    curve = []
    t = 1

    # Main optimization loop
    while t <= max_Iter:
        # Calculate fitness for each individual
        for i in range(1, N + 1):
            fit[i - 1] = fun(feat, label, S[i - 1, :] > thres, opts)
            # Update the best fitness if necessary
            if fit[i - 1] < fitG:
                fitG = fit[i - 1]
                Xgb = S[i - 1, :]
                # Update X and Y values
                Xb = X[i - 1, :]
                Yb = Y[i - 1, :]

        # Iterate over each individual and dimension
        for i in range(1, N + 1):
            for d in range(1, dim + 1):
                # Random values in the range [-1, 1]
                r1 = -1 + 2 * np.random.rand()
                r2 = -1 + 2 * np.random.rand()
                # Compute new X and Y
                X[i - 1, d - 1] = Xb[d - 1] + (ub - lb) * r1
                Y[i - 1, d - 1] = Yb[d - 1] + (ub - lb) * r2
                # Distance between X and Y axes
                dist = np.sqrt((X[i - 1, d - 1] ** 2) + (Y[i - 1, d - 1] ** 2))
                # Solution
                S[i - 1, d - 1] = 1 / dist

            # Apply boundary conditions
            S[i - 1] = np.clip(S[i - 1], lb, ub)

        # Store the best fitness value
        curve.append(fitG)
        print(f"Generation {t} Best (FOA) = {curve[-1]}")
        t += 1

    # Select features
    Pos = np.arange(1, dim + 1)
    Sf = Pos[(Xgb > thres)]
    sFeat = feat[:, Sf - 1]

    # Store results
    FOA = {"sf": Sf, "ff": sFeat, "nf": len(Sf), "c": curve, "f": feat, "l": label}

    return FOA
