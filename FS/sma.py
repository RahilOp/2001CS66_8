import numpy as np
from FS.functionHO import Fun


def jfs(feat, label, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    z = 0.03  # Control local and global

    if "N" in opts:
        N = opts["N"]
    if "T" in opts:
        max_Iter = opts["T"]
    if "z" in opts:
        z = opts["z"]
    if "thres" in opts:
        thres = opts["thres"]

    fun = Fun

    # Number of dimensions (features)
    dim = feat.shape[1]

    # Initial population
    X = np.random.uniform(lb, ub, (N, dim))

    # Initialize variables
    fit = np.zeros(N)
    fitG = float("inf")
    W = np.zeros((N, dim))
    curve = np.full(max_Iter, float("inf"))
    t = 1

    # Iteration loop
    while t <= max_Iter:
        # Fitness evaluation
        for i in range(N):
            # Convert X[i, :] to binary based on threshold
            binary_solution = (X[i, :] > thres).astype(int)
            # Calculate fitness using function Fun
            fit[i] = fun(feat, label, binary_solution, opts)
            # Update global best fitness and solution
            if fit[i] < fitG:
                fitG = fit[i]
                Xb = X[i, :]

        # Sort fitness and obtain sorted indices
        idxS = np.argsort(fit)
        fitS = fit[idxS]

        # Compute best and worst fitness
        bF = np.min(fit)
        wF = np.max(fit)

        # Compute W
        for i in range(N):
            for d in range(dim):
                r = np.random.rand()
                if i <= N / 2:
                    W[idxS[i], d] = 1 + r * np.log10(
                        ((bF - fitS[i]) / (bF - wF + np.finfo(float).eps)) + 1
                    )
                else:
                    W[idxS[i], d] = 1 - r * np.log10(
                        ((bF - fitS[i]) / (bF - wF + np.finfo(float).eps)) + 1
                    )

        # Compute `a` and `b`
        a = np.arctanh(-(t / max_Iter) + 1)
        b = 1 - (t / max_Iter)

        # Update the solutions
        for i in range(N):
            if np.random.rand() < z:
                # Randomly initialize the solution
                X[i, :] = np.random.uniform(lb, ub, dim)
            else:
                # Calculate probabilities
                p = np.tanh(abs(fit[i] - fitG))
                # Random values for `vb` and `vc`
                vb = np.random.uniform(-a, a, dim)
                vc = np.random.uniform(-b, b, dim)

                for d in range(dim):
                    r = np.random.rand()
                    # Select two random individuals
                    A = np.random.randint(0, N)
                    B = np.random.randint(0, N)

                    if r < p:
                        # Update solution based on W, vb, and random individuals
                        X[i, d] = Xb[d] + vb[d] * (W[i, d] * X[A, d] - X[B, d])
                    else:
                        # Update solution based on vc
                        X[i, d] = vc[d] * X[i, d]

            # Apply boundary constraints
            X[i, :] = np.clip(X[i, :], lb, ub)

        # Save the current best fitness value
        curve[t - 1] = fitG
        print(f"\nIteration {t} Best (SMA)= {curve[t - 1]}")
        t += 1

    # Convert 1-based index to 0-based index for final feature selection
    Sf = np.where(Xb > thres)[0]

    # Select the subset of features based on selected indices
    sFeat = feat[:, Sf]

    # Store results in a dictionary
    SMA = {"sf": Sf, "ff": sFeat, "nf": len(Sf), "c": curve, "f": feat, "l": label}

    return SMA
