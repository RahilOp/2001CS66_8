import numpy as np
from FS.functionHO import Fun


def jfs(feat, label, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5

    max_Iter = opts.get("T", 100)
    N = opts.get("N", 100)
    thres = opts.get("thres", thres)

    # Objective function
    fun = Fun

    # Number of dimensions
    dim = feat.shape[1]

    # Initial population
    X = np.random.uniform(lb, ub, size=(N, dim))

    # Initial fitness evaluation
    fit = np.zeros(N)
    fitP = np.inf

    for i in range(N):
        fit[i] = fun(feat, label, X[i, :] > thres, opts)
        # Pathfinder update
        if fit[i] < fitP:
            fitP = fit[i]
            Xpf = X[i, :]

    # Set previous pathfinder
    Xpf_old = Xpf

    # Pre-allocation
    Xpf_new = np.zeros(dim)
    Xnew = np.zeros((N, dim))

    # Track best fitness value
    curve = np.zeros(max_Iter)
    curve[0] = fitP

    # Iteration loop
    t = 1
    while t < max_Iter:
        # Alpha and beta in [1,2]
        alpha = 1 + np.random.rand()
        beta = 1 + np.random.rand()

        # Update pathfinder
        for d in range(dim):
            # Define u2 in [-1,1]
            u2 = -1 + 2 * np.random.rand()
            # Compute A (2.6)
            A = u2 * np.exp(-(2 * t) / max_Iter)
            # Update pathfinder (2.4)
            r3 = np.random.rand()
            Xpf_new[d] = Xpf[d] + 2 * r3 * (Xpf[d] - Xpf_old[d]) + A

        # Apply boundary conditions
        Xpf_new = np.clip(Xpf_new, lb, ub)

        # Update previous path
        Xpf_old = Xpf

        # Fitness evaluation
        Fnew = fun(feat, label, Xpf_new > thres, opts)

        # Greedy selection
        if Fnew < fitP:
            fitP = Fnew
            Xpf = Xpf_new

        # Sort members and update first solution
        idx = np.argsort(fit)
        X = X[idx, :]
        fit = fit[idx]

        if Fnew < fit[0]:
            fit[0] = Fnew
            X[0, :] = Xpf_new

        # Update remaining members
        for i in range(1, N):
            # Distance (2.5)
            Dij = np.linalg.norm(X[i, :] - X[i - 1, :])

            for d in range(dim):
                # Define u1 in [-1,1]
                u1 = -1 + 2 * np.random.rand()
                # Compute epsilon (2.5)
                eps = (1 - (t / max_Iter)) * u1 * Dij
                # Define R1, R2
                r1 = np.random.rand()
                r2 = np.random.rand()
                R1 = alpha * r1
                R2 = beta * r2

                # Update member (2.3)
                Xnew[i, d] = (
                    X[i, d]
                    + R1 * (X[i - 1, d] - X[i, d])
                    + R2 * (Xpf[d] - X[i, d])
                    + eps
                )

            # Apply boundary conditions
            Xnew[i] = np.clip(Xnew[i], lb, ub)

        # Fitness evaluation and selection
        for i in range(1, N):
            Fnew = fun(feat, label, Xnew[i] > thres, opts)
            # Selection
            if Fnew < fit[i]:
                fit[i] = Fnew
                X[i] = Xnew[i]

            # Pathfinder update
            if fit[i] < fitP:
                fitP = fit[i]
                Xpf = X[i]

        # Update the best fitness value curve
        curve[t] = fitP
        print(f"\nIteration {t + 1} Best (PFA) = {curve[t]}")
        t += 1

    # Select features
    Pos = np.arange(dim)
    Sf = Pos[Xpf > thres].tolist()  # Convert 1-based indexing to 0-based indexing
    sFeat = feat[:, Sf]

    # Store results
    PFA = {
        "sf": Sf,  # Selected features (0-based index)
        "ff": sFeat,  # Selected feature subset
        "nf": len(Sf),  # Number of selected features
        "c": curve,  # Best fitness value curve
        "f": feat,  # Features
        "l": label,  # Labels
    }

    return PFA
