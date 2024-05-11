import numpy as np
from scipy.stats import levy
from FS.functionHO import Fun


def jfs(feat, label, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    beta = 1.5  # Levy component
    P = 0.5  # Constant
    FADs = 0.2  # Fish aggregating devices effect

    # Set parameters from options
    N = opts.get("N", 100)
    max_Iter = opts.get("T", 100)
    thres = opts.get("thres", thres)
    P = opts.get("P", P)
    FADs = opts.get("FADs", FADs)

    # Objective function
    fun = Fun

    # Number of dimensions
    dim = feat.shape[1]

    # Initialize population (9)
    X = np.random.uniform(lb, ub, size=(N, dim))

    # Initialize fitness and global best
    fit = np.zeros(N)
    fitG = np.inf
    curve = np.zeros(max_Iter)
    t = 0
    Xgb = None

    # Main loop
    while t < max_Iter:
        # Calculate fitness for each individual
        for i in range(N):
            fit[i] = fun(feat, label, (X[i, :] > thres), opts)
            # Update global best
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]

        # Memory saving for comparison
        if t == 0:
            fitM = np.copy(fit)
            Xmb = np.copy(X)

        # Update individuals if the memory is better
        for i in range(N):
            if fitM[i] < fit[i]:
                fit[i] = fitM[i]
                X[i, :] = Xmb[i, :]

        # Update memory for next iteration
        Xmb = np.copy(X)
        fitM = np.copy(fit)

        # Calculate elite population and adaptive parameter
        Xe = np.tile(Xgb, (N, 1))
        CF = (1 - (t / max_Iter)) ** (2 * (t / max_Iter))

        # Phase updates
        if t <= max_Iter / 3:  # First phase (12)
            # Brownian random number
            RB = np.random.randn(dim)
            for i in range(N):
                R = np.random.rand()
                # Calculate stepsize
                stepsize = RB * (Xe[i, :] - RB * X[i, :])
                # Update positions
                X[i, :] += P * R * stepsize
                # Apply boundaries
                X[i, :] = np.clip(X[i, :], lb, ub)

        elif max_Iter / 3 < t <= 2 * max_Iter / 3:  # Second phase (13-14)
            for i in range(N):
                if i < N / 2:  # First half update
                    RL = 0.05 * levy.rvs(beta, size=dim)
                    for d in range(dim):
                        R = np.random.rand()
                        stepsize = RL[d] * (Xe[i, d] - RL[d] * X[i, d])
                        X[i, d] += P * R * stepsize
                else:  # Second half update
                    RB = np.random.randn(dim)
                    for d in range(dim):
                        # Calculate stepsize as a scalar
                        stepsize = RB[d] * (RB[d] * Xe[i, d] - X[i, d])
                        X[i, d] = Xe[i, d] + P * CF * stepsize

                # Apply boundaries
                X[i, :] = np.clip(X[i, :], lb, ub)

        else:  # Third phase (15)
            for i in range(N):
                RL = 0.05 * levy.rvs(beta, size=dim)
                for d in range(dim):
                    # Calculate stepsize as a scalar
                    stepsize = RL[d] * (RL[d] * Xe[i, d] - X[i, d])
                    X[i, d] = Xe[i, d] + P * CF * stepsize
                # Apply boundaries
                X[i, :] = np.clip(X[i, :], lb, ub)

        # Calculate fitness for each individual and update global best
        for i in range(N):
            fit[i] = fun(feat, label, (X[i, :] > thres), opts)
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]

        # Update memory for next iteration
        for i in range(N):
            if fitM[i] < fit[i]:
                fit[i] = fitM[i]
                X[i, :] = Xmb[i, :]

        Xmb = np.copy(X)
        fitM = np.copy(fit)

        # Eddy formation and FADs effect (16)
        if np.random.rand() <= FADs:
            for i in range(N):
                # Compute U
                U = np.random.rand(dim) < FADs
                for d in range(dim):
                    R = np.random.rand()
                    # Update positions with CF and boundary conditions
                    X[i, d] += CF * (lb + R * (ub - lb)) * U[d]
                X[i, :] = np.clip(X[i, :], lb, ub)
        else:
            # Compute uniform random number and define two prey randomly
            r = np.random.rand()
            Xr1 = X[np.random.permutation(N), :]
            Xr2 = X[np.random.permutation(N), :]
            for i in range(N):
                for d in range(dim):
                    # Update positions using FADs and boundary conditions
                    X[i, d] += (FADs * (1 - r) + r) * (Xr1[i, d] - Xr2[i, d])
                X[i, :] = np.clip(X[i, :], lb, ub)

        # Save the best fitness value at each iteration
        curve[t] = fitG
        print(f"\nIteration {t + 1} Best (MPA)= {curve[t]}")

        # Increment iteration counter
        t += 1

    # Select features based on the best global solution
    Pos = np.arange(dim)
    Sf = Pos[(Xgb > thres)].tolist()
    sFeat = feat[:, Sf]

    # Store results
    MPA = {
        "sf": Sf,  # 0-based index list of selected features
        "ff": sFeat,
        "nf": len(Sf),
        "c": curve,
        "f": feat,
        "l": label,
    }

    return MPA


# Levy function
def jLevy(beta, dim):
    num = np.exp(np.log(np.gamma(1 + beta)) + np.log(np.sin(np.pi * beta / 2)))
    deno = np.exp(
        np.log(np.gamma((1 + beta) / 2)) + np.log(beta) + np.log(2) * ((beta - 1) / 2)
    )
    sigma = (num / deno) ** (1 / beta)
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    LF = u / (np.abs(v) ** (1 / beta))
    return LF
