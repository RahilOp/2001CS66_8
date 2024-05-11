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

    # Number of dimensions (features)
    dim = feat.shape[1]

    # Initial population
    X = np.zeros((N, dim))
    for i in range(N):
        for d in range(dim):
            X[i, d] = lb + (ub - lb) * np.random.rand()

    # Fitness
    fit = np.zeros(N)
    fitG = np.inf
    for i in range(N):
        fit[i] = fun(feat, label, X[i, :] > thres, opts)
        # Global best
        if fit[i] < fitG:
            fitG = fit[i]
            Xgb = X[i, :]

    # Pre-allocate variables
    Xi = np.zeros(dim)
    Xj = np.zeros(dim)

    curve = np.zeros(max_Iter)
    curve[0] = fitG
    t = 2

    # Iteration loop
    while t <= max_Iter:
        for i in range(N):
            # Mutualism phase
            R = np.random.permutation(np.delete(np.arange(N), i))
            J = R[0]
            BF1 = np.random.randint(1, 3)
            BF2 = np.random.randint(1, 3)

            for d in range(dim):
                MV = (X[i, d] + X[J, d]) / 2
                Xi[d] = X[i, d] + np.random.rand() * (Xgb[d] - MV * BF1)
                Xj[d] = X[J, d] + np.random.rand() * (Xgb[d] - MV * BF2)

            # Boundary constraints
            Xi = np.clip(Xi, lb, ub)
            Xj = np.clip(Xj, lb, ub)

            # Fitness
            fitI = fun(feat, label, Xi > thres, opts)
            fitJ = fun(feat, label, Xj > thres, opts)

            # Update if better solution
            if fitI < fit[i]:
                fit[i] = fitI
                X[i, :] = Xi

            if fitJ < fit[J]:
                fit[J] = fitJ
                X[J, :] = Xj

            # Commensalism phase
            R = np.random.permutation(np.delete(np.arange(N), i))
            J = R[0]

            for d in range(dim):
                r1 = np.random.uniform(-1, 1)
                Xi[d] = X[i, d] + r1 * (Xgb[d] - X[J, d])

            # Boundary constraints
            Xi = np.clip(Xi, lb, ub)

            # Fitness
            fitI = fun(feat, label, Xi > thres, opts)

            # Update if better solution
            if fitI < fit[i]:
                fit[i] = fitI
                X[i, :] = Xi

            # Parasitism phase
            R = np.random.permutation(np.delete(np.arange(N), i))
            J = R[0]

            PV = X[i, :].copy()
            r_dim = np.random.permutation(dim)
            dim_no = np.random.randint(1, dim + 1)

            for d in range(dim_no):
                PV[r_dim[d]] = lb + (ub - lb) * np.random.rand()

            # Boundary constraints
            PV = np.clip(PV, lb, ub)

            # Fitness
            fitPV = fun(feat, label, PV > thres, opts)

            # Replace parasite if it is better than J
            if fitPV < fit[J]:
                fit[J] = fitPV
                X[J, :] = PV

        # Update global best
        for i in range(N):
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]

        curve[t - 1] = fitG
        print(f"\nIteration {t} GBest (SOS) = {curve[t - 1]:.6f}")
        t += 1

    # Convert 1-based index to 0-based index for feature selection
    Pos = np.arange(1, dim + 1)
    Sf = np.where(Xgb > thres)[0] - 1  # Convert to 0-based index

    # Select the subset of features based on selected indices
    sFeat = feat[:, Sf]

    # Store results in a dictionary
    SOS = {"sf": Sf, "ff": sFeat, "nf": len(Sf), "c": curve, "f": feat, "l": label}

    return SOS
