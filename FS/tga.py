import numpy as np
from FS.functionHO import Fun


def jfs(feat, label, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    num_tree1 = 3  # size of first group
    num_tree2 = 5  # size of second group
    num_tree4 = 3  # size of fourth group
    theta = 0.8  # tree reduction rate of power
    lambda_ = 0.5  # control nearest tree

    if "T" in opts:
        max_Iter = opts["T"]
    if "N" in opts:
        N = opts["N"]
    if "N1" in opts:
        num_tree1 = opts["N1"]
    if "N2" in opts:
        num_tree2 = opts["N2"]
    if "N4" in opts:
        num_tree4 = opts["N4"]
    if "theta" in opts:
        theta = opts["theta"]
    if "lambda" in opts:
        lambda_ = opts["lambda"]
    if "thres" in opts:
        thres = opts["thres"]

    # Limit number of N4 to N1 + N2
    if num_tree4 > num_tree1 + num_tree2:
        num_tree4 = num_tree1 + num_tree2

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
        # Best
        if fit[i] < fitG:
            fitG = fit[i]
            Xgb = X[i, :]

    # Sort trees from best to worst
    sorted_indices = np.argsort(fit)
    fit = fit[sorted_indices]
    X = X[sorted_indices]

    # Initial variables
    dist = np.zeros(num_tree1 + num_tree2)
    X1 = np.zeros((num_tree1, dim))
    Xnew = np.zeros((num_tree4, dim))
    Fnew = np.zeros(num_tree4)

    curve = np.zeros(max_Iter)
    curve[0] = fitG
    t = 2

    # Iteration loop
    while t <= max_Iter:
        # Best trees group
        for i in range(num_tree1):
            r1 = np.random.rand()
            for d in range(dim):
                # Local search
                X1[i, d] = (X[i, d] / theta) + r1 * X[i, d]

            # Boundary constraints
            X1[i, :] = np.clip(X1[i, :], lb, ub)

            # Fitness
            fitT = fun(feat, label, X1[i, :] > thres, opts)

            # Greedy selection
            if fitT <= fit[i]:
                X[i, :] = X1[i, :]
                fit[i] = fitT

        # Competitive for light tree group
        X_ori = X.copy()
        for i in range(num_tree1, num_tree1 + num_tree2):
            # Neighbor tree
            for j in range(num_tree1 + num_tree2):
                if j != i:
                    # Compute Euclidean distance
                    dist[j] = np.sqrt(np.sum((X_ori[j, :] - X_ori[i, :]) ** 2))
                else:
                    # Solve same tree problem
                    dist[j] = np.inf

            # Find 2 trees with shorter distance
            nearest_indices = np.argsort(dist)[:2]
            T1 = X_ori[nearest_indices[0], :]
            T2 = X_ori[nearest_indices[1], :]

            # Alpha in [0,1]
            alpha = np.random.rand()

            for d in range(dim):
                # Compute linear combination between 2 shorter trees
                y = lambda_ * T1[d] + (1 - lambda_) * T2[d]

                # Move tree i between 2 adjacent trees
                X[i, d] += alpha * y

            # Boundary constraints
            X[i, :] = np.clip(X[i, :], lb, ub)

            # Fitness
            fit[i] = fun(feat, label, X[i, :] > thres, opts)

        # Remove and replace group
        for i in range(num_tree1 + num_tree2, N):
            for d in range(dim):
                # Generate new tree by removing worst tree
                X[i, d] = lb + (ub - lb) * np.random.rand()

            # Fitness
            fit[i] = fun(feat, label, X[i, :] > thres, opts)

        # Reproduction group
        for i in range(num_tree4):
            # Randomly select a best tree
            r = np.random.randint(0, num_tree1)
            Xbest = X[r, :]

            # Mask operator
            mask = np.random.randint(0, 2, size=dim)

            for d in range(dim):
                # Generate new solution
                Xn = lb + (ub - lb) * np.random.rand()

                if mask[d]:
                    Xnew[i, d] = Xbest[d]
                else:
                    # Generate new tree
                    Xnew[i, d] = Xn

            # Fitness
            Fnew[i] = fun(feat, label, Xnew[i, :] > thres, opts)

        # Sort population to get best N trees
        XX = np.vstack((X, Xnew))
        FF = np.concatenate((fit, Fnew))
        sorted_indices = np.argsort(FF)
        FF = FF[sorted_indices]
        XX = XX[sorted_indices]

        # Update X and fit
        X = XX[:N, :]
        fit = FF[:N]

        # Update global best
        if fit[0] < fitG:
            fitG = fit[0]
            Xgb = X[0, :]

        curve[t - 1] = fitG
        print(f"\nIteration {t} Best (TGA) = {curve[t - 1]:.6f}")
        t += 1

    # Convert 1-based index to 0-based index for feature selection
    Pos = np.arange(1, dim + 1)
    Sf = np.where(Xgb > thres)[0]  # Convert to 0-based index

    # Select the subset of features based on selected indices
    sFeat = feat[:, Sf]

    # Store results in a dictionary
    TGA = {"sf": Sf, "ff": sFeat, "nf": len(Sf), "c": curve, "f": feat, "l": label}

    return TGA
