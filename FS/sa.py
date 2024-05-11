import numpy as np
from FS.functionHO import Fun


def jInitialization(N, dim):
    # Initialize X vectors
    X = np.zeros((N, dim))
    for i in range(N):
        for d in range(dim):
            if np.random.rand() > 0.5:
                X[i, d] = 1
    return X


def jfs(feat, label, opts):
    # Parameters
    c = 0.93  # cooling rate
    T0 = 100  # initial temperature

    if "T" in opts:
        max_Iter = opts["T"]
    if "c" in opts:
        c = opts["c"]
    if "T0" in opts:
        T0 = opts["T0"]

    # Objective function
    fun = Fun

    # Number of dimensions
    dim = feat.shape[1]

    # Initial population
    X = jInitialization(1, dim)

    # Initial fitness evaluation
    fit = fun(feat, label, X[0], opts)

    # Initial best
    Xgb = X[0]
    fitG = fit

    # Pre-allocation
    curve = np.zeros(max_Iter)

    # Iteration loop
    t = 1
    while t < max_Iter:
        # Probability of swap, insert, flip, or eliminate
        prob = np.random.randint(1, 5)

        # Swap operation
        if prob == 1:
            Xnew = np.copy(X[0])
            # Find index with bit '0' and '1'
            bit0 = np.where(X[0] == 0)[0]
            bit1 = np.where(X[0] == 1)[0]
            len_0 = len(bit0)
            len_1 = len(bit1)
            # Solve issue with missing bit '0' or '1'
            if len_0 != 0 and len_1 != 0:
                # Get one random index from bit0 and bit1
                ind0 = np.random.randint(len_0)
                ind1 = np.random.randint(len_1)
                # Swap between two indices
                Xnew[bit0[ind0]] = 1
                Xnew[bit1[ind1]] = 0

        # Insert operation
        elif prob == 2:
            Xnew = np.copy(X[0])
            # Find index with zero
            bit0 = np.where(X[0] == 0)[0]
            len_0 = len(bit0)
            # Solve problem when all indices are '1'
            if len_0 != 0:
                ind = np.random.randint(len_0)
                # Add one feature
                Xnew[bit0[ind]] = 1

        # Eliminate operation
        elif prob == 3:
            Xnew = np.copy(X[0])
            # Find index with one
            bit1 = np.where(X[0] == 1)[0]
            len_1 = len(bit1)
            # Solve problem when all indices are '0'
            if len_1 != 0:
                ind = np.random.randint(len_1)
                # Remove one feature
                Xnew[bit1[ind]] = 0

        # Flip operation
        elif prob == 4:
            Xnew = np.copy(X[0])
            # Flip all variables
            Xnew = 1 - Xnew

        # Fitness
        Fnew = fun(feat, label, Xnew, opts)

        # Global best update
        if Fnew <= fitG:
            Xgb = Xnew
            fitG = Fnew
            X[0] = Xnew
        # Accept worst solution with probability
        else:
            # Delta energy
            delta = Fnew - fitG
            # Boltzmann probability
            P = np.exp(-delta / T0)
            if np.random.rand() <= P:
                X[0] = Xnew

        # Temperature update
        T0 *= c

        # Save
        curve[t] = fitG
        print(f"\nIteration {t + 1} Best (SA)= {curve[t]}")
        t += 1

    # Select features
    Pos = np.arange(dim)
    Sf = Pos[Xgb == 1].tolist()  # Convert 1-based indexing to 0-based indexing
    sFeat = feat[:, Sf]

    # Store results
    SA = {
        "sf": Sf,  # Selected features (0-based index)
        "ff": sFeat,  # Selected feature subset
        "nf": len(Sf),  # Number of selected features
        "c": curve,  # Best fitness value curve
        "f": feat,  # Features
        "l": label,  # Labels
    }

    return SA
