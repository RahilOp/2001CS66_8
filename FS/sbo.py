import numpy as np
from numpy.random import rand, randn
from numpy import cumsum
from FS.functionHO import Fun


def init_position(lb, ub, N, dim):
    X = np.zeros((N, dim))
    for i in range(N):
        for d in range(dim):
            X[i, d] = lb + (ub - lb) * rand()

    return X


def j_roulette_wheel_selection(prob):
    # Cumulative summation
    C = cumsum(prob)
    # Random one value, most probable value [0~1]
    P = rand()
    # Roulette wheel
    for i, c_val in enumerate(C):
        if c_val > P:
            return i


def jfs(xtrain, ytrain, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    alpha = 0.94  # constant
    z = 0.02  # constant
    MR = 0.05  # mutation rate

    max_iter = opts["T"]
    N = opts["N"]
    if "alpha" in opts:
        alpha = opts["alpha"]
    if "z" in opts:
        z = opts["z"]
    if "MR" in opts:
        MR = opts["MR"]
    if "thres" in opts:
        thres = opts["thres"]

    # Objective function
    fun = Fun

    # Number of dimensions
    dim = xtrain.shape[1]
    # Initial
    X = init_position(lb, ub, N, dim)
    # Fitness
    fit = np.zeros(N)
    fitE = float("inf")
    for i in range(N):
        fit[i] = fun(xtrain, ytrain, (X[i, :] > thres), opts)
        # Elite update
        if fit[i] < fitE:
            fitE = fit[i]
            Xe = X[i, :]

    # Sigma (7)
    sigma = z * (ub - lb)
    # Pre
    Xnew = np.zeros((N, dim))
    Fnew = np.zeros(N)

    curve = np.zeros(max_iter)
    curve[0] = fitE
    t = 1

    # Iterations
    while t < max_iter:
        # Calculate probability (1-2)
        Ifit = 1 / (1 + fit)
        prob = Ifit / np.sum(Ifit)
        for i in range(N):
            for d in range(dim):
                # Select a bower using roulette wheel
                rw = j_roulette_wheel_selection(prob)
                # Compute lambda (4)
                lambda_ = alpha / (1 + prob[rw])
                # Update position (3)
                Xnew[i, d] = X[i, d] + lambda_ * (((X[rw, d] + Xe[d]) / 2) - X[i, d])
                # Mutation
                if rand() <= MR:
                    # Normal distribution & Position update (5-6)
                    r_normal = randn()
                    Xnew[i, d] = X[i, d] + (sigma * r_normal)
            # Boundary
            Xnew[i, :] = np.clip(Xnew[i, :], lb, ub)
        # Fitness
        for i in range(N):
            Fnew[i] = fun(xtrain, ytrain, (Xnew[i, :] > thres), opts)
        # Merge & Select best N solutions
        XX = np.vstack((X, Xnew))
        FF = np.hstack((fit, Fnew))
        idx = np.argsort(FF)
        X = XX[idx[:N], :]
        fit = FF[idx[:N]]
        # Elite update
        if fit[0] < fitE:
            fitE = fit[0]
            Xe = X[0, :]
        # Save
        curve[t] = fitE
        print(f"Iteration {t + 1} Best (SBO) = {curve[t]}")
        t += 1

    # Select features
    pos = np.arange(dim)
    Sf = pos[(Xe > thres)]
    # Store results
    sbo_data = {
        "sf": Sf,  # 0-based indexing
        "c": curve,
        "nf": len(Sf),
        "ff": xtrain[:, Sf],
        "f": xtrain,
        "l": ytrain,
    }

    return sbo_data
