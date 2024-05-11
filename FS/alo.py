import numpy as np
from FS.functionHO import Fun


def jfs(feat, label, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5

    if "T" in opts:
        max_iter = opts["T"]
    if "N" in opts:
        N = opts["N"]
    if "thres" in opts:
        thres = opts["thres"]

    # Objective function
    fun = Fun

    # Number of dimensions
    dim = feat.shape[1]

    # Initial: Ant & antlion
    Xal = np.random.uniform(lb, ub, size=(N, dim))
    Xa = np.random.uniform(lb, ub, size=(N, dim))

    # Fitness of antlion
    fitAL = np.zeros(N)
    fitE = np.inf
    for i in range(N):
        fitAL[i] = fun(feat, label, Xal[i, :], opts)
        # Elite update
        if fitAL[i] < fitE:
            Xe = Xal[i, :]
            fitE = fitAL[i]

    # Pre
    fitA = np.ones(N)

    curve = np.zeros(max_iter)
    curve[0] = fitE
    t = 1

    # Iteration
    while t < max_iter:
        # Set weight according to iteration
        I = 1
        if t > 0.1 * max_iter:
            w = 2
            I = (10**w) * (t / max_iter)
        elif t > 0.5 * max_iter:
            w = 3
            I = (10**w) * (t / max_iter)
        elif t > 0.75 * max_iter:
            w = 4
            I = (10**w) * (t / max_iter)
        elif t > 0.9 * max_iter:
            w = 5
            I = (10**w) * (t / max_iter)
        elif t > 0.95 * max_iter:
            w = 6
            I = (10**w) * (t / max_iter)
        # Radius of ant's random walks hyper-sphere (2.10-2.11)
        c = lb / I
        d = ub / I
        # Convert probability
        Ifit = 1 / (1 + fitAL)
        prob = Ifit / Ifit.sum()
        for i in range(N):
            # Select one antlion using roulette wheel
            rs = roulette_wheel_selection(prob)
            # Apply random walk of ant around antlion
            RA = jRandomWalkALO(Xal[rs, :], c, d, max_iter, dim)
            # Apply random walk of ant around elite
            RE = jRandomWalkALO(Xe, c, d, max_iter, dim)
            # Elitism process (2.13)
            Xa[i, :] = (RA[t, :] + RE[t, :]) / 2
            # Boundary
            Xa[i, :] = np.clip(Xa[i, :], lb, ub)
        # Fitness
        for i in range(N):
            # Fitness of ant
            fitA[i] = Fun(feat, label, Xa[i, :], opts)
            # Elite update
            if fitA[i] < fitE:
                Xe = Xa[i, :]
                fitE = fitA[i]
        # Update antlion position, assume ant with best fitness is consumed
        # by antlion and the position of ant has been replaced by antlion
        # for further trap building
        XX = np.vstack((Xal, Xa))
        FF = np.hstack((fitAL, fitA))
        idx = np.argsort(FF)
        Xal = XX[idx[:N], :]
        fitAL = FF[:N]
        # Save
        curve[t] = fitE
        print(f"\nIteration {t + 1} Best (ALO)= {curve[t]}")
        t += 1

    # Select features
    Pos = np.arange(1, dim + 1)
    Sf = Pos[(Xe > thres) == 1] - 1
    sFeat = feat[:, Sf]

    # Store results
    ALO = {
        "sf": Sf,
        "ff": sFeat,
        "nf": len(Sf),
        "c": curve,
        "f": feat,
        "l": label,
    }

    return ALO


# Roulette Wheel Selection
def roulette_wheel_selection(prob):
    # Cumulative summation
    C = np.cumsum(prob)
    # Random one value, most probability value [0~1]
    P = np.random.rand()
    # Route wheel
    for i in range(len(C)):
        if C[i] > P:
            return i


# Random Walk
def jRandomWalkALO(Xal, c, d, max_Iter, dim):
    # Pre
    RW = np.zeros((max_Iter + 1, dim))
    R = np.zeros(max_Iter)
    # Random walk with C on antlion (2.8)
    c = Xal + c if np.random.rand() > 0.5 else Xal - c
    # Random walk with D on antlion (2.9)
    d = Xal + d if np.random.rand() > 0.5 else Xal - d
    for j in range(dim):
        # Random distribution (2.2)
        for t in range(max_Iter):
            R[t] = 1 if np.random.rand() > 0.5 else 0
        # Actual random walk (2.1)
        X = np.concatenate(([0], np.cumsum(2 * R - 1)))
        # [a,b]-->[c,d]
        a, b = min(X), max(X)
        # Normalized (2.7)
        Xnorm = (((X - a) * (d[j] - c[j])) / (b - a)) + c[j]
        # Store result
        RW[:, j] = Xnorm
    return RW
