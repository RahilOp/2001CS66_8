import numpy as np
import random
from FS.functionHO import Fun


def jfs(feat, label, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = opts.get("thres", 0.5)
    S = opts.get("S", 2)  # somersault factor
    N = opts.get("N")
    max_Iter = opts.get("T")

    # Objective function
    fun = Fun

    # Number of dimensions
    dim = feat.shape[1]

    # Initial population
    X = np.random.uniform(lb, ub, (N, dim))

    # Fitness
    fit = np.zeros(N)
    fitG = float("inf")
    Xbest = None

    for i in range(N):
        fit[i] = fun(feat, label, (X[i] > thres), opts)
        # Best solution
        if fit[i] < fitG:
            fitG = fit[i]
            Xbest = X[i].copy()

    # Pre
    Xnew = np.zeros((N, dim))

    curve = np.zeros(max_Iter)
    curve[0] = fitG
    t = 1  # 0-based indexing, but the loop iteration is actually from 1 to max_Iter

    # Iteration
    while t < max_Iter:
        for i in range(N):
            # Cyclone foraging
            if random.random() < 0.5:
                if t / max_Iter < random.random():
                    # Compute beta (5)
                    r1 = random.random()
                    beta = (
                        2
                        * np.exp(r1 * ((max_Iter - t + 1) / max_Iter))
                        * np.sin(2 * np.pi * r1)
                    )
                    for d in range(dim):
                        # Create random solution (6)
                        Xrand = lb + random.random() * (ub - lb)
                        # First manta ray follow best food (7)
                        if i == 0:
                            Xnew[i][d] = (
                                Xrand
                                + random.random() * (Xrand - X[i][d])
                                + beta * (Xrand - X[i][d])
                            )
                        # Followers follow the front manta ray (7)
                        else:
                            Xnew[i][d] = (
                                Xrand
                                + random.random() * (X[i - 1][d] - X[i][d])
                                + beta * (Xrand - X[i][d])
                            )
                else:
                    # Compute beta (5)
                    r1 = random.random()
                    beta = (
                        2
                        * np.exp(r1 * ((max_Iter - t + 1) / max_Iter))
                        * np.sin(2 * np.pi * r1)
                    )
                    for d in range(dim):
                        # First manta ray follow best food (4)
                        if i == 0:
                            Xnew[i][d] = (
                                Xbest[d]
                                + random.random() * (Xbest[d] - X[i][d])
                                + beta * (Xbest[d] - X[i][d])
                            )
                        # Followers follow the front manta ray (4)
                        else:
                            Xnew[i][d] = (
                                Xbest[d]
                                + random.random() * (X[i - 1][d] - X[i][d])
                                + beta * (Xbest[d] - X[i][d])
                            )

            # Chain foraging
            else:
                for d in range(dim):
                    # Compute alpha (2)
                    r = random.random()
                    alpha = 2 * r * np.sqrt(abs(np.log(r)))
                    # First manta ray follow best food (1)
                    if i == 0:
                        Xnew[i][d] = (
                            X[i][d]
                            + random.random() * (Xbest[d] - X[i][d])
                            + alpha * (Xbest[d] - X[i][d])
                        )
                    # Followers follow the front manta ray (1)
                    else:
                        Xnew[i][d] = (
                            X[i][d]
                            + random.random() * (X[i - 1][d] - X[i][d])
                            + alpha * (Xbest[d] - X[i][d])
                        )

            # Boundary
            Xnew[i] = np.clip(Xnew[i], lb, ub)

        # Fitness
        for i in range(N):
            Fnew = fun(feat, label, (Xnew[i] > thres), opts)
            # Greedy selection
            if Fnew < fit[i]:
                fit[i] = Fnew
                X[i] = Xnew[i].copy()
            # Update best
            if fit[i] < fitG:
                fitG = fit[i]
                Xbest = X[i].copy()

        # Somersault foraging
        for i in range(N):
            r2 = random.random()
            r3 = random.random()
            for d in range(dim):
                Xnew[i][d] = X[i][d] + S * (r2 * Xbest[d] - r3 * X[i][d])

            # Boundary
            Xnew[i] = np.clip(Xnew[i], lb, ub)

        # Fitness
        for i in range(N):
            Fnew = fun(feat, label, (Xnew[i] > thres), opts)
            # Greedy selection
            if Fnew < fit[i]:
                fit[i] = Fnew
                X[i] = Xnew[i].copy()
            # Update best
            if fit[i] < fitG:
                fitG = fit[i]
                Xbest = X[i].copy()

        curve[t] = fitG
        print(f"\nIteration {t + 1} Best (MRFO)= {curve[t]}")
        t += 1

    # Select features based on selected index
    pos = np.arange(dim)
    Sf = pos[Xbest > thres]  # This will return 0-based indices
    sFeat = feat[:, Sf]

    # Store results
    MRFO = {
        "sf": Sf,
        "ff": sFeat,
        "nf": len(Sf),
        "c": curve,
        "f": feat,
        "l": label,
    }

    return MRFO
