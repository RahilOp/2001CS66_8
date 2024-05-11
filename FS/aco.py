import numpy as np
import random
from FS.functionHO import Fun


# Roulette wheel selection
def roulette_wheel_selection(prob):
    cumulative_prob = np.cumsum(prob)
    rand_value = random.random()
    for i, value in enumerate(cumulative_prob):
        if value > rand_value:
            return i


# Ant Colony Optimization for Text Feature Selection
def jfs(feat, label, opts):
    # Parameters
    tau = opts.get("tau", 1)  # pheromone value
    eta = opts.get("eta", 1)  # heuristic desirability
    alpha = opts.get("alpha", 1)  # control pheromone
    beta = opts.get("beta", 0.1)  # control heuristic
    rho = opts.get("rho", 0.2)  # pheromone trail decay coefficient

    N = opts["N"] if "N" in opts else 0
    max_iter = opts["T"] if "T" in opts else 0

    # Objective function
    fun = Fun

    # Number of dimensions
    dim = feat.shape[1]

    # Initial Tau & Eta matrices
    tau_matrix = tau * np.ones((dim, dim))
    eta_matrix = eta * np.ones((dim, dim))

    # Initialize variables
    fitG = float("inf")
    fit = np.zeros(N)
    curve = np.full(max_iter, float("inf"))
    t = 0

    # Main loop for iterations
    while t < max_iter:
        # Reset ants
        X = np.zeros((N, dim), dtype=int)

        for i in range(N):
            # Random number of features
            num_feat = np.random.randint(1, dim + 1)

            # Ant starts with random position
            X[i, 0] = np.random.randint(0, dim)

            if num_feat > 1:
                for d in range(1, num_feat):
                    # Start with previous tour
                    k = X[i, :d]

                    # Edge/Probability Selection
                    P = (tau_matrix[k[-1], :] ** alpha) * (eta_matrix[k[-1], :] ** beta)

                    # Set selected positions to zero probability
                    P[k] = 0

                    # Convert probability
                    prob = P / np.sum(P)

                    # Roulette Wheel selection
                    route = roulette_wheel_selection(prob)

                    # Store selected position for next tour
                    X[i, d] = route

        # Convert to binary
        X_bin = np.zeros((N, dim), dtype=int)
        for i in range(N):
            # Convert selected positions to binary
            indices = X[i, : np.count_nonzero(X[i])]
            X_bin[i, indices] = 1

        # Calculate fitness and update global best
        for i in range(N):
            # Calculate fitness
            fit[i] = fun(feat, label, X_bin[i], opts)

            # Update global best
            if fit[i] < fitG:
                Xgb = X[i]
                fitG = fit[i]

        # Update pheromone matrix
        tauK = np.zeros((dim, dim))
        for i in range(N):
            tour = X[i, : np.count_nonzero(X[i])]

            # Update delta tau k
            for d in range(len(tour) - 1):
                x, y = tour[d], tour[d + 1]
                tauK[x, y] += 1 / (1 + fit[i])

        # Update delta tau G
        tauG = np.zeros((dim, dim))
        tour = Xgb[: np.count_nonzero(Xgb)]
        for d in range(len(tour) - 1):
            x, y = tour[d], tour[d + 1]
            tauG[x, y] = 1 / (1 + fitG)

        # Evaporate pheromone and update the tau matrix
        tau_matrix = (1 - rho) * tau_matrix + tauK + tauG

        # Save best fitness for the current iteration
        curve[t] = fitG
        print(f"Iteration {t + 1} Best (ACO) = {curve[t]}")
        t += 1

    # Select features based on selected index
    Sf = Xgb[: np.count_nonzero(Xgb)]
    sFeat = feat[:, Sf]

    # Store results
    ACO = {"sf": Sf, "ff": sFeat, "nf": len(Sf), "c": curve, "f": feat, "l": label}

    return ACO
