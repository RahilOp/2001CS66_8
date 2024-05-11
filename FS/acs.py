import numpy as np
from FS.functionHO import Fun


def jfs(feat, label, opts):
    # Parameters
    tau = opts.get("tau", 1)  # pheromone value
    eta = opts.get("eta", 1)  # heuristic desirability
    alpha = opts.get("alpha", 1)  # control pheromone
    beta = opts.get("beta", 1)  # control heuristic
    rho = opts.get("rho", 0.2)  # pheromone trail decay coefficient
    phi = opts.get("phi", 0.5)  # pheromone coefficient

    N = opts.get("N", 0)
    max_iter = opts.get("T", 0)

    # Objective function
    fun = Fun

    # Number of dimensions
    dim = feat.shape[1]

    # Initial Tau & Eta matrices
    tau_matrix = tau * np.ones((dim, dim))
    eta_matrix = eta * np.ones((dim, dim))

    # Initialize variables
    fit_global = float("inf")
    fit = np.zeros(N)
    tau0 = tau_matrix.copy()

    curve = []
    t = 0

    # Main loop for iterations
    while t < max_iter:
        # Reset ant
        X = np.zeros((N, dim))
        for i in range(N):
            # Set number of features
            num_feat = np.random.randint(1, dim + 1)
            # Ant starts with random position
            X[i, 0] = np.random.randint(0, dim)
            k = []
            if num_feat > 1:
                for d in range(1, num_feat):
                    # Start with previous tour
                    k.append(int(X[i, d - 1]))
                    # Edge / Probability Selection
                    P = (tau_matrix[k[-1], :] ** alpha) * (eta_matrix[k[-1], :] ** beta)
                    # Set selected positions to zero probability
                    P[k] = 0
                    # Convert probability
                    prob = P / P.sum()
                    # Roulette wheel selection
                    route = roulette_wheel_selection(prob)
                    # Store selected position for next tour
                    X[i, d] = route

        # Convert to binary
        X_bin = np.zeros((N, dim))
        for i in range(N):
            # Convert selected positions to binary
            indices = np.array(X[i, :], dtype=int)
            indices = indices[indices != 0]
            X_bin[i, indices] = 1

        # Calculate fitness and update global best
        for i in range(N):
            # Calculate fitness
            fit[i] = fun(feat, label, X_bin[i, :], opts)

            # Update global best
            if fit[i] < fit_global:
                Xgb = X[i]
                fit_global = fit[i]

        # Update pheromone matrix
        tour = np.array(Xgb, dtype=int)
        tour = tour[tour != 0]
        tour = np.append(tour, tour[0])
        for d in range(len(tour) - 1):
            # Feature selected
            x, y = tour[d], tour[d + 1]
            # Delta tau
            Dtau = 1 / fit_global
            # Update tau
            tau_matrix[x, y] = (1 - phi) * tau_matrix[x, y] + phi * Dtau

        # Evaporate pheromone
        tau_matrix = (1 - rho) * tau_matrix + rho * tau0

        # Save best fitness for the current iteration
        curve.append(fit_global)
        print(f"Iteration {t + 1} Best (ACS)= {curve[-1]}")
        t += 1

    # Select features based on selected index
    Sf = np.unique(Xgb)
    Sf = Sf[Sf != 0]  # Remove zeros
    Sf = Sf.astype(int)
    sFeat = feat[:, Sf]

    # Store results
    ACS = {
        "sf": Sf,
        "ff": sFeat,
        "nf": len(Sf),
        "c": curve,
        "f": feat,
        "l": label,
    }

    return ACS


# Roulette Wheel Selection
def roulette_wheel_selection(prob):
    # Cumulative summation
    cumulative_sum = np.cumsum(prob)
    # Random value between 0 and 1
    random_val = np.random.rand()
    # Roulette wheel
    for i, value in enumerate(cumulative_sum):
        if value > random_val:
            return i  # 0-based index
