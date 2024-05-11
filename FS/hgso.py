import numpy as np
from FS.functionHO import Fun


def jfs(feat, label, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    num_gas = opts.get("Nc", 2)  # number of gas types/clusters
    K = opts.get("K", 1)  # constant
    alpha = opts.get("alpha", 1)  # influence of other gas
    beta = opts.get("beta", 1)  # constant
    L1 = opts.get("L1", 5e-3)
    L2 = opts.get("L2", 100)
    L3 = opts.get("L3", 1e-2)
    Ttheta = 298.15
    eps = 0.05
    c1 = opts.get("c1", 0.1)
    c2 = opts.get("c2", 0.2)
    max_iter = opts.get("T", 100)
    N = opts.get("N", 50)

    # Objective function
    fun = Fun

    # Number of dimensions
    dim = feat.shape[1]

    # Number of gas in Nc cluster
    Nn = np.ceil(N / num_gas).astype(int)

    # Initialize positions
    X = np.random.uniform(lb, ub, size=(N, dim))

    # Henry constant & E/R constant
    H = L1 * np.random.rand(num_gas)
    C = L3 * np.random.rand(num_gas)
    P = L2 * np.random.rand(num_gas, Nn)

    # Divide the population into Nc types of gas clusters
    Cx = []
    for j in range(num_gas):
        if j != num_gas - 1:
            Cx.append(X[j * Nn : (j + 1) * Nn, :])
        else:
            Cx.append(X[(num_gas - 1) * Nn :, :])

    # Initialize fitness values and best solutions
    Cfit = [np.zeros((cx.shape[0],)) for cx in Cx]
    fitCB = np.ones(num_gas)
    Cxb = np.zeros((num_gas, dim))
    fitG = float("inf")

    # Evaluate fitness and update best solutions
    for j in range(num_gas):
        for i in range(Cx[j].shape[0]):
            Cfit[j][i] = fun(feat, label, (Cx[j][i, :] > thres), opts)
            # Update best gas
            if Cfit[j][i] < fitCB[j]:
                fitCB[j] = Cfit[j][i]
                Cxb[j, :] = Cx[j][i, :]
            # Update global best
            if Cfit[j][i] < fitG:
                fitG = Cfit[j][i]
                Xgb = Cx[j][i, :]

    # Initialize iteration variables
    S = np.zeros((num_gas, Nn))
    curve = np.zeros(max_iter)
    curve[0] = fitG
    t = 1

    # Iterations
    while t < max_iter:
        # Compute temperature
        T = np.exp(-t / max_iter)

        for j in range(num_gas):
            # Update Henry coefficient
            H[j] *= np.exp(-C[j] * ((1 / T) - (1 / Ttheta)))

            for i in range(Cx[j].shape[0]):
                # Update solubility
                S[j][i] = K * H[j] * P[j][i]

                # Compute gamma
                gamma = beta * np.exp(-((fitG + eps) / (Cfit[j][i] + eps)))

                # Determine flag F
                F = -1 if np.random.rand() > 0.5 else 1

                for d in range(dim):
                    # Random constant
                    r = np.random.rand()

                    # Position update
                    Cx[j][i, d] += F * r * gamma * (
                        Cxb[j, d] - Cx[j][i, d]
                    ) + F * r * alpha * (S[j][i] * Xgb[d] - Cx[j][i, d])

                    # Apply boundary conditions
                    Cx[j][i, d] = min(max(Cx[j][i, d], lb), ub)

        # Recalculate fitness and update global best
        for j in range(num_gas):
            for i in range(Cx[j].shape[0]):
                # Calculate fitness
                Cfit[j][i] = fun(feat, label, (Cx[j][i, :] > thres), opts)

        # Select the worst solution
        Nw = round(N * (np.random.rand() * (c2 - c1) + c1))

        # Convert cell to array
        XX = np.vstack(Cx)
        FF = np.concatenate(Cfit)

        # Find indices of the worst solutions
        worst_indices = np.argsort(FF)[-Nw:]

        # Update positions of the worst solutions
        for idx in worst_indices:
            XX[idx, :] = np.random.uniform(lb, ub, size=(1, dim))
            # Recalculate fitness
            FF[idx] = fun(feat, label, (XX[idx, :] > thres), opts)

        # Divide the population back into Nc types of gas clusters
        start = 0
        for j in range(num_gas):
            end = start + Nn if j != num_gas - 1 else N
            Cx[j] = XX[start:end]
            Cfit[j] = FF[start:end]
            start = end

        # Update best solutions
        for j in range(num_gas):
            for i in range(Cx[j].shape[0]):
                # Update best gas
                if Cfit[j][i] < fitCB[j]:
                    fitCB[j] = Cfit[j][i]
                    Cxb[j, :] = Cx[j][i, :]
                # Update global best
                if Cfit[j][i] < fitG:
                    fitG = Cfit[j][i]
                    Xgb = Cx[j][i, :]

        # Update curve and iteration count
        curve[t] = fitG
        print(f"Iteration {t + 1} Best (HGSO)= {curve[t]}")
        t += 1

    # Select features based on the final best solution
    Pos = np.arange(dim)
    Sf = Pos[Xgb > thres]
    # Convert Sf to 0-based indexing by subtracting 1 from each element
    Sf = np.array(Sf) - 1
    sFeat = feat[:, Sf]

    # Store results
    HGSO = {
        "sf": Sf,
        "ff": sFeat,
        "nf": len(Sf),
        "c": curve,
        "f": feat,
        "l": label,
    }

    return HGSO
