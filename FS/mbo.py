import numpy as np
from scipy.special import gamma
from FS.functionHO import Fun


def jfs(feat, label, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    peri = 1.2  # migration period
    p = 5 / 12  # ratio
    Smax = 1  # maximum step
    BAR = 5 / 12  # butterfly adjusting rate
    num_land1 = 4  # number of butterflies in land 1
    beta = 1.5  # levy component

    # Read options
    max_Iter = opts.get("T", 100)  # Default max iterations
    N = opts.get("N", 100)  # Default population size
    peri = opts.get("peri", peri)
    p = opts.get("p", p)
    Smax = opts.get("Smax", Smax)
    BAR = opts.get("BAR", BAR)
    beta = opts.get("beta", beta)
    num_land1 = opts.get("N1", num_land1)
    thres = opts.get("thres", thres)

    # Objective function
    fun = Fun

    # Number of dimensions
    dim = feat.shape[1]

    # Initialize population
    X = np.random.uniform(lb, ub, size=(N, dim))

    # Initialize fitness and global best
    fit = np.zeros(N)
    fitG = np.inf
    Xgb = np.zeros(dim)

    # Calculate initial fitness and global best
    for i in range(N):
        fit[i] = fun(feat, label, X[i, :] > thres, opts)
        if fit[i] < fitG:
            fitG = fit[i]
            Xgb = X[i, :]

    # Pre-allocate memory for new population and fitness
    Xnew = np.zeros((N, dim))
    Fnew = np.zeros(N)

    # Initialize curve for tracking best fitness value
    curve = np.zeros(max_Iter)
    curve[0] = fitG
    t = 1

    # Iteration loop
    while t < max_Iter:
        # Sort butterflies by fitness
        idx = np.argsort(fit)
        fit = fit[idx]
        X = X[idx]

        # Weight factor
        alpha = Smax / (t**2)

        # First land: Migration operation
        for i in range(num_land1):
            for d in range(dim):
                # Random number
                r = np.random.rand() * peri
                if r <= p:
                    # Randomly select a butterfly in land 1
                    r1 = np.random.randint(0, num_land1)
                    # Update position
                    Xnew[i, d] = X[r1, d]
                else:
                    # Randomly select a butterfly in land 2
                    r2 = np.random.randint(num_land1, N)
                    # Update position
                    Xnew[i, d] = X[r2, d]
            # Apply boundary constraints
            Xnew[i] = np.clip(Xnew[i], lb, ub)

        # Second land: Butterfly adjusting operation
        for i in range(num_land1, N):
            # Levy distribution
            dx = jLevyDistribution(beta, dim)
            for d in range(dim):
                if np.random.rand() <= p:
                    # Position update
                    Xnew[i, d] = Xgb[d]
                else:
                    # Randomly select a butterfly in land 2
                    r3 = np.random.randint(num_land1, N)
                    # Update position
                    Xnew[i, d] = X[r3, d]
                    # Butterfly adjusting
                    if np.random.rand() > BAR:
                        Xnew[i, d] += alpha * (dx[d] - 0.5)
            # Apply boundary constraints
            Xnew[i] = np.clip(Xnew[i], lb, ub)

        # Combine population
        for i in range(N):
            # Calculate fitness
            Fnew[i] = fun(feat, label, Xnew[i, :] > thres, opts)
            # Update global best
            if Fnew[i] < fitG:
                fitG = Fnew[i]
                Xgb = Xnew[i, :]

        # Merge and select best N solutions
        XX = np.vstack((X, Xnew))
        FF = np.hstack((fit, Fnew))
        idx = np.argsort(FF)
        X = XX[idx[:N]]
        fit = FF[idx[:N]]

        # Save the best fitness value
        curve[t] = fitG
        print(f"\nIteration {t + 1} Best (MBO)= {curve[t]}")
        t += 1

    # Select features
    Pos = np.arange(dim)
    Sf = Pos[Xgb > thres].tolist()  # Convert 1-based indexing to 0-based indexing
    sFeat = feat[:, Sf]

    # Store results
    MBO = {"sf": Sf, "ff": sFeat, "nf": len(Sf), "c": curve, "f": feat, "l": label}

    return MBO


# Function for Levy Flight
def jLevyDistribution(beta, dim):
    # Calculate sigma
    nume = gamma(1 + beta) * np.sin(np.pi * beta / 2)
    deno = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (nume / deno) ** (1 / beta)

    # Generate random samples for u and v
    u = np.random.normal(0, sigma, size=(1, dim))
    v = np.random.normal(0, 1, size=(1, dim))

    # Calculate step
    step = u / (np.abs(v) ** (1 / beta))
    LF = step.flatten()  # Flatten the result to a 1D array

    return LF
