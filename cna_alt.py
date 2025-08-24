from itertools import permutations

import numpy as np
from scipy.stats import rankdata


def bandt_pompe_permutation_entropy(time_series, D, tau=1):
    """
    Calculate the Bandt-Pompe probability distribution and permutation entropy for a time series.
    
    Parameters:
    time_series (array-like): Input time series
    D (int): Embedding dimension (recommended 3 ≤ D ≤ 7)
    tau (int): Time delay (default=1)
    
    Returns:
    tuple: (probability distribution, permutation entropy)
    """
    N = len(time_series)
    if D < 2:
        raise ValueError("Embedding dimension D must be ≥ 2")
    if tau < 1:
        raise ValueError("Time delay tau must be ≥ 1")

    # Generate all possible permutations of order D
    possible_perms = list(permutations(range(D)))
    perm_counts = dict.fromkeys(possible_perms, 0)

    # Slide window through time series
    for i in range(N - (D - 1) * tau):
        # Extract D-dimensional vector
        vector = time_series[i:i + D*tau:tau]

        # Get the permutation pattern
        # Handle ties by keeping their original order (as per paper)
        perm = tuple(rankdata(vector, method="ordinal") - 1)  # 0-based ranking

        # Count this permutation
        if perm in perm_counts:
            perm_counts[perm] += 1

    # Total number of vectors
    total = sum(perm_counts.values())

    # Calculate probability distribution
    P = {perm: count/total for perm, count in perm_counts.items() if count > 0}

    # Calculate normalized permutation entropy
    S = -sum(p * np.log(p) for p in P.values())
    S_max = np.log(len(possible_perms))
    H_S = S / S_max

    return P, H_S

def jensen_shannon_divergence(P, P_e, Q0=None):
    """
    Calculate the Jensen-Shannon divergence between P and P_e.
    
    Parameters:
    P (dict): Probability distribution
    P_e (dict): Uniform probability distribution
    Q0 (float, optional): Normalization constant. If None, uses max possible value.
    
    Returns:
    float: Normalized Jensen-Shannon divergence Q_J
    """
    # Calculate (P + P_e)/2
    P_avg = {k: (P.get(k, 0) + P_e.get(k, 0))/2 for k in set(P) | set(P_e)}

    # Calculate terms for JS divergence
    S_avg = -sum(p * np.log(p) for p in P_avg.values() if p > 0)
    S_P = -sum(p * np.log(p) for p in P.values() if p > 0)
    S_Pe = -sum(p * np.log(p) for p in P_e.values() if p > 0)

    JS = S_avg - S_P/2 - S_Pe/2

    # Normalize (Q0 is the maximum possible JS divergence)
    if Q0 is None:
        # For uniform vs delta distribution
        Q0 = -0.5 * (1/len(P_e)) * np.log(1/len(P_e)) - 0.5 * (1 * np.log(1))
        Q0 += 0.5 * np.log(len(P_e))

    Q_J = JS / Q0

    return Q_J

def complexity_entropy_measures(time_series, D=6):
    """
    Calculate the complexity-entropy measures for a time series.
    
    Parameters:
    time_series (array-like): Input time series
    D (int): Embedding dimension (default=6 as in paper)
    
    Returns:
    tuple: (H_S (normalized entropy), C_JS (statistical complexity measure))
    """
    # Step 1: Get Bandt-Pompe probability distribution
    P, H_S = bandt_pompe_permutation_entropy(time_series, D)

    # Step 2: Create uniform distribution P_e
    possible_perms = list(permutations(range(D)))
    P_e = {perm: 1/len(possible_perms) for perm in possible_perms}

    # Step 3: Calculate Jensen-Shannon divergence Q_J
    Q_J = jensen_shannon_divergence(P, P_e)

    # Step 4: Calculate statistical complexity measure C_JS
    C_JS = Q_J * H_S

    return H_S, C_JS

# Example usage:
if __name__ == "__main__":
    # Generate some example data
    np.random.seed(42)
    random_series = np.random.rand(10000)  # Random noise
    chaotic_series = np.zeros(10000)
    chaotic_series[0] = 0.1
    for i in range(1, 10000):  # Logistic map with r=4
        chaotic_series[i] = 4 * chaotic_series[i-1] * (1 - chaotic_series[i-1])

    # Calculate measures
    print("Random noise:")
    H_S_rand, C_JS_rand = complexity_entropy_measures(random_series)
    print(f"H_S = {H_S_rand:.4f}, C_JS = {C_JS_rand:.4f}")

    print("\nChaotic series (logistic map):")
    H_S_chaos, C_JS_chaos = complexity_entropy_measures(chaotic_series)
    print(f"H_S = {H_S_chaos:.4f}, C_JS = {C_JS_chaos:.4f}")
