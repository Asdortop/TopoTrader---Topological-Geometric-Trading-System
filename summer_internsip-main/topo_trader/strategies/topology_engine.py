import numpy as np
from ripser import ripser

def persistence_entropy(diagram):
    """
    Calculate Persistence Entropy manually as per PDF Section 4.3.
    E = - sum(p_i * log(p_i)) where p_i = l_i / L
    """
    if len(diagram) == 0:
        return 0.0
        
    # Lifespans l_i = d_i - b_i
    # diagram is (N, 2) array of [birth, death]
    births = diagram[:, 0]
    deaths = diagram[:, 1]
    
    # Filter out potential infinity (inf) in death if H0 has one component that never dies
    # Usually H0 has one point with death=inf. We should remove it or replace it?
    # PDF doesn't specify handling inf, but standard PE ignores infinite bars or replaces with max.
    # In 'ripser', H0 usually has one inf.
    # We will ignore infinite bars for entropy calculation to return a finite number.
    finite_indices = np.isfinite(deaths)
    
    if np.sum(finite_indices) == 0:
        return 0.0
        
    lifespans = deaths[finite_indices] - births[finite_indices]
    
    # Avoid zero lifespans
    lifespans = lifespans[lifespans > 0]
    if len(lifespans) == 0:
        return 0.0
    
    total_lifespan = np.sum(lifespans)
    
    if total_lifespan == 0:
        return 0.0
        
    probs = lifespans / total_lifespan
    entropy = -np.sum(probs * np.log(probs))
    
    return entropy

def get_tda_features(returns_window):
    """
    Channel 10: H0 Persistence Entropy (TDA)
    Channel 11: H1 Persistence Sum (TDA)
    """
    # Distance matrix: sqrt(2 * (1 - rho))
    # We correlate the assets.
    corr = np.corrcoef(returns_window)
    np.nan_to_num(corr, copy=False)
    
    # Clip correlation to [-1, 1] to avoid numerical errors in sqrt
    corr = np.clip(corr, -1.0, 1.0)
    
    dist = np.sqrt(2 * (1 - corr))
    np.nan_to_num(dist, copy=False)
    
    # Compute Persistence Diagram
    # maxdim=1 computes H0 and H1
    try:
        diagrams = ripser(dist, maxdim=1, distance_matrix=True)['dgms']
    except Exception:
        return np.array([0.0, 0.0])

    # Calculate Entropy for H0 and H1
    if len(diagrams) > 0:
        h0_entropy = persistence_entropy(diagrams[0])
    else:
        h0_entropy = 0.0
        
    if len(diagrams) > 1:
        h1_entropy = persistence_entropy(diagrams[1])
    else:
        h1_entropy = 0.0
        
    return np.array([h0_entropy, h1_entropy])
