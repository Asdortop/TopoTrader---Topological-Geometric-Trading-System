import numpy as np
from scipy.linalg import fractional_matrix_power

def get_laplacian_residuals(returns_window, threshold=0.5):
    """
    Channel 8: The Laplacian Residual (The 'Locomis' Signal).
    
    Args:
        returns_window: (N_assets, Window_Size) numpy array of returns.
        threshold: Correlation threshold for adjacency matrix.
        
    Returns:
        residuals: (N_assets,) vector of residuals for the current time step (last column).
    """
    # 1. Compute Correlation Matrix (N_assets x N_assets)
    # Transprose to get (N_attributes, N_observations) for corrcoef if input is (N_assets, Time)
    # The PDF says "returns_window" but implies we correlate the assets time series.
    # np.corrcoef expects rows=variables, cols=observations. So (N_assets, Time) is correct.
    corr = np.corrcoef(returns_window) 
    np.nan_to_num(corr, copy=False)
    
    # 2. Adjacency Matrix with Hard Thresholding
    # "We set Tau = 0.5. This removes weak, spurious edges"
    # "Self-Loops: We set A_ii = 0"
    adj = np.where(np.abs(corr) > threshold, 1.0, 0.0)
    np.fill_diagonal(adj, 0.0)
    
    # Check if graph is empty to avoid singular matrices or issues
    if np.sum(adj) == 0:
        return np.zeros(returns_window.shape[0])

    # 3. Normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
    degrees = np.sum(adj, axis=1)
    
    # Avoid div/0. If degree is 0, d_inv_sqrt is 0.
    with np.errstate(divide='ignore', invalid='ignore'):
         d_inv_sqrt = 1.0 / np.sqrt(degrees)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    
    # Make it a diagonal matrix
    D_inv_sqrt_mat = np.diag(d_inv_sqrt)
    
    # L = I - ...
    L = np.eye(len(corr)) - (D_inv_sqrt_mat @ adj @ D_inv_sqrt_mat)
    
    # 4. Residual Calculation: Res = L * r_t
    # "Current return vector (last row of window)" -> PDF says last row, but assumes (Time, Assets) or (Assets, Time)?
    # In this function we assumed returns_window is (Assets, Time), so current return is last column.
    
    r_t = returns_window[:, -1]
    
    # "The residual is the difference between the actual return and the graph-implied return"
    # Formula in PDF: epsilon_t = alpha * L * R_t.
    # The code snippet in PDF says: residuals = L @ r_t
    # We will stick to the PDF code snippet implementation.
    residuals = L @ r_t
    
    return residuals
