import numpy as np

def lambda_phi(phi, lambda_0, lambda_j, phi_j, v_j):
    """
    Compute λ(φ) = λ0 + Σ_j [ λ_j / sqrt(2π v_j) * exp(-(φ - φ_j) / (2 v_j)) ]
    (NO SQUARING of (φ - φ_j))

    Parameters
    ----------
    phi : float or np.ndarray
        Points where λ(φ) is evaluated
    lambda_0 : float
        Baseline value
    lambda_j : array-like
        Array of λ_j coefficients
    phi_j : array-like
        Array of φ_j centers
    v_j : array-like
        Array of v_j parameters

    Returns
    -------
    np.ndarray
        λ(φ) evaluated at each φ
    """

    lambda_j = np.asarray(lambda_j)
    phi_j = np.asarray(phi_j)
    v_j = np.asarray(v_j)

    phi = np.asarray(phi)[..., None]

    norm = lambda_j / np.sqrt(2 * np.pi * v_j)
    exponent = np.exp(-(phi - phi_j) / (2 * v_j))

    return lambda_0 + np.sum(norm * exponent, axis=-1)
