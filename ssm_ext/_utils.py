# _utils.py

import numpy as np


# ---------------------------------------------------------------
# Internal: logsumexp and multivariate Gaussian log-pdf
# ---------------------------------------------------------------

def _lse(a, axis=None, keepdims=False):
    a = np.asarray(a, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    if not keepdims:
        if axis is not None:
            out = np.squeeze(out, axis=axis)
        else:
            out = float(np.squeeze(out))
    return out


def _mvn_logpdf(y, mu, Sigma):
    """
    Multivariate Gaussian log pdf, with jitter on Sigma for numerical stability.
    y, mu: (N,); Sigma: (N, N).
    """
    N = y.shape[0]
    # symmetrize + jitter
    S = 0.5 * (Sigma + Sigma.T) + 1e-10 * np.eye(N)
    L = np.linalg.cholesky(S)
    diff = y - mu
    z = np.linalg.solve(L, diff)
    return -0.5 * N * np.log(2.0 * np.pi) - np.sum(np.log(np.diag(L))) - 0.5 * np.sum(z**2)


# ---------------------------------------------------------------
# Effective log transition matrix
# ---------------------------------------------------------------

def effective_log_Ps(transitions, K):
    """Row-normalised (K, K) log transition matrix the model ACTUALLY uses,
    or None for a gate that carries no stationary matrix.

    Where the sticky weight kappa sits is a property of the transition class,
    not of the caller. A class that applies kappa on top of `log_Ps` at every
    step must fold it in here; a class whose kappa acted through a prior during
    fitting must NOT, because it is already inside the fitted `log_Ps`. Reading
    `.kappa` and adding `kappa * I` assumes the first scheme and silently
    double-counts under the second.

    So: ask the class. Any transitions object may expose

        effective_log_Ps() -> (K, K) array | None

    and this helper defers to it. Objects without that method are plain ssm
    transition classes carrying no stickiness at all -- every unrestricted fit
    is in this case -- for which the effective matrix is just the normalised
    `log_Ps`.
    """
    import numpy as _np
    from scipy.special import logsumexp as _lse_sp

    fn = getattr(transitions, "effective_log_Ps", None)
    if callable(fn):
        lp = fn()
        return None if lp is None else _np.asarray(lp, dtype=float)

    lp = getattr(transitions, "log_Ps", None)
    if lp is None:
        return None
    lp = _np.asarray(lp, dtype=float)
    return lp - _lse_sp(lp, axis=1, keepdims=True)
