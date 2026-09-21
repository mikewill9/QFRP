import numpy as np
import pandas as pd

def permute_within_date(zscores: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Shuffle one date's z-scores across stocks, leaving NaN entries in place."""
    out = zscores.copy()
    mask = zscores.notna()
    values = zscores[mask].to_numpy()
    if len(values)<=1:
        return out
    out[mask] = rng.permutation(values)
    return out