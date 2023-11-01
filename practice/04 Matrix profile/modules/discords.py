import numpy as np

from modules.utils import *


def top_k_discords(matrix_profile, top_k=3):
    """
    Find the top-k discords based on matrix profile.

    Parameters
    ---------
    matrix_profile : dict
        The matrix profile structure.

    top_k : int
        Number of discords.

    Returns
    --------
    discords : dict
        Top-k discords (indices, distances to its nearest neighbor 
        and the nearest neighbors indices).
    """
 
    discords_idx = {}


    top_k_idxs = np.argsort(matrix_profile['mp'])[-top_k:]
    
    discords_idx['indices'] = [idx for idx in top_k_idxs]

    return discords_idx