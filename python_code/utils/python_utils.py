import pickle as pkl
from typing import Dict, Any

import numpy as np

def save_pkl(pkls_path: str, array: np.ndarray, type: str):
    output = open(pkls_path + '_' + type + '.pkl', 'wb')
    pkl.dump(array, output)
    output.close()


def load_pkl(pkls_path: str, type: str) -> Dict[Any, Any]:
    output = open(pkls_path + '_' + type + '.pkl', 'rb')
    return pkl.load(output)
