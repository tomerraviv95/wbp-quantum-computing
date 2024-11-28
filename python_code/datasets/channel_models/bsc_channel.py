import math

import numpy as np

from python_code.utils.config_singleton import Config

conf = Config()


class BSCChannel:

    @staticmethod
    def transmit(tx: np.ndarray, p: float, rate: float) -> np.ndarray:
        """
        :param tx: to transmit symbol words
        :param p: probability to flip
        :return: received word
        """
        flip = (np.random.rand(*tx.shape) < p).astype(int)
        # XOR operation: flip bits where flip == True
        received_bits = tx * (-1) ** flip
        return received_bits
