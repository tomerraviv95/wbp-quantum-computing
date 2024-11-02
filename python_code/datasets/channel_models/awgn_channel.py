import numpy as np


class AWGNChannel:

    @staticmethod
    def transmit(tx: np.ndarray, snr: float) -> np.ndarray:
        """
        The MIMO SED Channel
        :param tx: to transmit symbol words
        :param snr: signal-to-noise value
        :return: received word
        """
        var = 10 ** (-0.1 * snr)
        w = np.sqrt(var) * np.random.randn(tx.shape[0], tx.shape[1])
        y = tx + w
        return y
