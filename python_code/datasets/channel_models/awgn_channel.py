import numpy as np


class AWGNChannel:

    @staticmethod
    def transmit(tx: np.ndarray, snr: float, rate: float) -> np.ndarray:
        """
        The AWGN Channel
        :param tx: to transmit symbol words
        :param snr: signal-to-noise value
        :return: received word
        """
        sigma = np.sqrt(0.5 * ((10 ** ((snr + 10 * np.log10(rate)) / 10)) ** (-1)))
        w = sigma * np.random.randn(tx.shape[0], tx.shape[1])
        y = tx + w
        return 2 * y / (sigma ** 2)
