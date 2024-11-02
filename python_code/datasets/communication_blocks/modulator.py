import numpy as np


class BPSKModulator:
    def __init__(self):
        print("BPSK Modulation")

    @staticmethod
    def modulate(c: np.ndarray) -> np.ndarray:
        """
        BPSK modulation 0->1, 1->-1
        :param c: the binary codeword
        :return: binary modulated signal
        """
        x = 1 - 2 * c
        return x


MODULATION_DICT = {
    'BPSK': BPSKModulator,
}
