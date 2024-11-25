import numpy as np

from dir_definitions import ECC_MATRICES_DIR
from python_code import conf
from python_code.utils.coding_utils import get_code_pcm_and_gm


class Encoder:
    def __init__(self, code_bits: int, message_bits: int, code_type: str):
        self._code_bits, self._message_bits, self._code_type = code_bits, message_bits, code_type
        self.code_pcm, self.code_gm = get_code_pcm_and_gm(conf.code_bits, conf.message_bits,
                                                          ECC_MATRICES_DIR, conf.code_type)
        self._encoding = lambda u: (np.dot(u, self.code_gm) % 2)
        print(f"Encoding data with {conf.code_type} ({conf.code_bits}, {conf.message_bits})")

    def encode(self, mx: np.array) -> np.array:
        tx = np.dot(mx, self.code_gm) % 2
        return tx
