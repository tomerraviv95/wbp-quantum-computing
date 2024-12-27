from numpy.random import default_rng

from python_code import conf


class Generator:
    def __init__(self):
        self._bits_generator = default_rng(seed=conf.seed)
        print(f"Generating {conf.blocks_num} X {conf.message_bits} to transmit bits matrix")

    def generate(self):
        mx = self._bits_generator.integers(0, 2, size=(conf.blocks_num, conf.message_bits))
        return mx
