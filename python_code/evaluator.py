import random
from collections import namedtuple
from typing import List, Tuple

import numpy as np
import torch

from python_code import conf
from python_code.datasets.channel_dataset import ChannelModelDataset
from python_code.decoders import DECODERS_TYPE_DICT
from python_code.utils.metrics import calculate_error_rate

random.seed(conf.seed)
torch.manual_seed(conf.seed)
torch.cuda.manual_seed(conf.seed)
np.random.seed(conf.seed)

MetricOutput = namedtuple(
    "MetricOutput",
    "ser_list ber_list ece_list"
)


class Evaluator(object):
    """
    Implements the evaluation pipeline. Start with initializing the dataloader, detector and decoder.
    Then, drawing from the triplets of message, transmitted and received words it runs the received words through
    the pipeline of detection and decoding. At each stage, we also calculate the BER at that given step (after detection
    and after decoding).
    """

    def __init__(self):
        # initialize matrices, datasets and detector
        self._initialize_dataloader()
        self._initialize_decoder()

    def _initialize_decoder(self):
        """
        Every evaluater must have some base decoder
        """
        self.decoder = DECODERS_TYPE_DICT[conf.decoder_type]()

    def _initialize_dataloader(self):
        """
        Sets up the data loader - a generator from which we draw batches, in iterations
        """
        self.test_channel_dataset = ChannelModelDataset()

    def evaluate(self) -> float:
        """
        Evaluate BER for a given configuration
        :return: ber value
        """
        print(f"Decoding using {str(self.decoder)}")
        torch.cuda.empty_cache()
        total_errors = 0
        n_decoding = 0
        total_ber = 0
        while total_errors < 100:
            print(f"Times: {n_decoding},Errors: {total_errors}, Ber so far: {total_ber}")
            # draw words for a given snr
            cx, tx, rx = self.test_channel_dataset.__getitem__()
            # get current word and datasets
            with torch.no_grad():
                decoded_words = self.decoder.forward(rx.float())
            ber, errors = calculate_error_rate(decoded_words, cx)
            total_errors += errors
            total_ber += ber
            n_decoding += 1
        total_ber /= n_decoding
        print(f'Final bit error rate: {total_ber}')
        return total_ber

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.evaluate()
