import numpy as np
import torch
from torch import nn

from dir_definitions import ECC_MATRICES_DIR
from python_code import conf
from python_code.utils.coding_utils import get_code_pcm_and_gm

ITERATIONS = 5
EPOCHS = 1000
LR = 1e-4


class DecoderTrainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)  # Single symbol probability inference
        self.odd_llr_mask_only = True
        self.even_mask_only = True
        self.multiloss_output_mask_only = True
        self.output_mask_only = False
        self.multi_loss_flag = True
        self.iteration_num = ITERATIONS
        self._code_bits = conf.code_bits
        self._message_bits = conf.message_bits
        self.code_pcm, self.code_gm = get_code_pcm_and_gm(conf.code_bits, conf.message_bits, ECC_MATRICES_DIR,
                                                          conf.code_type)
        self.neurons = int(np.sum(self.code_pcm))
        self.initialize_layers()

    def train(self):
        pass

    def initialize_layers(self):
        pass
