from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from python_code import DEVICE, conf
from python_code.datasets.communication_blocks.encoder import Encoder
from python_code.datasets.communication_blocks.generator import Generator
from python_code.datasets.communication_blocks.modulator import MODULATION_DICT
from python_code.datasets.communication_blocks.transmitter import Transmitter


class ChannelModelDataset(Dataset):
    """
    Dataset object for the datasets. Used in training and evaluation.
    Returns (transmitted, received) batch.
    """

    def __init__(self,):
        self.blocks_num = conf.blocks_num
        self.generator = Generator()
        self.encoder = Encoder(conf.code_bits, conf.message_bits, conf.code_type)
        self.modulator = MODULATION_DICT[conf.modulation_type]
        self.transmitter = Transmitter(conf.channel_model)

    def get_snr_data(self) -> Tuple[np.array, np.array, np.array]:
        mx = self.generator.generate()
        cx = self.encoder.encode(mx)
        tx = self.modulator.modulate(cx)
        rx = self.transmitter.transmit(tx)
        return (cx, tx, rx)

    def __getitem__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mx, tx, rx = self.get_snr_data()
        mx, tx, rx = torch.Tensor(mx).to(device=DEVICE), torch.Tensor(tx).to(device=DEVICE), torch.from_numpy(rx).to(
            device=DEVICE)
        return mx, tx, rx

    def __str__(self):
        coding_name = f'{self.encoder._code_type}_{self.encoder._code_bits}_{self.encoder._message_bits}'
        channel_name = f'{self.transmitter._channel_model_name}_{self.transmitter._noise_metric}'
        name_string = channel_name + '_' + coding_name
        return name_string
