from typing import Tuple

import numpy as np

from python_code import conf
from python_code.datasets.channel_models.awgn_channel import AWGNChannel
from python_code.datasets.channel_models.bsc_channel import BSCChannel
from python_code.utils.constants import ChannelModels

CHANNELS_DICT = {ChannelModels.AWGN.name: AWGNChannel,
                 ChannelModels.BSC.name: BSCChannel}


class Transmitter:

    def __init__(self, channel_model: str, rate: float):
        print(f"Transmitting over {channel_model} channel")
        self._channel_model = CHANNELS_DICT[channel_model]
        if ChannelModels.AWGN.name == channel_model:
            self._noise_metric = conf.snr
        elif ChannelModels.BSC.name == channel_model:
            self._noise_metric = conf.p
        else:
            raise ValueError("No such channel!")
        self._channel_model_name = channel_model
        self._rate = rate

    def transmit(self, tx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rx = self._channel_model.transmit(tx, self._noise_metric,self._rate)
        return rx
