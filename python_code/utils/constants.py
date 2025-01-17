from enum import Enum

HALF = 0.5
CLIPPING_VAL = 20


class Phase(Enum):
    TRAIN = 'train'
    TEST = 'test'


class ChannelModels(Enum):
    AWGN = 'AWGN'
    BSC = 'BSC'


class DecoderType(Enum):
    bp = 'bp'
    wbp = 'wbp'
    hd = 'hd'
    hard_bp = 'hard_bp'
    hard_wbp = 'hard_wbp'


class ModulationType(Enum):
    BPSK = 'BPSK'
