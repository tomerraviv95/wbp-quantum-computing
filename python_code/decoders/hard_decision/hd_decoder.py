from python_code.decoders.decoder_trainer import DecoderTrainer
from python_code.utils.constants import Phase


class HDDecoder(DecoderTrainer):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'HD Decoder'

    def calc_loss(self, decision, labels):
        pass

    def forward(self, x):
        """
        compute forward pass in the network
        :param x: [batch_size,N]
        :return: decoded word [batch_size,N]
        """
        return (x < 0).int()
