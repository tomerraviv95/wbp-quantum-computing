import torch

from python_code import DEVICE
from python_code.decoders.decoder_trainer import DecoderTrainer
from python_code.utils.config_singleton import Config
from python_code.utils.constants import Phase
from ldpc import BpDecoder

conf = Config()


class HardBPDecoder(DecoderTrainer):
    def __init__(self):
        """
        Initialize the BPDecoder with a parity-check matrix.

        Args:
            parity_check_matrix (torch.Tensor): The parity-check matrix (H) as a binary tensor.
            max_iter (int): Maximum number of iterations for belief propagation.
        """
        super().__init__()
        self.max_iter = 100
        print('------------------')
        print(conf.p)
        self.model = BpDecoder(
            self.code_pcm.astype(int),  # the parity check matrix
            error_rate=conf.p,  # the error rate on each bit
            max_iter=self.max_iter,  # the maximum iteration depth for BP
            bp_method="product_sum",  # BP method. The other option is `minimum_sum'
        )

    def __str__(self):
        return 'Binary BP Decoding'

    def forward(self, x, mode: Phase = Phase.TEST):
        """
        Perform belief propagation decoding.

        Args:
            corrupted_codeword (torch.Tensor): The received corrupted codeword (1D tensor of 0s and 1s).
            p_flip (float): The probability of bit-flipping.

        Returns:
            torch.Tensor: The decoded codeword.
        """
        decoded = []
        demod_x = (1 - x) / 2
        for corrupted_codeword in demod_x:
            codeword = self.model.decode(corrupted_codeword.cpu().numpy())
            decoded.append(torch.tensor(codeword).float().reshape(1, -1))

        return torch.cat(decoded, dim=0).to(DEVICE)
