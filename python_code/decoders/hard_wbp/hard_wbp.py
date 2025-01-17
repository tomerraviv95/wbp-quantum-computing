import math

import numba
import numpy as np
import torch

from python_code import DEVICE
from python_code.decoders.decoder_trainer import DecoderTrainer
from python_code.utils.config_singleton import Config
from python_code.utils.constants import Phase

conf = Config()


@numba.njit(parallel=False)
def bp_decode_parallel_numba(
        row_indices, column_indices, syndrome, bit_count, check_count, llr, max_iterations
):
    """
    Perform parallel Belief Propagation decoding with memory optimizations.
    """
    decoding = np.zeros(bit_count, dtype=np.uint8)
    check_to_bit = np.zeros((check_count, bit_count), dtype=np.float32)
    bit_to_check = np.zeros((check_count, bit_count), dtype=np.float32)
    candidate_syndrome = np.zeros(check_count, dtype=np.uint8)
    for it in range(1, max_iterations + 1):
        candidate_syndrome[:] = 0  # Reuse memory

        # Check-to-bit messages update
        for i in numba.prange(check_count):
            temp_forward = 1.0
            indices = row_indices[i]
            for j in indices:
                check_to_bit[i, j] = temp_forward
                temp_forward *= np.tanh(bit_to_check[i, j] / 2)

            temp_backward = 1.0
            for j in indices[::-1]:
                check_to_bit[i, j] *= temp_backward
                message_sign = -1.0 if syndrome[i] != 0 else 1.0
                x = check_to_bit[i, j]
                x = max(min(x, 0.9999999), -0.9999999)
                check_to_bit[i, j] = message_sign * np.log((1 + x) / (1 - x))
                temp_backward *= np.tanh(bit_to_check[i, j] / 2)

        # Compute log probability ratios and make hard decisions
        for j in range(bit_count):
            temp = llr[j]
            indices = column_indices[j]
            for i in indices:
                bit_to_check[i, j] = temp
                temp += check_to_bit[i, j]

            if temp <= 0:
                decoding[j] = 1
                for i in indices:
                    candidate_syndrome[i] ^= 1
            else:
                decoding[j] = 0

        # Check for convergence
        if np.array_equal(candidate_syndrome, syndrome):
            break

    return decoding


# Non-zero indices for each row
def non_zero_indices_row(csr):
    row_indices = []
    for i in range(len(csr.indptr) - 1):
        start = csr.indptr[i]
        end = csr.indptr[i + 1]
        row_indices.append(np.array(csr.indices[start:end], dtype=np.int64))
    return row_indices


# Non-zero indices for each column
def non_zero_indices_col(csc):
    col_indices = []
    for i in range(len(csc.indptr) - 1):
        start = csc.indptr[i]
        end = csc.indptr[i + 1]
        col_indices.append(np.array(csc.indices[start:end], dtype=np.int64))
    return col_indices


class BpDecoder:
    """
    Belief Propagation (BP) Decoder for LDPC (Low-Density Parity-Check) codes.

    This class implements a log-domain belief propagation decoder with support
    for different methods and schedules.
    """

    def __init__(self, pcm, channel_probabilities, max_iterations=50, alpha=0.625):
        """
        Initialize the BP Decoder.

        :param pcm: Sparse parity check matrix
        :param channel_probabilities: Probability of bit being in error for each bit
        :param max_iterations: Maximum number of decoding iterations
        :param alpha: Scaling factor for Minimum Sum method
        """
        # Validate input
        if len(channel_probabilities) != pcm.shape[1]:
            raise ValueError("Channel probabilities must match number of bits")

        self.pcm = pcm
        self.check_count, self.bit_count = pcm.shape
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.numba = True

        # Compute initial log-likelihood ratios (LLRs)
        # LLR = log((1-p)/p)
        self.llr = np.log((1 - channel_probabilities) / channel_probabilities, dtype=np.float32)
        self.log_prob_ratios = np.zeros(self.bit_count, dtype=np.uint8)

        # Allocate space for messages:
        # bit_to_check[i, j]: message from bit j to check i
        # check_to_bit[i, j]: message from check i to bit j
        # We'll store these as large arrays initialized to 0.
        # Only positions where pcm[i,j] = 1 matter.
        self.bit_to_check = np.zeros((self.check_count, self.bit_count), dtype=np.uint8)
        self.check_to_bit = np.zeros((self.check_count, self.bit_count), dtype=np.uint8)

        # Initialize messages: bit-to-check = initial LLR for all edges
        for j in range(self.bit_count):
            idxs = np.where(self.pcm[:, j] == 1)[0]
            self.bit_to_check[idxs, j] = self.llr[j]

    def decode(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Decode the input vector using Belief Propagation.

        :param input_vector: Input vector to decode
        :return: Decoded vector
        """
        # Compute syndrome
        syndrome = (np.dot(self.pcm, input_vector) % 2).astype(np.uint8)

        # Perform decoding based on schedule
        if not self.numba:
            return self._bp_decode_parallel(syndrome)
        from scipy.sparse import csr_matrix, csc_matrix
        pcm_csr = csr_matrix(self.pcm)
        row_indices = non_zero_indices_row(pcm_csr)
        del pcm_csr
        pcm_csc = csc_matrix(self.pcm)
        column_indices = non_zero_indices_row(pcm_csc)
        del pcm_csc
        return bp_decode_parallel_numba(row_indices, column_indices, syndrome, self.bit_count, self.check_count,
                                        self.llr,
                                        self.max_iterations)

    def _bp_decode_parallel(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Perform parallel Belief Propagation decoding.

        :param syndrome: Syndrome vector
        :return: Decoded vector
        """
        decoding = np.zeros(self.bit_count)
        # Main iteration loop
        for it in range(1, self.max_iterations + 1):
            candidate_syndrome = np.zeros(self.check_count, dtype=bool)
            decoding = np.zeros(self.bit_count)
            # Check to bit messages update
            for i in range(self.check_count):
                # Forward pass: compute tanh products
                temp = 1.0
                for j in np.where(self.pcm[i] != 0)[0]:
                    self.check_to_bit[i, j] = temp
                    temp *= math.tanh(self.bit_to_check[i, j] / 2)

                # Backward pass: compute messages and syndrome
                temp = 1.0
                for j in np.where(self.pcm[i] != 0)[0][::-1]:
                    self.check_to_bit[i, j] *= temp
                    message_sign = -1.0 if syndrome[i] != 0 else 1.0
                    # Compute check to bit message
                    x = self.check_to_bit[i, j]
                    x = max(min(x, 0.9999999), -0.9999999)
                    self.check_to_bit[i, j] = message_sign * math.log(
                        (1 + x) / (1 - x)
                    )
                    # Update temporary product
                    temp *= math.tanh(self.bit_to_check[i, j] / 2)

            # Compute log probability ratios and make hard decisions
            for j in range(self.bit_count):
                # Sum log messages from connected check nodes
                temp = self.llr[j]
                for i in np.where(self.pcm[:, j] != 0)[0]:
                    self.bit_to_check[i, j] = temp
                    temp += self.check_to_bit[i, j]  # In actual implementation, this would sum check-to-bit messages

                # Hard decision based on log probability ratio
                if temp <= 0:
                    decoding[j] = 1
                    # Update candidate syndrome
                    for i in np.where(self.pcm[:, j] != 0)[0]:
                        candidate_syndrome[i] ^= 1
                else:
                    decoding[j] = 0

            # Check for convergence
            if np.array_equal(candidate_syndrome, syndrome):
                self.converge = True
                break

        return decoding


class HardWBPDecoder(DecoderTrainer):
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
            pcm=self.code_pcm.astype(int),  # the parity check matrix
            channel_probabilities=np.full(self.code_pcm.shape[1], conf.p),  # the error rate on each bit
            max_iterations=self.max_iter,  # the maximum iteration depth for BP
        )

    def __str__(self):
        return 'Translated Binary BP Decoding'

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
            numpy_corrupted_codeword = corrupted_codeword.cpu().numpy()
            corruption = self.model.decode(numpy_corrupted_codeword)
            codeword = (numpy_corrupted_codeword + corruption) % 2
            decoded.append(torch.tensor(codeword).float().reshape(1, -1))

        return torch.cat(decoded, dim=0).to(DEVICE)
