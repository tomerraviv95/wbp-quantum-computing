import numpy as np


class MinSumDecoder:
    def __init__(self, pcm, channel_probabilities, max_iterations=50, alpha=0.625):
        """
        pcm: Parity-check matrix of shape (M, N)
        channel_probabilities: array of shape (N,), P(bit=1)
        max_iterations: number of BP iterations
        alpha: scaling factor for min-sum approximation
        """
        self.pcm = pcm
        self.M, self.N = pcm.shape
        self.max_iterations = max_iterations
        self.alpha = alpha

        # Compute initial log-likelihood ratios (LLRs)
        # LLR = log((1-p)/p)
        self.llr = np.log((1 - channel_probabilities) / channel_probabilities)

        # Allocate space for messages:
        # bit_to_check[i, j]: message from bit j to check i
        # check_to_bit[i, j]: message from check i to bit j
        # We'll store these as large arrays initialized to 0.
        # Only positions where pcm[i,j] = 1 matter.
        self.bit_to_check = np.zeros((self.M, self.N))
        self.check_to_bit = np.zeros((self.M, self.N))

        # Initialize messages: bit-to-check = initial LLR for all edges
        for j in range(self.N):
            idxs = np.where(self.pcm[:, j] == 1)[0]
            self.bit_to_check[idxs, j] = self.llr[j]

    def decode(self, received):
        """
        Decode using min-sum approximation.
        received: binary array of length N
        returns: decoded binary array of length N
        """
        # Compute syndrome from received word
        # syndrome = pcm * received (mod 2)
        syndrome = (self.pcm @ received) % 2

        for iteration in range(self.max_iterations):
            # Check node update (min-sum):
            # For each check node i:
            # 1) Gather the incoming messages from connected bits.
            # 2) Compute sign and minimum magnitudes.
            for i in range(self.M):
                # Indices of bits connected to this check node
                connected_bits = np.where(self.pcm[i, :] == 1)[0]
                if len(connected_bits) == 0:
                    continue

                # Extract messages from bit to check
                msgs = self.bit_to_check[i, connected_bits]

                # Compute the sign parity:
                # Count how many msgs are negative + add syndrome bit
                neg_count = np.sum(msgs < 0) + syndrome[i]
                sign = -1 if (neg_count % 2 == 1) else 1

                abs_msgs = np.abs(msgs)

                # Find the minimum and second minimum values
                # We need the smallest and second smallest to handle "excluded" edge scenario
                sorted_abs = np.sort(abs_msgs)
                min1 = sorted_abs[0]  # smallest
                # If there's more than one edge:
                min2 = sorted_abs[1] if len(connected_bits) > 1 else min1

                # For each edge, if this edge's abs msg is the unique min, use min2; else use min1
                for idx, bit_idx in enumerate(connected_bits):
                    if abs_msgs[idx] == min1 and np.sum(abs_msgs == min1) == 1:
                        # This edge has the unique smallest abs value
                        new_msg = sign * self.alpha * min2
                    else:
                        # Otherwise use the smallest value
                        new_msg = sign * self.alpha * min1

                    self.check_to_bit[i, bit_idx] = new_msg

            # Bit node update:
            # For each bit node j:
            # sum all incoming check_to_bit except the one corresponding to that edge
            for j in range(self.N):
                connected_checks = np.where(self.pcm[:, j] == 1)[0]
                # Total LLR for bit j (initial plus sum of check_to_bit)
                total_msg = self.llr[j] + np.sum(self.check_to_bit[connected_checks, j])
                # Update bit-to-check messages:
                # bit_to_check[i,j] = total LLR - check_to_bit[i,j]
                # We do this to provide extrinsic information
                for i in connected_checks:
                    self.bit_to_check[i, j] = total_msg - self.check_to_bit[i, j]

            # Compute hard decisions:
            total_llr = np.zeros(self.N)
            for j in range(self.N):
                connected_checks = np.where(self.pcm[:, j] == 1)[0]
                total_llr[j] = self.llr[j] + np.sum(self.check_to_bit[connected_checks, j])

            decoded = (total_llr < 0).astype(np.uint8)

            # Check if syndrome of decoded word matches
            current_syndrome = (self.pcm @ decoded) % 2
            if not current_syndrome.any():
                # Converged
                return decoded ^ received

        # If not converged within max_iterations, return best guess
        return decoded ^ received


# Example usage:
if __name__ == "__main__":
    # Define an 3x7 parity-check matrix
    # Each row has a selection of bits participating in the parity check
    H = np.array([
        [1, 1, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 1, 0],
        [1, 0, 1, 1, 0, 0, 1],
    ], dtype=np.uint8)

    # Original codeword c (all zeros is a valid codeword)
    c = np.array([1, 0, 1, 1, 0, 1, 0], dtype=np.uint8)

    # Received word r: flip bits at indices 2, 5, and 10
    r = c.copy()
    r[2] = 0
    # r[10] = 1

    # Channel probabilities: probability(bit=1)
    # Assume a channel probability, e.g., 0.2 for each bit
    channel_probs = np.full(len(c), 0.1)

    # Initialize and run the Min-Sum decoder
    decoder = MinSumDecoder(H, channel_probs, max_iterations=100, alpha=0.625)
    decoded = decoder.decode(r)

    print("Original codeword (c):", c)
    print("Received word (r)     :", r)
    print("Decoded word          :", decoded)
    # If the decoder corrects the errors, decoded should match c.
