import os
import time

import numba
import numpy as np
from scipy.sparse import csr_matrix

from dir_definitions import ECC_MATRICES_DIR


def parse_tsv(tsv_file):
    with open(tsv_file, 'r') as infile:
        line = next(infile).strip()
        n, m = map(int, line.split())
        row_indices = []
        col_indices = []
        targets = []
        for i, line in enumerate(infile):
            line = line.strip().split()
            if line[0] == "-0.5000":
                targets.append([1])
            else:
                assert line[0] == "0.5000"
                targets.append([0])
            for j in line[1:]:
                j = int(j)
                assert i < m, f'too high i = {i}'
                assert j < n, f'too high j = {j}'
                row_indices.append(i)
                col_indices.append(j)

        H = csr_matrix((np.ones(len(row_indices), dtype=int), (row_indices, col_indices)), shape=(m, n))
        return H.T


def xor_rows_sparse(matrix, row1, row2):
    """
    XOR two rows in a sparse matrix.
    This modifies the matrix in place.

    Parameters:
        matrix (csr_matrix): The sparse matrix in CSR format.
        row1 (int): Index of the first row.
        row2 (int): Index of the second row to XOR with the first.

    Returns:
        None: The matrix is updated in place.
    """
    # Get the indices of the non-zero elements in each row
    row1_indices = set(matrix[row1].indices)
    row2_indices = set(matrix[row2].indices)

    # Perform XOR: Symmetric difference of indices
    xor_indices = row1_indices.symmetric_difference(row2_indices)

    # Update the first row with the XOR result
    matrix[row1] = csr_matrix(
        ([1] * len(xor_indices), ([0] * len(xor_indices), list(xor_indices))),
        shape=(1, matrix.shape[1])
    )


def get_standard_form_sparse(pc_matrix):
    rows, cols = pc_matrix.shape
    next_col = min(rows, cols)
    print('Min columns/rows:', next_col)

    for ii in range(min(rows, cols)):
        print(f"Processing column {ii}")
        # Find rows with a 1 in the current column
        rows_ones = ii + pc_matrix[ii:, ii].nonzero()[0]

        if len(rows_ones) == 0:
            # No pivot in this column: shift columns
            if ii < cols - 1:
                temp_col = pc_matrix[:, ii].copy()
                pc_matrix[:, ii:cols - 1] = pc_matrix[:, ii + 1:cols]
                pc_matrix[:, cols - 1] = temp_col
            next_col += 1
            continue

        # Swap rows to bring the pivot row to the top
        pivot_row = rows_ones[0]
        if pivot_row != ii:
            pc_matrix[ii], pc_matrix[pivot_row] = pc_matrix[pivot_row], pc_matrix[ii]

        # Eliminate 1s in the current column for all other rows
        col_data = pc_matrix[:, ii].indices  # Non-zero rows in column ii
        for row_idx in col_data:
            if row_idx != ii:
                xor_rows_sparse(pc_matrix, row_idx, ii)

    return pc_matrix.astype(int)


@numba.jit(nopython=True)
def xor_rows_dense(pc_matrix, ii, mask):
    rows_to_update = pc_matrix[mask]
    pc_matrix[mask] = np.bitwise_xor(rows_to_update, pc_matrix[ii])
    return pc_matrix


def get_standard_form(pc_matrix):
    rows, cols = pc_matrix.shape
    next_col = min(rows, cols)
    print('Min columns/rows:', next_col)

    # Start timing the entire function
    start_time = time.time()

    for ii in range(min(rows, cols)):
        print(f"Processing column {ii}")

        # Start timing the "while" loop
        loop_start_time = time.time()

        while True:
            # Find rows with a 1 in the current column
            rows_ones = ii + pc_matrix[ii:, ii].nonzero()[0]
            if len(rows_ones) == 0:
                # Shift columns if no row has a 1 in the current column
                if ii < cols - 1:
                    pc_matrix[:, ii:cols - 1] = pc_matrix[:, ii + 1:cols]
                next_col += 1
                continue
            break

        loop_end_time = time.time()
        print(f"Time spent on while loop for column {ii}: {loop_end_time - loop_start_time:.4f} seconds")

        # Start timing the row swap
        swap_start_time = time.time()
        # Swap rows to bring the pivot row to the top
        pivot_row = rows_ones[0]
        if pivot_row != ii:
            pc_matrix[[ii, pivot_row], :] = pc_matrix[[pivot_row, ii], :]
        swap_end_time = time.time()
        print(f"Time spent swapping rows for column {ii}: {swap_end_time - swap_start_time:.4f} seconds")
        # Eliminate 1s in the current column for all other rows
        col_data = pc_matrix[:, ii].flatten()
        mask = (col_data == 1)
        mask[ii] = False  # Skip the pivot row
        # Start timing the XOR operation
        xor_start_time = time.time()
        # Use the optimized xor_rows_broadcasting function
        pc_matrix = xor_rows_dense(pc_matrix, ii, mask)
        xor_end_time = time.time()
        print(f"Time spent on XOR operation for column {ii}: {xor_end_time - xor_start_time:.4f} seconds")

    # End timing the entire function
    end_time = time.time()
    print(f"Total time for get_standard_form: {end_time - start_time:.4f} seconds")

    return pc_matrix.astype(int)


def main(ecc_mat_path, code_type, bits_num, message_bits_num):
    pc_matrix_path = os.path.join(ecc_mat_path, f'{code_type}_{bits_num}_{message_bits_num}')
    code_pcm = parse_tsv(pc_matrix_path + '.tsv')
    code_pcm_out = get_standard_form(code_pcm.toarray()).astype(int)
    # code_pcm_out = get_standard_form_sparse(code_pcm).astype(int)
    np.save(f'{pc_matrix_path}.npy', code_pcm_out)


if __name__ == "__main__":
    main(ecc_mat_path=ECC_MATRICES_DIR, code_type='LDPC', bits_num=50000, message_bits_num=18784)
