#!/usr/bin/python3
# -*- coding: utf-8 -*-

from .sweep_support import mask2vec_bin,mask2vec_count,generate_chunk
from .utilities import fastaread
from .default_proj_mat_ope import check_default_proj_mat
import math
import h5py
import os
import numpy as np
import sys
import time
from scipy.sparse import lil_matrix, vstack
from tqdm import tqdm
from joblib import Parallel, delayed
from numpy.lib.stride_tricks import sliding_window_view
from datetime import datetime, timezone
import psutil

def _current_utc_timestamp():
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace('+00:00', 'Z')
    )

def _get_process_cpu_seconds(process):
    cpu_times = process.cpu_times()
    return float(cpu_times.user + cpu_times.system)

def _get_process_rss_mb(process):
    return float(process.memory_info().rss / (1024 ** 2))

def _normalize_mask(mask):
    """
    Project-local extension: accept either the upstream [left, gap, right]
    mask format or a binary keep/drop mask such as [1, 0, 1, 0, 1].
    """
    mask = np.array(mask)
    if len(mask) == 3 and np.issubdtype(mask.dtype, np.integer):
        return mask.astype(int), 'sweep_triplet'

    if np.all(np.isin(mask, [0, 1])):
        return mask.astype(int), 'binary_keep_mask'

    raise Exception('Mask must be an array with 3 integer values or a binary keep/drop mask.')

def _build_lookup(defSize):
    lookup = np.full(256, -1, dtype=np.int16)

    if defSize == 20:
        alphabet = 'ARNDCQEGHILKMFPSTWYV'
    elif defSize == 4:
        alphabet = 'ACGT'
    else:
        raise ValueError('Unsupported alphabet size.')

    for idx, token in enumerate(alphabet):
        lookup[ord(token)] = idx
        lookup[ord(token.lower())] = idx

    if defSize == 4:
        lookup[ord('U')] = lookup[ord('T')]
        lookup[ord('u')] = lookup[ord('t')]

    return lookup

def _binary_mask_to_vector(xseq, mask, defSize, composition):
    mask = np.array(mask, dtype=np.int8)
    keep_positions = np.flatnonzero(mask == 1)
    out_dim = defSize ** int(keep_positions.size)
    out = np.zeros((1, out_dim), dtype=int)

    if len(xseq) < len(mask):
        return out

    lookup = _build_lookup(defSize)
    encoded_seq = lookup[np.frombuffer(xseq.encode('ascii', 'ignore'), dtype=np.uint8)]
    seq_windows = sliding_window_view(encoded_seq, len(mask))
    kept_windows = seq_windows[:, keep_positions]
    valid_window_mask = np.all(kept_windows >= 0, axis=1)

    if not np.any(valid_window_mask):
        return out

    kept_windows = kept_windows[valid_window_mask].astype(np.int64, copy=False)
    positional_weights = np.power(
        defSize,
        np.arange(keep_positions.size - 1, -1, -1, dtype=np.int64),
        dtype=np.int64,
    )
    inds = kept_windows @ positional_weights

    if composition == 'binary':
        out[0, np.unique(inds)] = 1
    elif composition == 'count':
        out[0, :] = np.bincount(inds, minlength=out_dim)
    else:
        raise Exception("composition must be 'binary' or 'count'.")

    return out

def fas2sweep(xfas, orth_mat=None, mask=None, composition='binary',
              verbose=False, verbose_start='', chunk_size=1000,
              projection=True, fasta_type='AA', skip_seq_len_check=False,
              dtype=None, n_jobs=1, work_unit_callback=None):
    """
    Perform the SWeeP algorithm on sequences in FASTA format.

    Args:
        - xfas (str or list): Path to the FASTA file or a list of FASTA
          sequences.
        - orth_mat (ndarray, optional): Projection matrix. If not provided,
          the default matrix will be used.
        - mask (ndarray, optional): Mask for encoding. If not provided, the
          default mask [2, 1, 2] will be used.
        - composition (str, optional): Composition type. Either 'binary' or
          'count'. Defaults to 'binary'.
        - verbose (bool, optional): Verbosity mode. Defaults to False.
        - chunk_size (int, optional): Size of each chunk for concurrent
          processing by job. Defaults to 1000.
        - projection (bool, optional): Perform projection. Defaults to True.
        - fasta_type (str, optional): Type of FASTA sequences. Either 'AA' for
          amino acids or 'NT' for nucleotides. Defaults to 'AA'.
        - skip_seq_len_check (bool, optional): Skip the check for sequence
          length. Defaults to False.
        - dtype (dtype, optional): Data type of the output matrix. If not
          provided, float32 is used for projection and int32 for no projection.
        - n_jobs (int, optional): Number of parallel jobs. Defaults to 1.
        - work_unit_callback (callable, optional): Called once per internal
          chunk with profiling metadata about that chunk.

    Returns:
        - ndarray or sparse matrix: Resulting SWeeP matrix.
    """

    if dtype is None:
        if projection:
            dtype = np.float32
        else:
            dtype = np.int32

    if mask is None:
        mask = [2, 1, 2]
    mask, mask_kind = _normalize_mask(mask)

    if fasta_type == 'AA':
        defSize = 20
    elif fasta_type == 'NT':
        defSize = 4

    if mask_kind == 'sweep_triplet':
        mask_sum = int(mask[0] + mask[2])
        mask_window_size = int(np.sum(mask))
    else:
        mask_sum = int(np.sum(mask))
        mask_window_size = len(mask)

    # Check if orth_mat is unnecessary when projection is disabled
    if not (orth_mat is None) and not projection:
        raise Exception('The orth_mat parameter is unnecessary if ' +
                        'projection=False.')

    # Check if the size of mask parts is too high
    elif (mask_sum > 5 and fasta_type == 'AA') or (mask_sum > 10 and
                                                   fasta_type == 'NT'):
        raise Exception('The size of the mask parts is too high.')

    # Extract sequences from the FASTA file
    if isinstance(xfas, str):
        fas_cell = fastaread(xfas)
    else:
        fas_cell = xfas

    seqs = []
    for i in fas_cell:
        seqs.append(str(i.seq))

    # Calculate the number of chunks
    chunks = math.ceil(len(seqs) / chunk_size)
    len_seqs = len(seqs)

    # Checking if all sequences are bigger than the mask size
    if not skip_seq_len_check:
        for i, n in enumerate(seqs):
            if len(n) < mask_window_size:
                message = 'Sequence %i smaller than the mask size.' % i
                raise Exception(message)

    # Generate chunk indices
    idx = generate_chunk(chunks, len_seqs) - 1
    if mask_kind == 'sweep_triplet':
        proj_mat_size_req = defSize ** mask[0] * defSize ** mask[2]
    else:
        proj_mat_size_req = defSize ** int(np.sum(mask))

    if projection:
        if orth_mat is None:
            if mask_kind != 'sweep_triplet' or proj_mat_size_req != 160000:
                message = ('The default matrix is intended for the sweep of'
                           ' amino acids with the default mask, for other '
                           'cases you can disable the projection or set the '
                           'orth_mat parameter.')
                raise Exception(message)

            # Download default projection matrix if not available
            libLocal = os.path.dirname(os.path.realpath(__file__))
            mat_file_local = os.path.join(libLocal,
                                    'sweep-default-projection-matrix-600.mat')
            check_default_proj_mat(mat_file_local)
            orth_mat = h5py.File(mat_file_local, 'r')
            var_name = list(orth_mat.keys())[0]
            orth_mat = orth_mat[var_name][()].T
        else:
            if orth_mat.shape[0] != proj_mat_size_req:
                if mask_kind == 'sweep_triplet':
                    size_formula = "(x**mask[0])*(x**mask[2])"
                else:
                    size_formula = "x**sum(mask)"
                message = ("The defined orth_mat does not have the appropriate"
                           " dimensions."
                           "\n\nThe number of lines must be:"
                           "\n{},".format(size_formula) +
                           "\nwhere x=20, if fasta_type=='AA',"
                           "\nand x=4, if fasta_type=='NT'."
                           "\n\nUsing the function sweep.calc_proj_mat_size is"
                           " possible to check the necessary size and "
                           "sweep.orthbase to create the projection mat."
                           "\n\nWith the current input, the number of lines "
                           "should be {}.".format(proj_mat_size_req))
                raise Exception(message)
        orth_mat = orth_mat.astype(dtype)
        m = orth_mat.shape[1]

    if composition == 'binary':
        mask2vec = mask2vec_bin
    elif composition == 'count':
        mask2vec = mask2vec_count
    else:
        raise Exception("composition must be 'binary' or 'count'.")

    # Define mask2vec function for conversion
    if mask_kind == 'sweep_triplet':
        m2v = lambda a: mask2vec(a, mask, defSize)[0]
    else:
        m2v = lambda a: _binary_mask_to_vector(a, mask, defSize, composition)[0]

    process = psutil.Process()

    def sweep_chunk(i):
        row_start = int(idx[i, 0])
        row_end_exclusive = int(idx[i, 1]) + 1
        parcM = seqs[row_start:row_end_exclusive]
        started_at_utc = _current_utc_timestamp()
        elapsed_started_at = time.perf_counter()
        cpu_seconds_before = _get_process_cpu_seconds(process)
        rss_memory_before_mb = _get_process_rss_mb(process)
        out_mat = np.array([m2v(x) for x in parcM], dtype=dtype)
        if projection:
            n = out_mat.shape[0]
            out = np.zeros((n, m), dtype=dtype)
            out_mat = np.dot(out_mat, orth_mat, out=out)
        else:
            out_mat = lil_matrix(out_mat, dtype=dtype)
        completed_at_utc = _current_utc_timestamp()
        cpu_seconds_after = _get_process_cpu_seconds(process)
        rss_memory_after_mb = _get_process_rss_mb(process)

        if work_unit_callback is not None:
            work_unit_callback(
                {
                    'chunk_index': int(i),
                    'row_start_index': row_start,
                    'row_end_index_exclusive': row_end_exclusive,
                    'row_count': len(parcM),
                    'started_at_utc': started_at_utc,
                    'completed_at_utc': completed_at_utc,
                    'elapsed_seconds': float(time.perf_counter() - elapsed_started_at),
                    'cpu_seconds': float(cpu_seconds_after - cpu_seconds_before),
                    'rss_memory_before_mb': rss_memory_before_mb,
                    'rss_memory_after_mb': rss_memory_after_mb,
                    'rss_memory_delta_mb': float(
                        rss_memory_after_mb - rss_memory_before_mb
                    ),
                }
            )
        return out_mat

    range_s = range(0, chunks)

    # Run sweep on chunks in parallel
    result_mat = Parallel(n_jobs=n_jobs, prefer='threads')(delayed(sweep_chunk)
        (i) for i in tqdm(range_s, position=0, leave=True,
                          desc=f'{verbose_start}Running SWeeP',
                          file=sys.stdout,
                          disable=(not verbose)))

    # Concatenate the resulting matrices
    if projection:
        result_mat = np.vstack(result_mat)
    else:
        result_mat = lil_matrix(vstack(result_mat))

    return result_mat