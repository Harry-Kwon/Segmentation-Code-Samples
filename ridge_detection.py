import os
import sys
import math
from multiprocessing import Pool
from datetime import datetime

import numpy as np
import nibabel as nib

import image_utils as iu
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

T_FMT = "%H:%M:%S"
def p_main(start_x, end_x, H_elems, logging=True):
    """Process execution for parallel processing of modified hessian values.
    Works on a slice of the image determined by start_x, end_x

    Args:
        start_x (int): Start index of the slice (inclusive).
        end_x (int): End index of the slice (exclusive).
        H_elems (ndarray): Computed hessian matrix for the image.
        logging (bool, optional): Logs execution and termination of process if set to True. Defaults to True.

    Returns:
        [type]: [description]
    """

    if logging:
        pStartTime = datetime.now()
        print(f"\nstarting process | pid: {os.getpid()} | time: {pStartTime.strftime(T_FMT)}", end="")

    # slice the image, other slices are processed in parallel by sibling processes
    H_slice = [h[start_x:end_x] for h in H_elems]
    d = np.zeros_like(H_elems[0])
    eigs = hessian_matrix_eigvals(H_slice)

    # modified hessian ridge detection for 3d sulcus-like "ravines"
    d[start_x:end_x, :, :] = eigs[0] - np.abs(eigs[1]) - np.abs(eigs[2])

    if logging:
        duration = datetime.now() - pStartTime 
        print(f"\nprocess completed | pid: {os.getpid()} | duration: {duration}", end="")
    return d

def detect_ridges_concurrent(data, logging=True):
    """Detects sulcus-like "ravines" from 3d image using modified hessian ridge detection

    Args:
        data (ndarray): 3-dimensional image data array.
        logging (bool, optional): Process execution logging. Defaults to True.
    """
    (xMax, _, _) = data.shape
    P_COUNT = 4
    H_elems = hessian_matrix(data, sigma=1)

    with Pool(P_COUNT) as pool:
        width = math.ceil(xMax/P_COUNT)
        multi_res = [pool.apply_async(p_main, (width*i, min(width*(i+1), xMax), H_elems, logging)) for i in range(P_COUNT)]
        results = [res.get(timeout=300) for res in multi_res]

        output_data = np.zeros(data.shape)
        for i in range(P_COUNT):
            output_data += results[i]

        if logging:
            print("\nall processes completed")
    
        return(output_data)