import os
import sys
import re

import nibabel as nib

import image_utils as iu
from ridge_detection import detect_ridges_concurrent

pattern = re.compile(r"^(.*)\.nii(?:\.gz)?$")

def generate_ridge_data(filename):
    """Generates a ridge data file from a .nii.gz file

    Args:
        filename (str): Filename to evaluate

    Returns:
        ndarray: 3-dimensional array of normalized ridge scores
    """
    # extract image data from file
    background_img = nib.load(filename)
    background_data = background_img.get_fdata()

    # generate ridge scores
    ridge_data = detect_ridges_concurrent(background_data)

    # save new image
    base_filename = pattern.search(filename).group(1)
    ridge_data_filename = f"{base_filename}_ridgeScore.nii.gz"
    iu.save_image(ridge_data, ridge_data_filename, background_img.affine)

    # normalize and save normalized data
    normalized_ridge_data = iu.normalize(ridge_data)
    normalized_ridge_data_filename = os.path.join(f"{base_filename}_ridgeScore_normalized.nii.gz")
    iu.save_image(normalized_ridge_data, normalized_ridge_data_filename, background_img.affine)

    # return normalized ridgescores
    return ridge_data

from datetime import datetime
if __name__ == "__main__":
    T_FMT = "%H:%M:%S"
    startTime = datetime.now()
    filename = sys.argv[1]
    print(f"file: {filename} evalutation started: {startTime.strftime(T_FMT)}")

    if pattern.match(filename):
        generate_ridge_data(filename)
    else:
        print("invalid file")

    endTime = datetime.now()
    print(f"file: {filename} evalutation completed: {endTime.strftime(T_FMT)}")
    duration = endTime - startTime 
    print(f"file: {filename} duration: {duration}")