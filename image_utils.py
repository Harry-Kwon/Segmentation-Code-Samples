import numpy as np
import nibabel as nib
import scipy.ndimage as sp_ndimage

def save_image(data, filename, affine, logging=True):
    """Saves data to a NIFTI format file

    Args:
        data (ndarray): 3d image data to save.
        filename (str): Name of file to save to.
        affine ((4,4) array-like): Orientation of image.
        logging (bool, optional): Logging enabled. Defaults to True.
    """
    print(f"saving image to: {filename}") if logging else None
    img = nib.Nifti1Image(data, affine)
    nib.save(img, filename)

def normalize(data):
    return((data - np.min(data))/(np.max(data)-np.min(data)))

def std_threshold(data, stds=1, whitelist=np.ones(1), blacklist=np.zeros(1), logging=True):
    mean = np.mean(data)
    std = np.std(data)
    threshold_value = mean+std*stds
    print(f"mean: {mean} | std: {std} | devations: {stds} | threshold: {threshold_value}") if logging else None

    overlay = np.zeros(data.shape)
    overlay += np.where(data>threshold_value, 1, 0)

    overlay = overlay * np.where(blacklist==1, 0, 1)
    overlay = overlay * whitelist
    return(overlay)

def const_threshold(data, threshold_value, whitelist=np.ones(1), blacklist=np.zeros(1)):
    overlay = np.zeros(data.shape)
    overlay = np.where(data>threshold_value, 1, 0)

    overlay = overlay * np.where(blacklist==1, 0, 1)
    overlay = overlay * whitelist
    return(overlay)

def neighborhood(data, x, y, z, r) :
    return(data[x-r:x+r+1, y-r:y+r+1, z-r:z+r+1])

def correlation(data, kernel, logging=True) :
    correlation_data = sp_ndimage.correlate(data, kernel)
    return correlation_data
