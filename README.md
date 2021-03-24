# Segmentation Code Sample
A sample of code to detect and segment "ravines" in 3d images using a modified version of the Hessian ridge detection algorithm.

# Dependencies
- [Python3](https://www.python.org/downloads/)
- [scikit-image](https://scikit-image.org/)
- [NiBabel](https://nipy.org/nibabel/)

# Usage
- cli usage:
`python3 generate_ridge_data FILENAME`
- python library usage:
```python
from generate_ridge_data import generate_ridge_data
generate_ridge_data(FILENAME)
```

# References
- [https://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=hessian_matrix#skimage.feature.hessian_matrix]
- [https://dsp.stackexchange.com/questions/1714/best-way-of-segmenting-veins-in-leaves]
- [https://en.wikipedia.org/wiki/Ridge_detection]
- [https://en.wikipedia.org/wiki/Hessian_matrix]
- [https://nipy.org/nibabel/images_and_memory.html]
- [https://www.researchgate.net/publication/312830670_Automatic_Wrinkle_Detection_Using_Hybrid_Hessian_Filter] 