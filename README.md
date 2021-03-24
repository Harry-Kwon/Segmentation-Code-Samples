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