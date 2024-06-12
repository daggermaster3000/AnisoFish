# AnisoFish
Code for quantifying zebrafish muscle fiber anisotropy, using the 2D WTMM method.
## Usage
For batch usage run the Anisotropy_analysis.py script. You will still have to specify which channel is used for the analysis (if I didn't have time to clean everything up). Currently works only on .ims files. Outputs a csv file containing the mean anisotropic factor for a fixed number of slices through the z-stack as well as the angles from each vector. For graphs check the notebooks.