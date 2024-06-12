"""
Script to batch calculate anisotropy factor. F_a is calculated for every 10 slices, the mean is returned in the output csv file
Author: @Quillan
Date:   27.05.24
"""

# import packages
from imaris_ims_file_reader.ims import ims
import numpy as np
import matplotlib.pyplot as plt
from pyclesperanto_prototype import imshow
from scipy.ndimage import gaussian_filter, label, find_objects
from scipy.signal import find_peaks
from tqdm import tqdm
import pandas as pd
import os
import utils.WTMM as WTMM

def get_aniso(data,sig1,thresh=10):
    """
    Calculate the anisotropy factor. (refer to the notebooks for more details)
    Input:
        data: 2D array
        sig1: Scale for the Gaussian filter (px)
    Returns:
         F_a:               Anisotropic factor
         counts_smooth:     The pdf of the WTMMM angles
         bin_centers:       The bin centers of the pdf
    """
    res = WTMM.WTMM_compute(data,sig1,thresh)

    # Compute the histogram
    counts, bin_edges = np.histogram(res['WTMM_angles'], bins=100, density=True)

    # Compute the bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # window_size=5
    # counts_smooth = np.convolve(counts, np.ones(window_size)/window_size, mode='valid')

    # Adjust bin_centers to match the length of rolling_mean
    #bin_centers = bin_centers[window_size//2 : -window_size//2 +1]
    F_a = np.trapz(np.abs(counts-(1/(np.pi*2))),bin_centers)
    print(r"Anisotropic Factor (Fa): ",F_a)

    return F_a,counts,bin_centers


def process_ims_files(directory,scale):
    df = {'Name': [], 'Fa': [], 'stdv':[], 'angles':[], 'bins':[]}
    all_counts = []
    all_bin_centers = []
    
    ims_files = [file for file in os.listdir(directory) if (file.endswith(".ims") and not file.startswith("."))]

    for file in ims_files:
        path = os.path.join(directory, file)
        img = ims(path)
        counts_per_file = []
        bin_centers_per_file = []

        # Calculate F_a for every ten slices and take the mean
        for slice_index in range(10, img.shape[2] - 10, 15):
            Fa_s = []
            factor, counts, bin_centers = get_aniso(img[0, 2, slice_index, :, :],sig1=scale,thresh=20)
            Fa_s.append(factor)
            counts_per_file.append(counts)
            bin_centers_per_file.append(bin_centers)

        # Append data for each file
        all_counts.append(np.mean(counts_per_file, axis=0))
        all_bin_centers.append(bin_centers_per_file[0])  # Assuming bin_centers are same for all slices

        Fa = np.mean(Fa_s)
        df['Name'].append(file)
        df['Fa'].append(Fa)
        df['stdv'].append(np.std(Fa_s,axis=0))
        df['angles'].append(np.mean(counts_per_file, axis=0))
        df['bins'].append(bin_centers_per_file[0])

    # Plot overlay of all processed files
    plt.figure()
    for counts, filename in zip(all_counts, ims_files):
        plt.plot(all_bin_centers[0], counts, linestyle='-', marker='', label=filename)
    plt.xlim(-np.pi, np.pi)
    plt.hlines(1 / np.pi / 2, -np.pi, np.pi, color='black')
    plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
               [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'], fontsize=16)
    plt.yticks([0, 1 / np.pi / 2, 3 / 2 / np.pi, 5 / 2 / np.pi],
               [r'$0$', r'$\frac{1 }{2\pi}$', r'$\frac{3}{2\pi}$', r'$\frac{5}{2\pi}$'], fontsize=16)
    plt.title(f'Probability Density Function of WTMM Angles\n', fontsize=18, loc="center")
    plt.xlabel(r'$\text{Angle (radians)}$', fontsize=16)
    plt.ylabel(r'Probability $P_{a}(A)$', fontsize=16)
    plt.grid(True)
    # plt.legend()
    # plt.show()
    plt.savefig(os.path.join(directory,f"output-scale-{scale}.pdf"))

    # Convert dictionary to DataFrame
    df = pd.DataFrame(df)
    
    # Output CSV file in the same directory
    csv_filename = os.path.join(directory, f"output-scale-UV-{scale}.csv")
    df.to_csv(csv_filename, index=False)
    print("CSV file saved as:", csv_filename)

if __name__ == '__main__':
    directory = "/Volumes/G_MLS_RB_UHOME$/qfavey/01_Experiments/F_Spinal Muscle Staining/F01-005/2024-05-16"
    for a in [1,2,5,7]:
        print("Running analysis with scale a=",a)
        process_ims_files(directory,scale=a)

