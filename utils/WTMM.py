"""
Functions for computing the WTMM

Author: Quillan Favey
Date: 31-05-24
"""

import numpy as np
from scipy.ndimage import gaussian_filter 
from scipy.signal import find_peaks
# from scipy.ndimage import label, find_objects
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import random
from scipy import stats
import pandas as pd 

def WTMM_compute(data,scale,thresh):
    result = {}
    Sdata = gaussian_filter(data,scale)
    # calculate the wavelet transform
    T = np.gradient(Sdata)
    f_x, f_y = T[0],T[1]

    # calculate the square modulus
    M_2 = np.add(np.square(f_x),np.square(f_y))

    # calculate the modulus
    M = np.sqrt(M_2)

    # get the WTMM angles (the argument)
    WTMM_angles = np.angle(f_y + 1j * f_x)
    
    # calculate the secondary derivatives
    f_xy = np.gradient(f_x)[1]
    f_yy = np.gradient(f_y)[1]
    f_xx = np.gradient(f_x)[0]

    # calculate the scalar quantities N and N'
    N = (2*np.square(f_x)*f_xx)+(4*f_x*f_y*f_xy)+(2*np.square(f_y)*f_yy)
    dN = np.gradient(N*f_x)[0]+np.gradient(N*f_y)[1]
    
    # get the WTMM locations
    WTMM = np.zeros_like(dN) 
    WTMM_coords = []
    WTMMM_coords = []
    for i in tqdm(range(1,N.shape[0]-1)):
        for j in range(1,N.shape[1]-1):
            if dN[i,j] < 0:
                WTMM[i,j] = changes_sign_in_neighborhood(N, i, j, thresh)
                if M[i,j]>thresh: 
                    WTMM_coords.append([i,j])
                if np.abs(WTMM[i,j]) > 0:
                    if M[i,j]>thresh:               # keep the modulus that are above a certain threshold to remove any WTMMM detected in the noise
                        WTMMM_coords.append([i,j])

    WTMM_coords = np.array(WTMM_coords)
    WTMMM_coords = np.array(WTMMM_coords)
    
    # U2, V2 = np.gradient(WTMM)
   
    # WTMMM_angles = np.angle(V2 + 1j * U2)

    # find the WTMMM
    # Label the local maxima
    # labeled, num_objects = label(WTMM)

    # Get slices for each labeled region
    # slices = find_objects(labeled)

    # # Iterate through each labeled region
    # for label_num in tqdm(range(1, num_objects + 1)):
    #     if slices[label_num - 1] is None:
    #         continue

    #     slice_obj = slices[label_num - 1]
    #     sub_array = WTMM[slice_obj]

    #     # Find coordinates within the sub-array
    #     label_coords = np.argwhere(labeled[slice_obj] == label_num)

    #     # Detect local maxima in the sub-array using the 1D approach
    #     sub_array_flat = sub_array.flatten()
    #     peak_indexes = detect_local_maxima_1d(sub_array_flat, threshold=0.8)

    #     # Convert 1D peak indexes back to 2D coordinates within the sub-array
    #     maxima_coords_2d = np.column_stack(np.unravel_index(peak_indexes, sub_array.shape))

    #     # Adjust coordinates to the global array
    #     maxima_coords_global = maxima_coords_2d + np.array([slice_obj[0].start, slice_obj[1].start])

    #     # Store the coordinates in the list
    #     WTMMM_coords.extend(maxima_coords_global)

    # WTMMM_coords = np.array(WTMMM_coords)
    # print(len(WTMMM_coords))
    # update the angles
    if len(WTMMM_coords)!= 0:
        WTMMM_angles = WTMM_angles[WTMMM_coords[:, 0],WTMMM_coords[:, 1]]
        WTMM_angles = WTMM_angles[WTMM_coords[:, 0],WTMM_coords[:, 1]]
        
    else: 
        raise ValueError("No WTMMM found!")
        
    # WTMMM_angles = WTMMM_angles[WTMMM_coords[:, 0],WTMMM_coords[:, 1]]
    # print(WTMMM_coords.shape)
    # Compute the histogram
    counts, bin_edges = np.histogram(WTMMM_angles, bins=100, density=True)

    # Compute the bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    F_a = np.trapz(np.abs(counts-(1/(np.pi*2))),bin_centers)

    result['f_x'] = f_x
    result['f_y'] = f_y
    result['T'] = T
    result['WTMMM_coords_x'] = WTMMM_coords[:, 0]
    result['WTMMM_coords_y'] = WTMMM_coords[:, 1]
    result['WTMM_coords_x'] = WTMM_coords[:, 0]
    result['WTMM_coords_y'] = WTMM_coords[:, 1]
    result['WTMM'] = WTMM
    result['WTMM_angles'] = WTMM_angles
    result['WTMMM_angles'] = WTMMM_angles
    # result['label_coords'] = label_coords
    result['N'] = N
    result['dN'] = dN
    result['M'] = M
    result['M_2'] = M_2
    result['Sdata'] = Sdata
    result['f_xx'] = f_xx
    result['f_yy'] = f_yy
    result['data'] = data
    result['Fa'] = F_a
    result['counts'] = counts
    result['bin_centers'] = bin_centers
    result['scale'] = scale
 
    return result

def detect_local_maxima_1d(data,threshold):
    """
    Detect local maxima in a 2D array by flattening to 1D.

    Parameters:
    - data: 1D array of data.

    Returns:
    - A boolean array with True at local maxima positions.
    """
    # Get the shape of the original data
    original_shape = data.shape
    
    # Flatten the data to 1D
    flattened_data = np.abs(data)
    # calculate threshold
    threshold = np.quantile(flattened_data,threshold)
        
    # Find peaks in the flattened data
    peaks, _ = find_peaks(flattened_data,threshold=threshold)

    # # Initialize a boolean array of the same shape as the original data
    # local_maxima = np.zeros_like(flattened_data, dtype=bool)

    # # Mark the peak positions in the boolean array
    # local_maxima[peaks] = True

    # # Reshape the boolean array back to the original shape
    # local_maxima_2d = local_maxima.reshape(original_shape)

    # # Map 1D indices to 2D coordinates
    # coords = [np.unravel_index(peak, original_shape) for peak in peaks]

    # return coords
    return peaks

# Function to check if N changes sign in the 3x3 neighborhood
def changes_sign_in_neighborhood(N, i, j, thresh):
    """
    Checks if there is a change in the sign of N in the 3x3 neighborhood and returns the pixel's N value else, returns 0
    INPUTS:
         N: Scalar quantity = gradient of the square moduluses dot T
         i: coord x/lign
         j: coord y/column
    thresh: threshold
    OUTPUT:
        Pixel's N value or 0
    """
    neighborhood = N[max(0, i-1):min(N.shape[0], i+2), max(0, j-1):min(N.shape[1], j+2)]
    signs = np.sign(neighborhood)
    if np.any(signs != signs[0]):
        if N[i,j]>thresh:
            return N[i,j] 
        else:
            return 0
    else:
        return 0
    

def plot_wavelet_transform(data, step=10, scale=8000, title=r'(c) Gradient vectors $\nabla [ \phi_{\text{Gauss},\mathbf{b},a} * f]$',rect1=400,rect2=500):
    """
    Calculates the wavelet transform and plots the gradient vectors along with other specified plots.

    Parameters:
    - data: Dictionary containing the image data for plotting.
    - step: Step size for quiver plot (default is 1).
    - scale: Scale for quiver plot vectors (default is 8000).
    - title: Title for the plot (default is "r'(c) Gradient vectors $\nabla [ \phi_{\text{Gauss},\mathbf{b},a} * f]$'").
    """
    # Extract f_x, f_y
    T = data['T']
    U, V = T[0], T[1]

    # Create a new figure with subplots
    fig = plt.figure(figsize=(15, 8))
    grid = plt.GridSpec(2, 4, width_ratios=[1, 1, 1, 2])
 
    # Plot 1: Input Image
    ax1 = plt.subplot(grid[0, 0])
    ax1.imshow(data['data'])
    ax1.set_title(r'(a) Input image $f(x,y)$')
    ax1.set_xlabel("X (pix)")
    ax1.set_ylabel("Y (pix)")

    # Plot 2: Filtered Image
    ax2 = plt.subplot(grid[0, 1])
    ax2.imshow(data['Sdata'])
    ax2.set_title(r'(b) Filtered image $[\phi_{\text{Gauss},\mathbf{b},a} * f]$')
    ax2.set_xlabel("X (pix)")
    ax2.set_ylabel("Y (pix)")

    # Plot 3: Gradient Vectors
    X, Y = np.meshgrid(np.arange(0, np.shape(data['Sdata'])[1], 1), np.arange(0, np.shape(data['Sdata'])[0], 1))
    Q = plt.subplot(grid[0, 2])
    Q.quiver(X[::step, ::step], Y[::step, ::step], U[::step, ::step], V[::step, ::step], units='width', width=0.002,
             scale=scale, color="white")
    Q.set_title(title)
    Q.imshow(data['data'])
    Q.set_xlabel("X (pix)")
    Q.set_ylabel("Y (pix)")

    # Plot 4: Original image with quiver plot
    X, Y = np.meshgrid(np.arange(0, np.shape(data['Sdata'])[1], 1), np.arange(0, np.shape(data['Sdata'])[0], 1))
    X_sub = X[data['WTMMM_coords_x'],data['WTMMM_coords_y']]
    Y_sub = Y[data['WTMMM_coords_x'],data['WTMMM_coords_y']]
    U2,V2 = np.gradient(data['WTMM'])
    U_sub, V_sub =  U2[data['WTMMM_coords_x'],data['WTMMM_coords_y']], V2[data['WTMMM_coords_x'],data['WTMMM_coords_y']]
 
    ax4 = plt.subplot(grid[1, :2])
    ax4.imshow(data['data'], cmap="gray")
    magnitude = np.sqrt(U_sub**2 + V_sub**2)
  
    ax4.quiver(X_sub, Y_sub, U_sub/magnitude, V_sub/magnitude, units='width', width=0.002, scale=100, color='white')
    ax4.scatter(data['WTMM_coords_y'],data['WTMM_coords_x'], s=0.01, color='lime', marker="x")
    ax4.scatter(data['WTMMM_coords_y'], data['WTMMM_coords_x'], s=0.01, color='red')
    
    ax4.set_title('(d1) 2D WTMM Analysis')
    ax4.set_xlabel('X-axis')
    ax4.set_ylabel('Y-axis')
    ax4.axis('off')

    # Plot 5: Zoomed-in region
    zoomed_ax = plt.subplot(grid[1, 2])
    zoomed_ax.imshow(data['data'], cmap="gray")
    zoomed_ax.scatter(data['WTMM_coords_y'],data['WTMM_coords_x'], s=1, color='lime', marker=",")
    zoomed_ax.scatter(data['WTMMM_coords_y'], data['WTMMM_coords_x'], s=2, color='red', marker=",")
    zoomed_ax.quiver(X_sub, Y_sub, U_sub/magnitude, V_sub/magnitude, units='width', width=0.005, scale=15, color='white')
    zoomed_ax.set_xlim(rect1, rect2)
    zoomed_ax.set_ylim(rect1, rect2)
    zoomed_ax.set_title('(d2)')
    zoomed_ax.set_xlabel('X-axis')
    zoomed_ax.set_ylabel('Y-axis')
    zoomed_ax.axis('off')
    zoomed_ax.invert_yaxis()

    # Create a Rectangle patch
    rect = patches.Rectangle((rect1, rect1), rect2-rect1, rect2-rect1, linewidth=2, edgecolor='r', facecolor='none')
    ax4.add_patch(rect)

    plt.tight_layout()
    plt.savefig(f'Methods/images.pdf')
    plt.show()

def plot_wavelet_transform2(data, rect1=400,rect2=450,save=False):

    # Extract the coordinates of WTMMM
    x_coords = data['WTMMM_coords_x']
    y_coords = data['WTMMM_coords_y']

    # Calculate U and V components
    # U = np.cos(WTMM_angles)
    # V = np.sin(WTMM_angles)
    U2, V2 = np.gradient(data['WTMM'])

    # Create a grid for the quiver plot
    # x = np.arange(data['f_x'].shape[1])
    # y = np.arange(data['f_x'].shape[0])
    X, Y = np.meshgrid(np.arange(0, np.shape(data['Sdata'])[1], 1), np.arange(0, np.shape(data['Sdata'])[0], 1))

    # Subsample the data for plotting (plot every 10th vector)
    X_sub = X[x_coords, y_coords]
    Y_sub = Y[x_coords, y_coords]
    U_sub = U2[x_coords, y_coords]
    V_sub = V2[x_coords, y_coords]
    magnitude = np.sqrt(U_sub**2 + V_sub**2)
    magnitude[magnitude == 0] = 1
    # magnitude = data['M']
    # Create a figure and axis
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))

    # Plot the original image and quiver plot
    axs[0].imshow(data['data'],cmap="gray")
    axs[0].quiver(X_sub, Y_sub, U_sub/magnitude, V_sub/magnitude, units='width', width=0.002, scale=100, color='white')
    axs[0].scatter(data['WTMM_coords_y'], data['WTMM_coords_x'], s=0.01, color='lime', marker="x")
    axs[0].scatter(y_coords, x_coords, s=0.01, color='red')
    axs[0].set_title(f"Scale $a=${data['scale']} px")
    axs[0].set_xlabel('X-axis')
    axs[0].set_ylabel('Y-axis')
    axs[0].set_axis_off()

    # Plot the zoomed-in region
    zoomed_ax = axs[1]
    zoomed_ax.imshow(data['data'],cmap="gray")
    zoomed_ax.quiver(X_sub, Y_sub, U_sub/magnitude, V_sub/magnitude, units='width', width=0.005, scale=20, color='white')
    zoomed_ax.scatter(data['WTMM_coords_y'], data['WTMM_coords_x'], s=1, color='lime',marker=",")
    zoomed_ax.scatter(y_coords, x_coords, s=2, color='red',marker=",")
    zoomed_ax.set_xlim(rect1, rect2)
    zoomed_ax.set_ylim(rect1, rect2)
    zoomed_ax.invert_yaxis()

    # zoomed_ax.set_title('Zoomed-in Region')
    zoomed_ax.set_xlabel('X-axis')
    zoomed_ax.set_ylabel('Y-axis')
    zoomed_ax.set_axis_off()
    
    # Create a Rectangle patch
    rect = patches.Rectangle((rect1,rect1), rect2-rect1, rect2-rect1, linewidth=2, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    axs[0].add_patch(rect)
    if save:
        fig.savefig(f'{save}/scale-{data["scale"]}.pdf')
    plt.tight_layout()
    plt.show()

def violinplot_delice(df,x_group,y_variable,violin_width=0.85,y_label=None,palette="PuRd",violin_edge_color="black",point_size=10,jitter=0.05,title=None,title_loc="left",title_size=10):
    if y_label == None:
        ylabel = y_variable

    color_map = sns.color_palette(palette, n_colors=len(np.unique(df[x_group])))
    color_dic = {cond: color for cond, color in zip(np.unique(df[x_group]), color_map)}

    labels = [i for i in color_dic]

    # plot settings
    fig, axs = plt.subplots()
    colors = [i for i in color_dic.values()]

    # Test every combination
    # Check from the outside pairs of boxes inwards
    ls = list(range(1, len(labels) + 1))
    combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    significant_combinations = []
    for combination in combinations:
        data1 = df[y_variable][df[x_group] == labels[combination[0] - 1]]
        data2 = df[y_variable][df[x_group] == labels[combination[1] - 1]]
        # Significance
        U, p = stats.ttest_ind(data1, data2, alternative='two-sided')
        
        # bonferroni correction
        
        p_adj = p * len(combinations)
        print("{} x {:<30}   padj: {:<2}  p-val: {:<10}".format(
        labels[combination[0] - 1],
        labels[combination[1] - 1],
        p_adj,
        p
    ))

        if p < 0.05:
            significant_combinations.append([combination, p_adj])
        else:
            significant_combinations.append([combination, p_adj])
        #print(f"{list(groups.keys())[combination[0]-1]} - {list(groups.keys())[combination[1]-1]} | {p}")

    # individual points
    for i,cond in enumerate(color_dic):

        # workaround the truncation
        data_to_plot = df[y_variable][df[x_group]==cond]
        data_min = data_to_plot.min()
        data_max = data_to_plot.max()
        data_to_plot_adj =  pd.concat([pd.Series([data_min,data_max]),data_to_plot], ignore_index=True)


        x_values = [i + 1] * len(data_to_plot)
        x_jittered = [val + (jitter * (2 * (random.random() - 0.5))) for val in x_values]

        parts = plt.violinplot(data_to_plot_adj,[i+1],showmedians=False,
            showextrema=False, widths=violin_width)
        for pc in parts['bodies']:
            pc.set_facecolor('white')
            pc.set_edgecolor(violin_edge_color)
            pc.set_linewidths(2)
            pc.set_alpha(1)
            
        # mean
        plt.hlines(np.mean(df[y_variable][df[x_group]==cond]), i + 0.8, i + 1.2, color='black', linewidth=2, alpha=1, )
        print("{:>10} mean: {:>45}".format(cond,np.mean(df[y_variable][df[x_group]==cond])))
        # points
        plt.scatter(x_jittered, df[y_variable][df[x_group]==cond], color = colors[i], alpha=1, s=point_size, edgecolors='black',zorder=3)
        
        

    # add signif bars
    plt.xticks(range(1, len(labels) + 1), labels)
    # Add Significance bars
    # Get the y-axis limits
    bottom, top = plt.ylim()
    y_range = top - bottom
    for i, significant_combination in enumerate(significant_combinations):
        # Columns corresponding to the datasets of interest
        x1 = significant_combination[0][0]
        x2 = significant_combination[0][1]
        # What level is this bar among the bars above the plot?
        level = len(significant_combinations) - i
        # Plot the bar
        bar_height = (y_range * 0.07 * level) + top + 0.4
        bar_tips = bar_height - (y_range * 0.02)
        plt.plot(
            [x1, x1, x2, x2],
            [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k'
        )
        # Significance level
        p = significant_combination[1]
        if p < 0.001:
            sig_symbol = '***'
        elif p < 0.01:
            sig_symbol = '**'
        elif p < 0.05:
            sig_symbol = '*'
        else:
            sig_symbol = "ns"
        text_height = bar_height + (y_range * 0.01)
        plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k', weight='bold')

    # custom 
    axs.spines[['right', 'top']].set_visible(False)
    # Change the x-axis line weight
    axs.spines['bottom'].set_linewidth(2)  

    # Change the y-axis line weight
    axs.spines['left'].set_linewidth(2) 
    # Set labels and legend
    plt.xticks(range(1, len(color_dic) + 1), weight='bold')
    #plt.xlabel('treatment')
    plt.ylabel(y_label, weight='bold')
    plt.title(title, loc=title_loc, fontsize=title_size)
    plt.show()
    
    return fig,axs


def barplot_delice(df, x_group, y_variable, y_label=None,x_label=None, palette="PuRd", colors=None, bar_width=0.5, bar_edge_color="black", point_size=10, jitter=0.05, title=None, title_loc="left", title_size=10,label_rotation=45, bar_edge_width=3,errorbar_width=2):
    if y_label is None:
        y_label = y_variable
    if x_label is None:
        x_label = x_group

    color_map = sns.color_palette(palette, n_colors=len(np.unique(df[x_group])))

    if colors:
        color_dic = {cond: color for cond, color in zip(np.unique(df[x_group]), colors)}
    else:
        color_dic = {cond: color for cond, color in zip(np.unique(df[x_group]), color_map)}

    labels = [i for i in color_dic]

    # Plot settings
    fig, axs = plt.subplots()
    colors = [i for i in color_dic.values()]

    # Test every combination
    # Check from the outside pairs of boxes inwards
    ls = list(range(1, len(labels) + 1))
    combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    significant_combinations = []
    for combination in combinations:
        data1 = df[y_variable][df[x_group] == labels[combination[0] - 1]]
        data2 = df[y_variable][df[x_group] == labels[combination[1] - 1]]
        # Significance
        U, p = stats.ttest_ind(data1, data2, alternative='two-sided')
        
        # Bonferroni correction
        p_adj = p * len(combinations)
        print("{} x {:<30}   padj: {:<2}  p-val: {:<10}".format(
            labels[combination[0] - 1],
            labels[combination[1] - 1],
            p_adj,
            p
        ))

        if p_adj < 0.05:
            significant_combinations.append([combination, p_adj])
        else:
            # significant_combinations.append([combination, p_adj])
            continue

    # Individual bar plots
    for i, cond in enumerate(color_dic):
        data_to_plot = df[y_variable][df[x_group] == cond]
        mean = np.mean(data_to_plot)
        std = np.std(data_to_plot)

        # Plot bar with thicker edges
        bar = axs.bar(i + 1, mean, color=colors[i],width=bar_width, edgecolor=bar_edge_color,linewidth=bar_edge_width)
        
        # Plot error bars with thicker edges
        plt.errorbar(i + 1, mean, yerr=std, fmt='none', ecolor=bar_edge_color, elinewidth=errorbar_width, capsize=5, capthick=errorbar_width)

        # Mean
        #plt.scatter([i + 1], [mean], color='red', zorder=3)
        print("{:>10} mean: {:>45}".format(cond, mean))

        # Individual points with jitter
        # x_jittered = [i + 1 + (jitter * (2 * (random.random() - 0.5))) for _ in data_to_plot]
        # plt.scatter(x_jittered, data_to_plot, color=colors[i], alpha=1, s=point_size, edgecolors='black', zorder=3)

    # Adjust x-ticks
    axs.set_xticks(range(1, len(labels) + 1))
    axs.set_xticklabels(labels, rotation=label_rotation, ha='center')

    # Add signif bars
    bottom, top = plt.ylim()
    y_range = top - bottom
    for i, significant_combination in enumerate(significant_combinations):
        x1 = significant_combination[0][0]
        x2 = significant_combination[0][1]
        level = len(significant_combinations) - i
        bar_height = (y_range * 0.07 * level) + top + 0.4
        bar_tips = bar_height - (y_range * 0.02)
        plt.plot(
            [x1, x1, x2, x2],
            [bar_tips, bar_height, bar_height, bar_tips], lw=bar_edge_width, c='k'
        )
        p = significant_combination[1]
        if p < 0.001:
            sig_symbol = '***'
        elif p < 0.01:
            sig_symbol = '**'
        elif p < 0.05:
            sig_symbol = '*'
        else:
            # sig_symbol = "ns"
            continue
        text_height = bar_height + (y_range * 0.01)
        plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k', weight='bold', size=bar_edge_width*7)

    # Customization
    axs.spines[['right', 'top']].set_visible(False)
    axs.spines['bottom'].set_linewidth(2)  
    axs.spines['left'].set_linewidth(2) 
    plt.xlabel(x_label, weight='bold')
    plt.ylabel(y_label, weight='bold')
    plt.title(title, loc=title_loc, fontsize=title_size)
    plt.show()

    return fig, axs

def boxplot_delice(df, x_group, y_variable, y_label=None,x_label=None,fontsize=16, palette="PuRd", colors=None, bar_width=0.5,sbars=None, bar_edge_color="black", point_size=10, jitter=0.05, title=None, title_loc="left", title_size=10,label_rotation=45, bar_edge_width=3,errorbar_width=2):
    if y_label is None:
        y_label = y_variable
    if x_label is None:
        x_label = x_group

    color_map = sns.color_palette(palette, n_colors=len(np.unique(df[x_group])))

    if colors:
        color_dic = {cond: color for cond, color in zip(np.unique(df[x_group]), colors)}
    else:
        color_dic = {cond: color for cond, color in zip(np.unique(df[x_group]), color_map)}

    labels = [i for i in color_dic]

    # Plot settings
    fig, axs = plt.subplots()
    colors = [i for i in color_dic.values()]

    # Test every combination
    # Check from the outside pairs of boxes inwards
    ls = list(range(1, len(labels) + 1))
    combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    significant_combinations = []
    for combination in combinations:
        data1 = df[y_variable][df[x_group] == labels[combination[0] - 1]]
        data2 = df[y_variable][df[x_group] == labels[combination[1] - 1]]
        # Significance
        U, p = stats.ttest_ind(data1, data2, alternative='two-sided')
        
        # Bonferroni correction
        p_adj = p * len(combinations)
        print("{} x {:<30}   padj: {:<2}  p-val: {:<10}".format(
            labels[combination[0] - 1],
            labels[combination[1] - 1],
            p_adj,
            p
        ))

        if p_adj < 0.05:
            significant_combinations.append([combination, p_adj])
        else:
            # significant_combinations.append([combination, p_adj])
            continue

    # Individual bar plots
    for i, cond in enumerate(color_dic):
        data_to_plot = df[y_variable][df[x_group] == cond]
        mean = np.mean(data_to_plot)
        std = np.std(data_to_plot)

        x_values = [i + 1] * len(data_to_plot)
        x_jittered = [val + (jitter * (2 * (random.random() - 0.5))) for val in x_values]
            
        # mean
        plt.hlines(np.mean(df[y_variable][df[x_group]==cond]), i + 0.8, i + 1.2, color='black', linewidth=2, alpha=1, )
        print("{:>10} mean: {:>45}".format(cond,np.mean(df[y_variable][df[x_group]==cond])))
        # points
        plt.scatter(x_jittered, df[y_variable][df[x_group]==cond], color = colors[i], alpha=1, s=point_size, edgecolors='black',zorder=3)
        print("{:>10} mean: {:>45}".format(cond, mean))

        # Individual points with jitter
        # x_jittered = [i + 1 + (jitter * (2 * (random.random() - 0.5))) for _ in data_to_plot]
        # plt.scatter(x_jittered, data_to_plot, color=colors[i], alpha=1, s=point_size, edgecolors='black', zorder=3)

    # Adjust x-ticks
    axs.set_xticks(range(1, len(labels) + 1))
    axs.set_xticklabels(labels, rotation=label_rotation, ha='center')

    # Add signif bars
    if sbars:
        bottom, top = plt.ylim()
        y_range = top - bottom
        for i, significant_combination in enumerate(significant_combinations):
            x1 = significant_combination[0][0]
            x2 = significant_combination[0][1]
            level = len(significant_combinations) - i
            bar_height = (y_range * 0.07 * level) + top + 0.4
            bar_tips = bar_height - (y_range * 0.02)
            plt.plot(
                [x1, x1, x2, x2],
                [bar_tips, bar_height, bar_height, bar_tips], lw=bar_edge_width, c='k'
            )
            p = significant_combination[1]
            if p < 0.001:
                sig_symbol = '***'
            elif p < 0.01:
                sig_symbol = '**'
            elif p < 0.05:
                sig_symbol = '*'
            else:
                # sig_symbol = "ns"
                continue
            text_height = bar_height + (y_range * 0.01)
            plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k', weight='bold', size=bar_edge_width*7)

    # Customization
    # axs.spines[['right', 'top']].set_visible(False)
    # axs.spines['bottom'].set_linewidth(2)  
    # axs.spines['left'].set_linewidth(2) 
    plt.xlabel(x_label,fontsize=fontsize)
    plt.ylabel(y_label,fontsize=fontsize)
    plt.title(title, loc=title_loc, fontsize=title_size)
    plt.show()

    return fig, axs