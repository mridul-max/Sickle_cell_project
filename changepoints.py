import os
import cv2
import numpy as np
import seaborn as sns
import ruptures as rpt
import matplotlib.pyplot as plt

def get_changepoints(image_folder):
    # Load the list of images in the processed_data folder
    image_list = os.listdir(image_folder)
    # Sort the list of images by timestamp
    image_list.sort()

    # Create an empty list to store the MSE values
    mse_values = []

    # Loop through pairs of consecutive images and calculate the MSE
    for i in range(len(image_list) - 1):
        # Read the two consecutive images
        img_path_1 = os.path.join(image_folder, image_list[i])
        img_path_2 = os.path.join(image_folder, image_list[i+1])
        img1 = cv2.imread(img_path_1)
        img2 = cv2.imread(img_path_2)
        # Compute the MSE between the two images
        mse = np.mean((img1 - img2)**2)

        # Add the MSE value to the list
        mse_values.append(mse)

    # Create a time axis based on the timestamps in the image filenames
    time_axis = [int(img.split(".")[0]) for img in image_list[:-1]]

    # Run changepoint detection on the MSE values using the Pelt algorithm
    model = "rbf"
    algo = rpt.Pelt(model=model).fit(np.array(mse_values))
    result = algo.predict(pen=10)

    # Generate a 2D matrix with the MSE values and changepoints
    matrix = np.zeros((len(result) + 1, len(time_axis)))
    changepoints = []
    for i in range(len(result)):
        if i == 0:
            start = 0
        else:
            start = result[i-1]
        end = result[i]
        matrix[i, start:end] = mse_values[start:end]
        changepoints.append(time_axis[end-1])
    matrix[-1, :] = mse_values

    return time_axis, mse_values, result, changepoints, matrix


def plot_changepoints(time_axis, mse_values, result, changepoints):
    # Plot the segmented time series with changepoints
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, rpt_id in enumerate(result):
        if i == 0:
            start = 0
        else:
            start = result[i-1]
        end = result[i]
        ax.plot(time_axis[start:end], mse_values[start:end], label=f"Segment {i+1}")
        ax.axvline(x=time_axis[end-1], color='k', linestyle='--')  # add vertical line at changepoint
    ax.plot(time_axis, mse_values, alpha=0.3, label="MSE")
    ax.legend()
    plt.xlabel("Time")
    plt.ylabel("MSE")
    plt.xticks(np.arange(time_axis[0], time_axis[-1], step=100))
    return fig, ax

def get_image_path(changepoint):
    image_folder = "processed_images"
    image_extension = ".jpg"

    # Assuming the image filenames are in the format "timestamp.jpg"
    filename = str(changepoint) + image_extension

    # Construct the full image path
    img_path = os.path.join(image_folder, filename)

    return img_path
