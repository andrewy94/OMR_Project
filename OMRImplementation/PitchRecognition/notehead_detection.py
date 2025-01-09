import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
import os

root = tk.Tk()
root.withdraw()

# Set the initial directory
initial_directory = '/Users/andrewyan/Desktop/OMR_Project/OMRImplementation/datasets/images'

# Open a file dialog to select an image
file_path = filedialog.askopenfilename(
    title="Select an Image",
    initialdir=initial_directory,
    filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif")]
)

if not file_path:
    print("No file selected!")
else:
    # Load the input image
    image = cv2.imread(file_path)
    if image is None:
        print(f"Unable to load image from {file_path}")
    else:
        if len(image.shape) == 3:  # If the image is in color
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, binarized_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        binarized_image = cv2.bitwise_not(binarized_image)

# Define the save directory
save_directory = "/Users/andrewyan/Desktop/OMR_datasets/projection"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Horizontal projection for initial segmentation
# We do horizontal projection to get a 1D array of the sums of pixels of a note image.
# We split the array in half and sum up the halves.
# The half with a larger sum is the half where the notehead is located.
# We can then also infer the orientation of the note. We must take this into account
# during distance calculations.
h_proj = np.sum(binarized_image, axis=1)
print(h_proj)
#if odd round down
mid_index = len(h_proj)/2
if mid_index % 2 == 1:
    mid_index = mid_index - 0.5
mid_index = int(mid_index)

top_half = sum(h_proj[:mid_index])
bottom_half = sum(h_proj[mid_index:])

if top_half > bottom_half:
    upside_down = True
    print("note is upside down")
else:
    upside_down = False
    print("note is rightside up")








