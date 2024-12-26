import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
import os
from scipy.signal import find_peaks

"""
Projection profiles
"""

# Withdraw the default Tkinter root window
root = tk.Tk()
root.withdraw()

# Set the initial directory
initial_directory = '/Users/andrewyan/Desktop/OMR_Project/OMRImplementation/datasets/images'
unmarked = "unmarked"

# Open a file dialog to select an image
file_path = filedialog.askopenfilename(
    title="Select an Image",
    initialdir=initial_directory,  # Set the starting directory
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
        cv2.imshow('Original', image)
        cv2.waitKey(0)

        #invert image from white background to black background
        invert_image = cv2.bitwise_not(image)
        cv2.imshow('inverted', invert_image)
        cv2.waitKey(0)

        #detect lines
        #do canny edge detection to find edges for next step, Hough Transform
        edges = cv2.Canny(image, 50, 150, apertureSize=3)

        #perform Probabilistic Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=20, maxLineGap=5)

        mask = np.zeros_like(invert_image)

        #remove lines
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Filter only horizontal lines based on y-coordinates
                if abs(y1 - y2) < 5:  # Adjust tolerance for horizontal lines
                    cv2.line(mask, (x1, y1), (x2, y2), (255,255,255), thickness=2)

        lines_removed = cv2.subtract(invert_image, mask)
        cv2.imshow('invert', lines_removed)
        cv2.waitKey(0)

        #fix symbols


        final_image = cv2.bitwise_not(lines_removed)
        cv2.imshow('invert', final_image)
        cv2.waitKey(0)

        # new_dir = os.path.join(os.path.dirname(file_path), "..", "staff_lines_removed")
        # save_path = os.path.join(
        #     new_dir,
        #     os.path.splitext(os.path.basename(file_path))[0] + "_r2.png"
        # )

        # cv2.imwrite(save_path, final_image)
        # print(f"Deskewed image saved to {save_path}")