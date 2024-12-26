import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
import os

"""
Super simple implementation that only uses morph ops. Only works on clean printed score.
"""

# Withdraw the default Tkinter root window
root = tk.Tk()
root.withdraw()

# Set the initial directory
initial_directory = '/Users/andrewyan/Desktop/OMR_Project/OMRImplementation/datasets/images'

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

        #kernel for horizontal line structures
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
        detect_horizontal = cv2.morphologyEx(invert_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        cv2.imshow('morph', detect_horizontal)
        cv2.waitKey(0)

        ret,detect_horizontal = cv2.threshold(detect_horizontal,30,255,cv2.THRESH_BINARY)
        cv2.imshow('binarized morph', detect_horizontal)
        cv2.waitKey(0)

        image_without_lines = cv2.subtract(invert_image, detect_horizontal)
        cv2.imshow('subtract', image_without_lines)
        cv2.waitKey(0)

        # vertical_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,7))
        # repair_gaps = cv2.morphologyEx(image_without_lines, cv2.MORPH_CLOSE, vertical_kernel)
        # cv2.imshow('repair', repair_gaps)
        # cv2.waitKey(0)

        final_image = cv2.bitwise_not(image_without_lines)
        cv2.imshow('invert', final_image)
        cv2.waitKey(0)

        new_dir = os.path.join(os.path.dirname(file_path), "..", "staff_lines_removed")
        save_path = os.path.join(
            new_dir,
            os.path.splitext(os.path.basename(file_path))[0] + "_r1.png"
        )

        cv2.imwrite(save_path, final_image)
        print(f"Deskewed image saved to {save_path}")