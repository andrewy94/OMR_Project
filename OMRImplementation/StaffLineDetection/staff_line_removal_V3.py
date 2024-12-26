import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
import os

"""
Morph ops and contours
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

        # Step 1: Detect potential staff lines with a horizontal kernel
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))  # Adjust width (40) for line length
        detected_lines = cv2.morphologyEx(invert_image, cv2.MORPH_OPEN, horizontal_kernel)

        #
        if len(detected_lines.shape) == 3:
            detected_lines = cv2.cvtColor(detected_lines, cv2.COLOR_BGR2GRAY)
        _, detected_lines_binary = cv2.threshold(detected_lines, 128, 255, cv2.THRESH_BINARY)
        detected_lines_binary = detected_lines_binary.astype(np.uint8)

        # Step 2: Filter out non-staff line artifacts using contours
        contours, _ = cv2.findContours(detected_lines_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(image)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h < 5 and w > 30:  # Thresholds to target long, thin lines
                cv2.drawContours(mask, [contour], -1, (255,255,255), thickness=cv2.FILLED)
            image_without_lines = cv2.subtract(invert_image, mask)

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
            os.path.splitext(os.path.basename(file_path))[0] + "_r3.png"
        )

        cv2.imwrite(save_path, final_image)
        print(f"Deskewed image saved to {save_path}")