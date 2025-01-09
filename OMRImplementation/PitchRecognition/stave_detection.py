import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

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
        ret, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_not(binary)

# Kernel for horizontal line detection
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
detect_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

# Binarize the result
_, detect_horizontal = cv2.threshold(detect_horizontal, 30, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Store y-coordinates of detected lines
line_positions = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    line_positions.append(y)

# Sort line positions
line_positions = sorted(line_positions)

# Identify groups of 5 lines with consistent spacing
stave_groups = []
threshold = 3  # Allowable deviation in spacing

for i in range(len(line_positions) - 4):  # Check sequences of 5 lines
    group = line_positions[i:i + 5]
    spacings = [group[j + 1] - group[j] for j in range(4)]  # Calculate spacing between lines

    if max(spacings) - min(spacings) <= threshold:  # Check for consistent spacing
        stave_groups.append(group)
        i += 4  # Skip ahead to avoid overlapping groups

# Calculate the middle line for each stave group
middle_lines = [group[2] for group in stave_groups]  # Third line is the middle line

# Debugging: Draw the results
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for group in stave_groups:
    for y in group:
        cv2.line(output_image, (0, y), (output_image.shape[1], y), (0, 255, 0), 1)
    middle_y = group[2]
    cv2.line(output_image, (0, middle_y), (output_image.shape[1], middle_y), (0, 0, 255), 2)

cv2.imshow('Detected Staves with Middle Lines', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()