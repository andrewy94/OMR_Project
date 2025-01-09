# Version 0.1
# Sheet Music APP OMR pipeline 
# Andrew Yan
# Jan 2025
import tkinter as tk
from tkinter import filedialog
import cv2
import os
import csv
import shutil
import numpy as np
from keras.models import load_model


# Prologue

# Select Image:
root = tk.Tk()
root.withdraw()

initial_directory = '/Users/andrewyan/Desktop/OMR_Project/OMRImplementation/datasets/images'

file_path = filedialog.askopenfilename(
    title="Select an Image",
    initialdir=initial_directory,
    filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
)

if not file_path:
    print("No file selected!")
else:

# IMAGE PREPROCESSING:
# Step 1.1 (Grayscaling):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Unable to load image from {file_path}")

# Step 1.2 (Binarization):
#blur   
blur = cv2.GaussianBlur(image, (0,0), sigmaX=33, sigmaY=33)
cv2.imshow('blur', blur)

#divide
divide = cv2.divide(image, blur, scale=255)
cv2.imshow('divide', divide)

#binarize
ret,binarized_image = cv2.threshold(divide,127,255,cv2.THRESH_BINARY)

# SYMBOL SEGMENTATION
# Function to find segments
def find_segments(proj):
    segments = []
    in_segment = False
    start = 0
    for i, value in enumerate(proj):
        if value > 0 and not in_segment:
            in_segment = True
            start = i  # Start of the segment
        elif value == 0 and in_segment:
            in_segment = False
            end = i  # End of the segment
            segments.append((start, end))
    if in_segment:
        segments.append((start, len(proj)))  # Handle case where segment ends at the last element
    return segments

# Step 2.1 (Bruteforce Symbol Segmentation)
# Define the save directory
save_directory = "/Users/andrewyan/Desktop/OMR_datasets/segmented_symbols"

# Erase the save directory if it exists and recreate it
if os.path.exists(save_directory):
    shutil.rmtree(save_directory)  # Remove the existing directory
os.makedirs(save_directory)  # Create a new empty directory

# Horizontal projection for initial segmentation
h_proj = np.sum(binarized_image, axis=1)
h_segments = find_segments(h_proj)

bounding_boxes = []
symbol_count = 0

for start_row, end_row in h_segments:
    # Extract the row segment
    row_image = binarized_image[start_row:end_row, :]
    v_proj = np.sum(row_image, axis=0)
    v_segments = find_segments(v_proj)

    for start_col, end_col in v_segments:
        # Extract the vertical segment
        symbol_region = binarized_image[start_row:end_row, start_col:end_col]

        # Sub-segmentation: Apply horizontal projection again within the vertical segment
        sub_h_proj = np.sum(symbol_region, axis=1)
        sub_h_segments = find_segments(sub_h_proj)

        for sub_start_row, sub_end_row in sub_h_segments:
            # Calculate the bounds of the second round
            second_top = start_row + sub_start_row
            second_bottom = start_row + sub_end_row
            second_left = start_col
            second_right = end_col

            # Extract the sub-segment region
            second_region = binarized_image[second_top:second_bottom, second_left:second_right]

            # Third round of segmentation: Apply vertical projection within the second region
            third_v_proj = np.sum(second_region, axis=0)
            third_v_segments = find_segments(third_v_proj)

            for third_start_col, third_end_col in third_v_segments:
                # Calculate final bounding box for the third round
                final_top = second_top
                final_bottom = second_bottom
                final_left = second_left + third_start_col
                final_right = second_left + third_end_col

                # Extract the third region
                third_region = binarized_image[final_top:final_bottom, final_left:final_right]

                # Fourth round of segmentation: Apply horizontal projection within the third region
                fourth_h_proj = np.sum(third_region, axis=1)
                fourth_h_segments = find_segments(fourth_h_proj)

                for fourth_start_row, fourth_end_row in fourth_h_segments:
                    # Calculate final bounding box for the fourth round
                    fourth_top = final_top + fourth_start_row
                    fourth_bottom = final_top + fourth_end_row
                    fourth_left = final_left
                    fourth_right = final_right

                    # Check if the segment size is greater than 1x1
                    height = fourth_bottom - fourth_top
                    width = fourth_right - fourth_left

                    if height > 1 and width > 1:

                        # Append the bounding box with symbol number
                        bbox = (symbol_count, fourth_top, fourth_bottom, fourth_left, fourth_right)
                        bounding_boxes.append(bbox)

                        # Crop and save the symbol
                        final_symbol = binarized_image[fourth_top:fourth_bottom, fourth_left:fourth_right]
                        final_symbol = cv2.bitwise_not(final_symbol)
                        symbol_filename = os.path.join(save_directory, f"symbol_{symbol_count}.png")
                        cv2.imwrite(symbol_filename, final_symbol)
                        symbol_count += 1

# Step 2.2 (Save Bounding Boxes to CSV)
# Save bounding boxes to a CSV file
bbox_csv = os.path.join(save_directory, "bounding_boxes.csv")
with open(bbox_csv, "w") as f:
    f.write("number,top,bottom,left,right\n")
    for bbox in bounding_boxes:
        f.write(",".join(map(str, bbox)) + "\n")

print(f"Segmentation complete. Symbols saved to {save_directory}")

# SYMBOL RECOGNITION:
# Step 3.1 (use CNN model)
# Load model
model_path = '/Users/andrewyan/Desktop/OMR_Project/symbol_classification_model.keras'
train_dir = '/Users/andrewyan/Desktop/OMR_datasets/Rebelo Dataset/database/train' 
class_names = sorted(os.listdir(train_dir))  # Get class names from folder names
model = load_model(model_path)
print(f"Model loaded from {model_path}")

# Step 3.2 (Extract specific symbols)
# treble clefs, bass clefs, notes, notesOpen

# PITCH RECOGNITION:
# Step 4.1 (Determine center of noteheads)

# Step 4.2 (Determine staff reference line)

# Step 4.3 (Calculate distance to determine pitch, with clef modifier)

# SHEET MUSIC ANNOTATION
# Step 5.1 (Annotate original sheet music with note pitches)

