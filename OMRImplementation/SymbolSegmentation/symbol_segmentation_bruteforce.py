import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
import os
import shutil

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

    # Define the save directory
    save_directory = "/Users/andrewyan/Desktop/OMR_datasets/segmented_symbols"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Erase the save directory if it exists and recreate it
    if os.path.exists(save_directory):
        shutil.rmtree(save_directory)  # Remove the existing directory
    os.makedirs(save_directory)  # Create a new empty directory

    bounding_boxes = []
    symbol_count = 0

    # Horizontal projection for initial segmentation
    h_proj = np.sum(binarized_image, axis=1)
    h_segments = find_segments(h_proj)

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

    # Save bounding boxes to a CSV file
    bbox_csv = os.path.join(save_directory, "bounding_boxes.csv")
    with open(bbox_csv, "w") as f:
        f.write("number,top,bottom,left,right\n")
        for bbox in bounding_boxes:
            f.write(",".join(map(str, bbox)) + "\n")

    print(f"Segmentation complete. Symbols saved to {save_directory}")
