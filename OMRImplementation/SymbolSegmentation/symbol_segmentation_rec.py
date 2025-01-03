import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
import os
import shutil

def find_segments(proj):
    segments = []
    in_segment = False
    start = 0
    for i, value in enumerate(proj):
        if value > 0 and not in_segment:
            in_segment = True
            start = i
        elif value == 0 and in_segment:
            in_segment = False
            end = i
            segments.append((start, end))
    if in_segment:
        segments.append((start, len(proj)))
    return segments

def recursive_segmentation(image, depth, max_depth, direction, symbol_count, bounding_boxes, save_directory):
    if depth > max_depth or image.size == 0:
        return symbol_count

    if direction == "horizontal":
        projection = np.sum(image, axis=1)
    else:  # vertical
        projection = np.sum(image, axis=0)

    segments = find_segments(projection)

    for start, end in segments:
        if direction == "horizontal":
            sub_image = image[start:end, :]
        else:  # vertical
            sub_image = image[:, start:end]

        # Skip 1x1 segments
        if sub_image.shape[0] <= 1 and sub_image.shape[1] <= 1:
            continue

        if depth == max_depth:
            # Save the final segment
            final_symbol = cv2.bitwise_not(sub_image)
            symbol_filename = os.path.join(save_directory, f"symbol_{symbol_count}.png")
            cv2.imwrite(symbol_filename, final_symbol)

            bbox = (symbol_count, start, end, 0, 0) if direction == "horizontal" else (symbol_count, 0, 0, start, end)
            bounding_boxes.append(bbox)

            symbol_count += 1
        else:
            # Alternate direction for the next recursion
            next_direction = "vertical" if direction == "horizontal" else "horizontal"
            symbol_count = recursive_segmentation(
                sub_image,
                depth + 1,
                max_depth,
                next_direction,
                symbol_count,
                bounding_boxes,
                save_directory
            )

    return symbol_count

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
    save_directory = "/Users/andrewyan/Desktop/OMR_datasets/segmented_symbols"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Erase the save directory if it exists and recreate it
    if os.path.exists(save_directory):
        shutil.rmtree(save_directory)
    os.makedirs(save_directory)

    bounding_boxes = []
    symbol_count = 0
    max_depth = 5

    # Start recursive segmentation
    symbol_count = recursive_segmentation(
        binarized_image,
        depth=1,
        max_depth=max_depth,
        direction="horizontal",
        symbol_count=symbol_count,
        bounding_boxes=bounding_boxes,
        save_directory=save_directory
    )

    # Save bounding boxes to a CSV file
    bbox_csv = os.path.join(save_directory, "bounding_boxes.csv")
    with open(bbox_csv, "w") as f:
        f.write("number,top,bottom,left,right\n")
        for bbox in bounding_boxes:
            f.write(",".join(map(str, bbox)) + "\n")

    print(f"Segmentation complete. Symbols saved to {save_directory}")
