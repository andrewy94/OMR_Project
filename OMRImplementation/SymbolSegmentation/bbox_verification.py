import tkinter as tk
from tkinter import filedialog
import cv2
import os
import csv

root = tk.Tk()
root.withdraw()

# Set the initial directory
image_directory = '/Users/andrewyan/Desktop/OMR_Project/OMRImplementation/datasets/images'
bbox_directory = '/Users/andrewyan/Desktop/OMR_datasets/segmented_symbols/bounding_boxes.csv'

# Open a file dialog to select an image
file_path = filedialog.askopenfilename(
    title="Select an Image",
    initialdir=image_directory,
    filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif")]
)

if not file_path:
    print("No file selected!")
else:
    # Load the input image
    original_image = cv2.imread(file_path)
    if original_image is None:
        print(f"Unable to load image from {file_path}")
    else:
        # Read the bounding boxes from the CSV file
        bounding_boxes = []
        with open(bbox_directory, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header
            for row in reader:
                bbox = list(map(int, row))
                bounding_boxes.append(bbox)

        # Overlay the bounding boxes on the original image
        for bbox in bounding_boxes:
            symbol_number, top, bottom, left, right = bbox
            # Draw the rectangle
            cv2.rectangle(original_image, (left, top), (right, bottom), (0, 255, 0), 2)
            # Optionally, add a label with the symbol number
            cv2.putText(original_image, f"{symbol_number}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


        new_dir = os.path.join(os.path.dirname(file_path), "..", "bbox_annotated")
        save_path = os.path.join(
            new_dir,
            os.path.splitext(os.path.basename(file_path))[0] + "_bbox.png"
        )

        cv2.imwrite(save_path, original_image)
        print(f"Annotated image saved to {save_path}") 

        # Optionally display the annotated image
        cv2.imshow("Annotated Image", original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

