import tkinter as tk
from tkinter import filedialog
import cv2
import os

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

        # Use the cvtColor() function to grayscale the image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        new_dir = os.path.join(os.path.dirname(file_path), "..", "grayscaled")
        save_path = os.path.join(
            new_dir,
            os.path.splitext(os.path.basename(file_path))[0] + "_g.png"
        )

        cv2.imwrite(save_path, gray_image)
        print(f"Grayscale image saved to {save_path}")

        cv2.imshow('Grayscale', gray_image)
        cv2.waitKey(0)

# Destroy all OpenCV windows
cv2.destroyAllWindows()
