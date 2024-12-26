import tkinter as tk
from tkinter import filedialog
import cv2
import os

# Withdraw the default Tkinter root window
root = tk.Tk()
root.withdraw()

# Set the initial directory
initial_directory = '/Users/andrewyan/Desktop/OMRSheetMusicReader/OMRImplementation/datasets/images'

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
        
        #blur   
        blur = cv2.GaussianBlur(image, (0,0), sigmaX=33, sigmaY=33)
        cv2.imshow('blur', blur)
        cv2.waitKey(0)

        #divide
        divide = cv2.divide(image, blur, scale=255)
        cv2.imshow('divide', divide)
        cv2.waitKey(0)

        ret,binarized_image = cv2.threshold(divide,200,255,cv2.THRESH_BINARY)
        # Save the grayscale image in the same directory as the original
        new_dir = os.path.join(os.path.dirname(file_path), "..", "binarized")
        save_path = os.path.join(
            new_dir,
            os.path.splitext(os.path.basename(file_path))[0] + "_b.png"
        )
        cv2.imwrite(save_path, binarized_image)
        print(f"binarized image saved to {save_path}")

        cv2.imshow('Binarized', binarized_image)
        cv2.waitKey(0)

# Destroy all OpenCV windows
cv2.destroyAllWindows()
