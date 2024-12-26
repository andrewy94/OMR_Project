import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
import os
from draw_axes import draw_axes

#withdraw the default Tkinter root window
root = tk.Tk()
root.withdraw()

#set the initial directory
initial_directory = '/Users/andrewyan/Desktop/OMR_Project/OMRImplementation/datasets/images'

#open a file dialog to select an image
file_path = filedialog.askopenfilename(
    title="Select an Image",
    initialdir=initial_directory,  # set the starting directory
    filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif")]
)

#start of deskewing process
if not file_path:
    print("No file selected!")
else:
    #load the input image
    image = cv2.imread(file_path)
    if image is None:
        print(f"Unable to load image from {file_path}")
    else:
        #show original image
        cv2.imshow('Original', image)
        cv2.waitKey(0)

        #do canny edge detection to find edges for next step, Hough Transform
        edges = cv2.Canny(image, 50, 150, apertureSize=3)

        #perform Standard Hough Transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        #copy of original image to keep an unmarked version at the end
        marked_image = image.copy()

        # draw detected lines
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(marked_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        #draw red axes (pre-deskewing)
        draw_axes(marked_image, color=(0, 0, 255))

        #show marked image with sHt lines and red axes
        cv2.imshow('Standard Hough Transform Lines', marked_image)
        cv2.waitKey(0)

        #calculate angles of lines and append angles to a list
        angles = []
        if lines is not None:
            for rho, theta in lines[:, 0]:
                angle_deg = np.degrees(theta) - 90  #convert to degrees and center around horizontal
                if -45 < angle_deg < 45:  #filter near-horizontal lines
                    angles.append(angle_deg)

        #error check for if angles empty
        if len(angles) > 0:
            # Compute the average angle
            skew_angle = np.mean(angles)
            print(f"Skew angle: {skew_angle:.2f} degrees")
        else:
            skew_angle = 0
            print("No significant lines detected for deskewing.")

        #calculate rotation matrix from skewed angle and center of image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)

        #deskew the marked image using rotation matrix
        deskewed_marked_image = cv2.warpAffine(marked_image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        #draw blue axes (post-deskewing)
        draw_axes(deskewed_marked_image, color=(255, 0, 0))

        #show deskewed marked image with blue axes
        cv2.imshow('Deskewed Image', deskewed_marked_image)
        cv2.waitKey(0)

        #deskew the image using rotation matrix
        deskewed_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        #save the image in the unmarked folder
        new_dir = os.path.join(os.path.dirname(file_path), "..", "deskewed")
        save_path = os.path.join(
            new_dir,
            os.path.splitext(os.path.basename(file_path))[0] + "_sHt_d.png"
        )
 
        #write deskewed image to files (fully unmarked)
        cv2.imwrite(save_path, deskewed_image)
        print(f"Deskewed image saved to {save_path}")

        #draw blue axes (post-deskewing)
        draw_axes(deskewed_image, color=(255, 0, 0))  # blue axes

        #show deskewed image with blue axes
        cv2.imshow('Deskewed Image', deskewed_image)
        cv2.waitKey(0)

# Destroy all OpenCV windows
cv2.destroyAllWindows()
