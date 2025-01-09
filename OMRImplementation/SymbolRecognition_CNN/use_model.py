import tkinter as tk
from tkinter import filedialog
import cv2
import os
import csv
import numpy as np
from keras.models import load_model

# Initialize tkinter
root = tk.Tk()
root.withdraw()

# === Load the Symbol Classification Model === #
model_path = '/Users/andrewyan/Desktop/OMR_Project/symbol_classification_model.keras'  # Path to your trained model
train_dir = '/Users/andrewyan/Desktop/OMR_datasets/Rebelo Dataset/database/train'  # Path to your training directory (to get class names)
class_names = sorted(os.listdir(train_dir))  # Get class names from folder names
model = load_model(model_path)
print(f"Model loaded from {model_path}")

# === Function to Preprocess Symbols for Classification === #
def preprocess_image(cropped_image):
    # Resize the image to the model input size and normalize
    img_array = cv2.resize(cropped_image, (128, 128))
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# === Main Script === #
# Set the initial directories
image_directory = '/Users/andrewyan/Desktop/OMR_Project/OMRImplementation/datasets/images'
bbox_directory = '/Users/andrewyan/Desktop/OMR_datasets/segmented_symbols/bounding_boxes.csv'
output = '/Users/andrewyan/Desktop/OMR_datasets/output'

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
        # Create output directory for classified symbols
        output_dir = os.path.join(os.path.dirname(output), "..", "classified_symbols")
        os.makedirs(output_dir, exist_ok=True)
        for class_name in class_names:
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

        # Read the bounding boxes from the CSV file
        bounding_boxes = []
        with open(bbox_directory, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header
            for row in reader:
                bbox = list(map(int, row))
                bounding_boxes.append(bbox)

        # Overlay the bounding boxes and classify symbols
        for bbox in bounding_boxes:
            symbol_number, top, bottom, left, right = bbox
            
            # Crop the image using the bounding box
            cropped_symbol = original_image[top:bottom, left:right]
            
            # Skip empty or invalid crops
            if cropped_symbol.size == 0:
                continue

            # Classify the cropped symbol
            processed_image = preprocess_image(cropped_symbol)
            predictions = model.predict(processed_image)
            predicted_class_index = np.argmax(predictions)
            predicted_class = class_names[predicted_class_index]

            # Save the cropped symbol to its respective class folder
            class_folder = os.path.join(output_dir, predicted_class)
            symbol_filename = f"symbol_{symbol_number}.png"
            symbol_save_path = os.path.join(class_folder, symbol_filename)
            cv2.imwrite(symbol_save_path, cropped_symbol)
            print(f"Saved symbol {symbol_number} to {symbol_save_path}")

            # Draw the rectangle on the original image
            cv2.rectangle(original_image, (left, top), (right, bottom), (0, 255, 0), 2)

            # Add the predicted class label above the bounding box
            label = f"{predicted_class} ({symbol_number})"
            cv2.putText(
                original_image,
                label,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 0, 0),
                1
            )

        # Save the annotated image to a new directory
        new_dir = os.path.join(os.path.dirname(file_path), "..", "bbox_annotated")
        os.makedirs(new_dir, exist_ok=True)
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
