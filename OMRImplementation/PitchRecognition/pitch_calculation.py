import os
import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog

def load_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to load image from {file_path}")
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_not(binary)

def detect_staves_and_middle_lines(image):
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    _, detect_horizontal = cv2.threshold(detect_horizontal, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    line_positions = [cv2.boundingRect(cnt)[1] for cnt in contours]
    line_positions.sort()

    stave_groups = []
    threshold = 3
    for i in range(len(line_positions) - 4):
        group = line_positions[i:i + 5]
        spacings = [group[j + 1] - group[j] for j in range(4)]
        if max(spacings) - min(spacings) <= threshold:
            stave_groups.append(group)
            i += 4

    middle_lines = [group[2] for group in stave_groups]
    return stave_groups, middle_lines

def determine_note_orientation(note_image):
    h_proj = np.sum(note_image, axis=1)
    mid_index = len(h_proj) // 2
    top_half = sum(h_proj[:mid_index])
    bottom_half = sum(h_proj[mid_index:])
    return top_half > bottom_half

def calculate_pitch(note_bbox, middle_lines, stave_spacing, clef):
    x, y, w, h = note_bbox
    note_bottom = y + h

    closest_middle_line = min(middle_lines, key=lambda ml: abs(ml - note_bottom))
    distance_from_middle = (note_bottom - closest_middle_line) / stave_spacing

    if clef == "treble":
        pitch_map = ["B", "C", "D", "E", "F", "G", "A"]
        pitch_index = 2  # Middle line = B
    elif clef == "bass":
        pitch_map = ["D", "E", "F", "G", "A", "B", "C"]
        pitch_index = 0  # Middle line = D
    else:
        raise ValueError("Unsupported clef type")

    pitch_index += round(-distance_from_middle * 2)
    return pitch_map[pitch_index % len(pitch_map)] + str(4 + pitch_index // len(pitch_map))

def annotate_image(image, annotations):
    for annotation in annotations:
        x, y, w, h, pitch = annotation
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, pitch, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return image

def main():
    root = tk.Tk()
    root.withdraw()

    # Select directories and files
    image_file = filedialog.askopenfilename(title="Select Original Image")
    notes_folder = filedialog.askdirectory(title="Select Folder Containing Note Symbols")
    clefs_folder = filedialog.askdirectory(title="Select Folder Containing Clef Symbols")
    csv_file = filedialog.askopenfilename(title="Select CSV File with Bounding Box Info")

    # Load data
    original_image = load_image(image_file)
    stave_groups, middle_lines = detect_staves_and_middle_lines(original_image)
    stave_spacing = np.mean([group[1] - group[0] for group in stave_groups])

    bounding_boxes = pd.read_csv(csv_file)
    results = []

    for _, row in bounding_boxes.iterrows():
        symbol_file = os.path.join(notes_folder if "note" in row['type'] else clefs_folder, f"{row['name']}.png")
        symbol_image = load_image(symbol_file)

        if "note" in row['type']:
            upside_down = determine_note_orientation(symbol_image)
            if upside_down:
                row['y'] -= stave_spacing

            clef = "treble" if row['last_detected_clef'] == "treble" else "bass"
            pitch = calculate_pitch((row['x'], row['y'], row['width'], row['height']), middle_lines, stave_spacing, clef)
            results.append((row['x'], row['y'], row['width'], row['height'], pitch))

    # Annotate image
    annotated_image = annotate_image(original_image.copy(), results)
    output_image_path = os.path.join(os.path.dirname(image_file), "annotated_image.png")
    cv2.imwrite(output_image_path, annotated_image)

    # Save results
    results_df = pd.DataFrame([(res[4],) for res in results], columns=["Pitch"])
    results_df.to_csv(os.path.join(os.path.dirname(csv_file), "output_pitches.csv"), index=False)

    print(f"Pitch calculation complete. Results saved to output_pitches.csv and annotated image saved to {output_image_path}.")

if __name__ == "__main__":
    main()
