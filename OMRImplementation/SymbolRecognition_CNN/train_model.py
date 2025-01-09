import os
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers, models

# === Step 1: Split the Dataset === #

# Define paths
dataset_dir = "/Users/andrewyan/Desktop/OMR_datasets/Rebelo Dataset/database/custom"  # Path to your dataset
train_dir = "/Users/andrewyan/Desktop/OMR_datasets/Rebelo Dataset/database/train"           # Path to store training data
test_dir = "/Users/andrewyan/Desktop/OMR_datasets/Rebelo Dataset/database/test"             # Path to store test data

# Create training and testing directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Split the data
for folder in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder)
    if os.path.isdir(folder_path):
        # List all files in the current folder
        files = os.listdir(folder_path)
        train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
        
        # Create class folders in train and test directories
        os.makedirs(os.path.join(train_dir, folder), exist_ok=True)
        os.makedirs(os.path.join(test_dir, folder), exist_ok=True)

        # Move files
        for file in train_files:
            shutil.copy(os.path.join(folder_path, file), os.path.join(train_dir, folder))
        for file in test_files:
            shutil.copy(os.path.join(folder_path, file), os.path.join(test_dir, folder))

print("Dataset split into training and testing sets!")

# === Step 2: Prepare the Dataset for TensorFlow === #

# Load training and testing datasets
image_size = (128, 128)  # Resize images to 128x128
batch_size = 32

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=image_size,
    batch_size=batch_size
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=image_size,
    batch_size=batch_size
)

# Normalize the pixel values to [0, 1]
normalization_layer = layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# === Step 3: Build the CNN Model === #

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(os.listdir(train_dir)), activation='softmax')  # Number of classes
])

# === Step 4: Compile the Model === #

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# === Step 5: Train the Model === #

epochs = 10
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=epochs
)

# === Step 6: Evaluate the Model === #

test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# === Step 7: Save the Model (Optional) === #

model.save('symbol_classification_model.keras')
print("Model saved as 'symbol_classification_model.keras'")
