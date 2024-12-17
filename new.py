import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input
from PIL import Image
import shutil

# Load the pre-trained model
model = tf.keras.models.load_model("image_classifier_model.h5")

# Compile the model with appropriate optimizer, loss function, and metrics
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Directory for raw images
raw_img_directory = "raw-img"

# Directory for feedback images
feedback_images_directory = "feedback_images"

# Dictionary to map class indices to class labels
map_dict = {}

# Function to predict image class with a given confidence threshold
def predict_image(img_array, confidence_threshold=0.995):
    raw_predictions = model.predict(img_array)
    prediction_confidence = np.max(raw_predictions)
    predicted_class_index = np.argmax(raw_predictions)
    if prediction_confidence >= confidence_threshold:
        predicted_class = map_dict.get(predicted_class_index)
        return predicted_class, prediction_confidence
    else:
        return None, None

# Function to read class labels from a text file and populate the mapping dictionary
def read_class_labels_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for index, label in enumerate(lines):
            map_dict[index] = label.strip()

# Read class labels from a text file
read_class_labels_from_file("class_labels.txt")

# Function to process a single image
def process_single_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize the image to match the model's expected input shape
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = mobilenet_v2_preprocess_input(img_array)

    # Predict class label
    predicted_class, prediction_confidence = predict_image(img_array)
    if predicted_class is not None:
        # Display prediction result
        prediction_result = f"Predicted Label for the image is {predicted_class} with confidence {prediction_confidence:.4f}"
        # Ask for user feedback
        feedback = messagebox.askquestion("Feedback", prediction_result + "\nIs this prediction correct?")
        if feedback == "yes":
            return predicted_class
        else:
            correct_label = simpledialog.askstring("Correct Label", "Please enter the correct label:")
            if correct_label:
                map_dict[len(map_dict)] = correct_label
                # Check if the folder with the same label already exists
                label_folder = os.path.join(raw_img_directory, correct_label)
                if not os.path.exists(label_folder):
                    os.makedirs(label_folder)
                # Copy the image to the folder with the same label
                new_image_path = os.path.join(label_folder, os.path.basename(image_path))
                shutil.copy(image_path, new_image_path)
                return correct_label
            else:
                return None
    else:
        add_new_label = messagebox.askquestion("New Label", "The predicted class is not in the map. Do you want to add it?")
        if add_new_label == "yes":
            new_label = simpledialog.askstring("New Label", "Please enter a label for the new class:")
            if new_label:
                map_dict[len(map_dict)] = new_label
                # Create folder for the new label inside raw-img directory
                new_label_folder = os.path.join(raw_img_directory, new_label)
                os.makedirs(new_label_folder, exist_ok=True)
                # Copy the image to the new folder
                new_image_path = os.path.join(new_label_folder, os.path.basename(image_path))
                shutil.copy(image_path, new_image_path)
                return new_label
            else:
                return None
        else:
            # Save the image to the feedback images directory for retraining
            shutil.copy(image_path, feedback_images_directory)
            return None

# Function to process multiple images from a folder
def process_images_in_folder(folder_path):
    if os.path.isdir(folder_path):
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        results = []
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            result = process_single_image(image_path)
            if result is not None:
                results.append((image_file, result))
        return results
    else:
        return "Invalid folder path."

# Function to handle button click event for processing a folder
def process_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        results = process_images_in_folder(folder_path)
        formatted_results = "\n".join([f"Image: {result[0]}\nPredicted Label: {result[1]}" for result in results])
        messagebox.showinfo("Results", formatted_results)

# Function to handle button click event for processing a single image
def process_single_image_gui():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        result = process_single_image(file_path)
        if result is not None:
            messagebox.showinfo("Result", f"Predicted Label: {result}")

# Create main application window
root = tk.Tk()
root.title("Image Classifier")

# Create a button to select folder
select_folder_button = tk.Button(root, text="Select Folder", command=process_folder)
select_folder_button.pack(pady=10)

# Create a button to upload a single image
upload_image_button = tk.Button(root, text="Upload Single Image", command=process_single_image_gui)
upload_image_button.pack(pady=10)

# Run the GUI application
root.mainloop()
