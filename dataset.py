# Import necessary libraries
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Display the shape of the dataset
print("Train images shape:", train_images.shape)  # (60000, 28, 28)
print("Train labels shape:", train_labels.shape)  # (60000,)
print("Test images shape:", test_images.shape)    # (10000, 28, 28)
print("Test labels shape:", test_labels.shape)    # (10000,)

# Function to visualize a single image and its label
def visualize_single_image(index):
    plt.imshow(train_images[index], cmap='gray')  # Display the image in grayscale
    plt.title(f"Label: {train_labels[index]}")    # Show the label as the title
    plt.axis('off')  # Turn off axes for better visibility
    plt.show()

# Visualize the first image in the training dataset
print("\nVisualizing the first image in the training dataset:")
visualize_single_image(0)

# Function to visualize multiple images in a grid
def visualize_multiple_images(num_images=12):
    plt.figure(figsize=(10, 5))  # Set the figure size
    for i in range(num_images):
        plt.subplot(3, 4, i+1)  # Create a 3x4 grid of subplots
        plt.imshow(train_images[i], cmap='gray')  # Display each image in grayscale
        plt.title(f"Label: {train_labels[i]}")    # Show the label
        plt.axis('off')  # Turn off axes
    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()

# Visualize the first 12 images in the training dataset
print("\nVisualizing the first 12 images in the training dataset:")
visualize_multiple_images()

# Function to inspect the raw pixel values of an image
def inspect_raw_data(index):
    print(f"\nRaw pixel values of image at index {index}:")
    print(train_images[index])

# Inspect the raw pixel values of the first image
inspect_raw_data(0)
