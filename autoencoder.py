import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

IMG_SIZE = 160
COLOR_PATH = "C:/Users/raksh/Documents/CV PROJECT/landscape Images/color"
GRAY_PATH = "C:/Users/raksh/Documents/CV PROJECT/landscape Images/gray"

#Color images are converted to RGB and Grayscale images are converted to grayscale and then reshaped to ensure that they have a single channel (axis=-1).
def load_images(folder_path, is_color):
    images = []
    for file_name in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, file_name)
        img = cv2.imread(img_path)

        if is_color:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=-1)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)

    return np.array(images)

color_images = load_images(COLOR_PATH, is_color=True)
gray_images = load_images(GRAY_PATH, is_color=False)

#Images are divided by 255.0 to normalize the pixel values to a range of 0 to 1 (since images are usually represented with values between 0 and 255).
color_images = color_images / 255.0
gray_images = gray_images / 255.0

#train_test_split splits the data into training and validation sets, with 20% of the data used for validation.
gray_train, gray_val, color_train, color_val = train_test_split(
    gray_images, color_images, test_size=0.2, random_state=42
)


#This model uses an encoder-decoder architecture, where the encoder (with Conv2D and MaxPooling2D layers) extracts features from grayscale images, and the decoder (with UpSampling2D layers) reconstructs the colorized RGB image. The final output is generated using a Conv2D layer with a sigmoid activation function.
def Colorizer(input_shape):
    encoder_input = layers.Input(shape=input_shape, name="grayscale_input")
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoder_output = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    return Model(encoder_input, decoder_output, name="Colorizer")

colorizer = Colorizer((IMG_SIZE, IMG_SIZE, 1))

#The model is compiled with the Adam optimizer, mean squared error (mse) as the loss function, and mean absolute error (mae) as an additional metric to monitor the model's performance.
colorizer.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

epochs = 50
batch_size = 32

#fit(): The model is trained on the grayscale images (gray_train) and learns to predict the color images (color_train). It also validates the performance on the validation set (gray_val, color_val) during training.
history = colorizer.fit(
    gray_train, color_train,
    validation_data=(gray_val, color_val),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)   

#The model is saved to a file (colorizer_model.h5) so that it can be reused or deployed later.
model_path = "C:/Users/raksh/Documents/CV PROJECT/colorizer_model.h5"
colorizer.save(model_path)
print(f"Model saved at {model_path}")

#The model predicts colorized versions of the first 10 grayscale images in the validation set.
predicted_images = colorizer.predict(gray_val[:10])

plt.figure(figsize=(15, 10))
for i in range(10):
    plt.subplot(3, 10, i + 1)
    plt.imshow(gray_val[i].squeeze(), cmap='gray')
    plt.title("Gray")
    plt.axis('off')

    plt.subplot(3, 10, i + 11)
    plt.imshow(predicted_images[i])
    plt.title("Predicted")
    plt.axis('off')

    plt.subplot(3, 10, i + 21)
    plt.imshow(color_val[i])
    plt.title("Original")
    plt.axis('off')

plt.tight_layout()
plt.show()
#Encoder: The encoder captures the essential features of the input grayscale image using several Conv2D layers with ReLU activation. These layers are followed by MaxPooling2D layers that down-sample the image, progressively reducing its spatial dimensions while extracting hierarchical features.
#Decoder: The decoder reconstructs the image back to its original size using UpSampling2D layers to upsample the feature maps, followed by Conv2D layers to refine the colorized image. The final layer uses a sigmoid activation to output the colorized image in RGB format.