import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras import losses, metrics
from sklearn.model_selection import train_test_split

IMG_SIZE = 160
COLOR_PATH = "C:/Users/raksh/Documents/CV PROJECT/landscape Images/color"
GRAY_PATH = "C:/Users/raksh/Documents/CV PROJECT/landscape Images/gray"

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

color_images = color_images / 255.0
gray_images = gray_images / 255.0
gray_train, gray_val, color_train, color_val = train_test_split(
    gray_images, color_images, test_size=0.2, random_state=42
)

model_path = "C:/Users/raksh/Documents/CV PROJECT/colorizer_model.h5"
colorizer = load_model(model_path, custom_objects={'mse': losses.MeanSquaredError()})
colorizer.compile(
    optimizer='adam',
    loss=losses.MeanSquaredError(),
    metrics=[metrics.MeanAbsoluteError()]
)

loss, mae = colorizer.evaluate(gray_val, color_val, verbose=1)
print(f"Loss (MSE): {loss}")
print(f"Mean Absolute Error (MAE): {mae}")
#The model is evaluated using the validation set (gray_val and color_val), which calculates the loss and MAE between the predicted and true color images.