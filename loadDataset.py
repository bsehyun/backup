import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow import keras
import cv2



def load_data(input_dim):
    dataset_path = "./dataset/"
    categories = ["good", "bad"]
    labels = {category: idx for idx, category in enumerate(categories)}

    x_data = []
    y_data = []

    for category in categories:
        folder_path = os.path.join(dataset_path, category)
        
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            image = cv2.imread(img_path)
            
            upscaled = cv2.resize(image, (input_dim, input_dim), interpolation = cv2.INTER_CUBIC)
            
            b,g,r = cv2.split(upscaled)
            
            kernel = np.ones((5,5), np.float32)
            filled = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel)

            # _, binary = cv2.threshold(filled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

            # min_area = 100
            # filtered = np.zeros_like(binary)
            
            # for i in range(1, num_labels):
            #     if stats[i, cv2.CC_STAT_AREA] > min_area:
            #         filtered[labels == i] = 255
            # background = cv2.bitwise_not(filtered)
            # no_background = cv2.bitwise_and(b, b, mask= filtered)
            # kernel2 = np.ones((3,3), np.float32)
            # hole_filled = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel2)

            # impurities = cv2.bitwise_and(b, mask= hole_filled)
            # _, anomaly_mask = cv2.threshold(impurities, 30, 255, cv2.THRESH_BINARY)


            x_data.append(filled)
            y_data.append(labels[category])


    x_data = np.array(x_data, dtype=np.float32)
    y_data = np.array(y_data)
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42, stratify=y_data)
    
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    num_classes = 2
    input_dim = 64
    input_shape = (input_dim, input_dim, 1)

    x_train, x_test, y_train, y_test = load_data(input_dim)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i])
    plt.show()