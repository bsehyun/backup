import os
from PIL import Image
from PIL import ImageEnhance
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras.optimizers import Adam 
from swinTransformer import SwinTransformer
from util import PatchEmbedding, PatchMerging, PatchExtract

from tensorflow.keras.callbacks import EarlyStopping
import cv2

# def load_data(input_dim):

#     dataset_path = "./dataset/"
#     categories = ["good", "bad"]
#     labels = {category: idx for idx, category in enumerate(categories)}

#     x_data = []
#     y_data = []


#     for category in categories:
#         folder_path = os.path.join(dataset_path, category)
        
#         for filename in os.listdir(folder_path):
#             img_path = os.path.join(folder_path, filename)

#             image = Image.open(img_path)
            
#             # input image > color space conversation (rgb to hsv) > 
#             # engance saturation of impurity regions > 
#             # threshold segmentation using otsu method > 
#             # sobel operator is used to filter edge > mark goreground and background 
#             # > hole filing 

              
#             x_data.append(image_array)
#             y_data.append(labels[category])


#     x_data = np.array(x_data, dtype=np.float32)
#     y_data = np.array(y_data)
    
#     x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42, stratify=y_data)

#     return x_train, x_test, y_train, y_test 
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
            
            try:
                image = Image.open(img_path).convert("RGB")
                image = image.resize((input_dim, input_dim))
                
                # 1️⃣ RGB -> HSV 변환
                hsv = image.convert("HSV")
                h, s, v = hsv.split()
                s = np.array(s, dtype=np.uint8)
                
                # 2️⃣ 불순물 영역의 채도 증가
                s[s < 50] = np.clip(s[s < 50] * 1.5, 0, 255)
                s = Image.fromarray(s)
                hsv = Image.merge("HSV", (h, s, v))
                image = hsv.convert("RGB")
                
                # 3️⃣ Otsu Thresholding (이진화)
                gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # 4️⃣ Sobel 필터 적용 (엣지 검출)
                sobelx = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(binary, cv2.CV_64F, 0, 1, ksize=3)
                edges = cv2.magnitude(sobelx, sobely)
                edges = np.uint8(edges / np.max(edges) * 255)
                
                # 5️⃣ 전경 및 배경 마킹
                foreground = cv2.bitwise_and(binary, binary, mask=edges)
                background = cv2.bitwise_not(foreground)
                
                # 6️⃣ Hole Filling 적용
                hole_filled = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
                
                # 7️⃣ 모델 입력을 위한 정규화
                image_array = np.array(hole_filled, dtype=np.float32) / 255.0
                
                x_data.append(image_array)
                y_data.append(labels[category])
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

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