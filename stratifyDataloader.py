import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class StratifiedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, batch_size, target_size=(224, 224), shuffle=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.class_0_images, self.class_1_images = self._load_image_paths()
        self.on_epoch_end()
        
        # Image augmentation for class_1 only
        self.class_1_datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
                                                 shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    
    def _load_image_paths(self):
        class_0_dir = os.path.join(self.data_dir, 'class_0')
        class_1_dir = os.path.join(self.data_dir, 'class_1')
        class_0_images = [os.path.join(class_0_dir, f) for f in os.listdir(class_0_dir) if f.endswith(('.jpg', '.png'))]
        class_1_images = [os.path.join(class_1_dir, f) for f in os.listdir(class_1_dir) if f.endswith(('.jpg', '.png'))]
        return class_0_images, class_1_images
    
    def __len__(self):
        return int(np.floor(len(self.class_0_images + self.class_1_images) / self.batch_size))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.class_0_images)
            np.random.shuffle(self.class_1_images)
    
    def __getitem__(self, index):
        class_0_count = int(self.batch_size * 0.75)  # 75% class_0
        class_1_count = self.batch_size - class_0_count  # 25% class_1

        class_0_batch = self.class_0_images[index * class_0_count:(index + 1) * class_0_count]
        class_1_batch = self.class_1_images[index * class_1_count:(index + 1) * class_1_count]
        
        images, labels = [], []
        
        for img_path in class_0_batch:
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=self.target_size)
            img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            images.append(img)
            labels.append(0)
        
        for img_path in class_1_batch:
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=self.target_size)
            img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img = self.class_1_datagen.random_transform(img)  # Augmentation 적용
            images.append(img)
            labels.append(1)
        
        return np.array(images), np.array(labels)

# 사용 예시
data_gen = StratifiedDataGenerator(data_dir='image', batch_size=32)
for x_batch, y_batch in data_gen:
    print(x_batch.shape, y_batch.shape)
    break
