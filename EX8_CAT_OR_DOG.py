#CAT OR GOD
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,
BatchNormalization
from tensorflow.keras.layers.experimental import preprocessing
from keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.applications import VGG16
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
# Define the test and train data directories
test_data_dir = r'C:\Users\ygoku\Desktop\cat_dog\test'
train_data_dir = r'C:\Users\ygoku\Desktop\cat_dog\train'
def is_image(file_path):
 try:
 with Image.open(file_path) as img:
 return True
 except Exception as e:
 return False
def preprocess_image(file_path):
 try:
 with Image.open(file_path) as img:
# Ensure the image has 3 channels (RGB)
 if img.mode != 'RGB':
 print(f"Converting image to RGB: {file_path}")
 img = img.convert('RGB')
 img.save(file_path) # Save the converted image back to the file
 return img
 except Exception as e:
 print(f"Error preprocessing image: {file_path}, Error: {str(e)}")
 return None
def remove_invalid_images_from_subfolders(main_folder):
 subfolders = ['cats', 'dogs'] # Add more subfolders if needed
 for subfolder in subfolders:
 subfolder_path = os.path.join(main_folder, subfolder) 
 for filename in os.listdir(subfolder_path):
 file_path = os.path.join(subfolder_path, filename)
 # Check if the file is a valid image
 if is_image(file_path):
 img = preprocess_image(file_path)
 if img is not None:
 # Save the image back to the file if it needed conversion
 if img.mode != 'RGB':
 img.save(file_path)
 else:
 print(f"Removing invalid or corrupt file: {file_path}")
 os.remove(file_path)
 else:
 print(f"Skipping non-image file: {file_path}")
train_data=tf.keras.utils.image_dataset_from_directory(train_data_dir, color_mode='rgb',
image_size=(224, 224), batch_size=32)
test_data=tf.keras.utils.image_dataset_from_directory(test_data_dir, color_mode='rgb',
image_size=(224, 224), batch_size=32)
batch_train_data = train_data.as_numpy_iterator().next()
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch_train_data[0][:4]):
 ax[idx].imshow(img.astype(int))
 ax[idx].title.set_text(batch_train_data[1][idx])
scaled_train_data = train_data.map(lambda x, y: (x / 255.0, y))
scaled_test_data = test_data.map(lambda x, y: (x / 255.0, y))
scaled_batch_train = scaled_train_data.as_numpy_iterator().next()
scaled_batch_test = scaled_train_data.as_numpy_iterator().next()
#min value
scaled_batch_train[0].min()
#min value
scaled_batch_test[0].min()
scaled_batch_train[0].max()
scaled_batch_test[0].max()
val_size = int(len(test_data) * 0.2)
test_size = int(len(test_data) * 0.8)
scaled_val_data = scaled_test_data.take(val_size)
scaled_test_data = scaled_test_data.skip(val_size).take(test_size)
len(scaled_test_data)
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(scaled_batch_train[0][:4]):
 ax[idx].imshow(img)
 ax[idx].title.set_text(scaled_batch_train[1][idx])
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
 layer.trainable = False
model = Sequential()
model.add(preprocessing.RandomFlip('horizontal')),
model.add(preprocessing.RandomContrast(0.5)),
model.add(preprocessing.RandomTranslation(0.2, 0.2))
model.add(base_model)
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # Binary classification, so use 'sigmoid' activation
model.compile(optimizer='adam',
 loss=tf.losses.BinaryCrossentropy(),
 metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Define a learning rate schedule function
def learning_rate_schedule(epoch):
 if epoch < 20:
 return 0.001
 elif epoch < 40:
 return 0.0001
 else:
 return 0.00001
# Create a learning rate scheduler
lr_scheduler = LearningRateScheduler(learning_rate_schedule)
hist = model.fit(scaled_train_data, epochs=50, callbacks=[early_stopping], 
validation_data=scaled_val_data)
predictions = model.predict(scaled_test_data)
if predictions[0][0] >= 0.5:
 print("It's a dog!") # Your positive class label
else:
 print("It's a cat!")
pre = Precision()
re = Recall()
acc = BinaryAccuracy()
for batch in scaled_test_data.as_numpy_iterator():
 X, y = batch
 yhat = model.predict(X)
 pre.update_state(y, yhat)
 re.update_state(y, yhat)
 acc.update_state(y, yhat)
print(f'Precision {pre.result().numpy()} \nRecall {re.result().numpy()} \nAccuracy 
{acc.result().numpy()}') 