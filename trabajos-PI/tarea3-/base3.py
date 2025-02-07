import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from google.colab import drive
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

drive.mount('/content/drive')
path = '/content/drive/My Drive/dataset'
train_dir = '/content/drive/My Drive/dataset/train'
validation_dir = '/content/drive/My Drive/dataset/validation'
test_dir = '/content/drive/My Drive/dataset/test'

train_cats_dir = '/content/drive/My Drive/dataset/train/bcc'
train_dogs_dir = '/content/drive/My Drive/dataset/train/nv'

validation_cats_dir = '/content/drive/My Drive/dataset/validation/bcc'
validation_dogs_dir ='/content/drive/My Drive/dataset/validation/nv'

test_cats_dir = '/content/drive/My Drive/dataset/test/bcc'
test_dogs_dir = '/content/drive/My Drive/dataset/test/nv'

num_cats_train = len(os.listdir(train_cats_dir))
num_dogs_train = len(os.listdir(train_dogs_dir))

num_cats_validation = len(os.listdir(validation_cats_dir))
num_dogs_validation = len(os.listdir(validation_dogs_dir))

num_cats_test = len(os.listdir(test_cats_dir))
num_dogs_test = len(os.listdir(test_dogs_dir))

print('Total Training Images of Cats',num_cats_train)
print('Total Training Images of Dogs',num_dogs_train)
print('\n************************\n')
print('Total Validation Images of Cats',num_cats_validation)
print('Total Validation Imagges of Dogs',num_dogs_validation)
print('\n************************\n')
print('Total Testing Images of Cats',num_cats_test)
print('Total Testing Images of Dogs',num_dogs_test)

total_train = num_cats_train+num_dogs_train
total_validation = num_cats_validation+num_dogs_validation
total_test = num_cats_test+num_dogs_test

print('Total Training Images',total_train)
print('Total Validation Images',total_validation)
print('Total Testing Images',total_test)

BATCH_SIZE = 100
IMG_SHAPE  = 150

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()
    
image_gen_train = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,
                                     zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')

train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE, directory=train_dir, shuffle=True, target_size=(IMG_SHAPE, IMG_SHAPE), class_mode='binary')

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=BATCH_SIZE, directory=validation_dir, target_size=(IMG_SHAPE, IMG_SHAPE), class_mode='binary')

image_gen_test = ImageDataGenerator(rescale=1./255)

test_data_gen = image_gen_test.flow_from_directory(batch_size=BATCH_SIZE, directory=test_dir, target_size=(IMG_SHAPE, IMG_SHAPE), class_mode='binary')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()


epocas = 3
history = model.fit( train_data_gen, steps_per_epoch=total_train // BATCH_SIZE, 
                    epochs=epocas,
                    validation_data=val_data_gen,
                    validation_steps=total_validation // BATCH_SIZE )

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epocas)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()