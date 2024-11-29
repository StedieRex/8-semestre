# se sobreescribre el metodo ImageDataGenerator de keras para preprocesar las imagenes
# en la funcion preprocess_image se aplican todos los filtros que necesite, como la transformada de fourier
import os
import cv2  # Biblioteca OpenCV
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

def procesamiento_Imagenes(image_path):
    img = cv2.imread(image_path)  # Cargar imagen con OpenCV
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convertir a blanco y negro
    return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)  # Convertir de nuevo a tres canales

class MiImageDataGenerator(ImageDataGenerator):# el error es aproposito para identificar donde hacer los cambios
    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = super()._get_batches_of_transformed_samples(index_array)
        for i in range(len(batch_x)):
            img_path = self.filepaths[index_array[i]]
            batch_x[i] = procesamiento_Imagenes(img_path)  # Preprocesar la imagen
        return batch_x

image_gen_train = CustomImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                           zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE, directory=train_dir, shuffle=True, target_size=(IMG_SHAPE, IMG_SHAPE), class_mode='categorical')

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

image_gen_val = CustomImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=BATCH_SIZE, directory=validation_dir, target_size=(IMG_SHAPE, IMG_SHAPE), class_mode='categorical')

image_gen_test = CustomImageDataGenerator(rescale=1./255)

test_data_gen = image_gen_test.flow_from_directory(batch_size=BATCH_SIZE, directory=test_dir, target_size=(IMG_SHAPE, IMG_SHAPE), class_mode='categorical')
