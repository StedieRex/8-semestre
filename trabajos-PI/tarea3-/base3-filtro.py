import cv2
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

train_nv_dir = '/content/drive/My Drive/dataset/train/nv'
train_mel_dir = '/content/drive/My Drive/dataset/train/mel'
train_df_dir = '/content/drive/My Drive/dataset/train/df'
train_bkl_dir = '/content/drive/My Drive/dataset/train/bkl'
train_bcc_dir = '/content/drive/My Drive/dataset/train/bcc'

validation_nv_dir = '/content/drive/My Drive/dataset/validation/nv'
validation_mel_dir = '/content/drive/My Drive/dataset/validation/mel'
validation_df_dir = '/content/drive/My Drive/dataset/validation/df'
validation_bkl_dir = '/content/drive/My Drive/dataset/validation/bkl'
validation_bcc_dir = '/content/drive/My Drive/dataset/validation/bcc'

test_nv_dir = '/content/drive/My Drive/dataset/test/nv'
test_mel_dir = '/content/drive/My Drive/dataset/test/mel'
test_df_dir = '/content/drive/My Drive/dataset/test/df'
test_bkl_dir = '/content/drive/My Drive/dataset/test/bkl'
test_bcc_dir = '/content/drive/My Drive/dataset/test/bcc'

num_nv_train = len(os.listdir(train_nv_dir))
num_mel_train = len(os.listdir(train_mel_dir))
num_df_train = len(os.listdir(train_df_dir))
num_bkl_train = len(os.listdir(train_bkl_dir))
num_bcc_train = len(os.listdir(train_bcc_dir))

num_nv_validation = len(os.listdir(validation_nv_dir))
num_mel_validation = len(os.listdir(validation_mel_dir))
num_df_validation = len(os.listdir(validation_df_dir))
num_bkl_validation = len(os.listdir(validation_bkl_dir))
num_bcc_validation = len(os.listdir(validation_bcc_dir))

num_nv_test = len(os.listdir(test_nv_dir))
num_mel_test = len(os.listdir(test_mel_dir))
num_df_test = len(os.listdir(test_df_dir))
num_bkl_test = len(os.listdir(test_bkl_dir))
num_bcc_test = len(os.listdir(test_bcc_dir))

print('Total Training Images of nv:', num_nv_train)
print('Total Training Images of mel:', num_mel_train)
print('Total Training Images of df:', num_df_train)
print('Total Training Images of bkl:', num_bkl_train)
print('Total Training Images of bcc:', num_bcc_train)
print('\n************************\n')
print('Total Validation Images of nv:', num_nv_validation)
print('Total Validation Images of mel:', num_mel_validation)
print('Total Validation Images of df:', num_df_validation)
print('Total Validation Images of bkl:', num_bkl_validation)
print('Total Validation Images of bcc:', num_bcc_validation)
print('\n************************\n')
print('Total Testing Images of nv:', num_nv_test)
print('Total Testing Images of mel:', num_mel_test)
print('Total Testing Images of df:', num_df_test)
print('Total Testing Images of bkl:', num_bkl_test)
print('Total Testing Images of bcc:', num_bcc_test)

total_train = num_nv_train + num_mel_train + num_df_train + num_bkl_train + num_bcc_train
total_validation = num_nv_validation + num_mel_validation + num_df_validation + num_bkl_validation + num_bcc_validation
total_test = num_nv_test + num_mel_test + num_df_test + num_bkl_test + num_bcc_test

print('Total Training Images:', total_train)
print('Total Validation Images:', total_validation)
print('Total Testing Images:', total_test)

BATCH_SIZE = 100
IMG_SHAPE  = 150

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

image_gen_train = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                     zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE, directory=train_dir, shuffle=True, target_size=(IMG_SHAPE, IMG_SHAPE), class_mode='categorical')

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=BATCH_SIZE, directory=validation_dir, target_size=(IMG_SHAPE, IMG_SHAPE), class_mode='categorical')

image_gen_test = ImageDataGenerator(rescale=1./255)

test_data_gen = image_gen_test.flow_from_directory(batch_size=BATCH_SIZE, directory=test_dir, target_size=(IMG_SHAPE, IMG_SHAPE), class_mode='categorical')

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
    tf.keras.layers.Dense(5)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs=2
def infinite_generator(generator):
    while True:
        for batch in generator:
            yield batch

def preprocess_image(img):
    # Convertir a escala de grises
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Redimensionar la imagen
    img_gray = cv2.resize(img_gray, (150, 150))
    
    # Aplicar la transformada de Fourier
    dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Calcular la magnitud
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    
    # Crear un filtro pasa-bajas
    rows, cols = img_gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols, 2), np.uint8)
    r = 20  # Radio del filtro
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 1
    
    # Aplicar la máscara y la transformada inversa
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    # Aplicar filtro gaussiano
    img_back = cv2.GaussianBlur(img_back, (11, 11), 0)
    
    # Normalizar a rango 0-1 (como en ImageDataGenerator)
    img_back = img_back / 255.0
    return img_back

def custom_generator(generator):
    while True:
        batch = next(generator)
        images, labels = batch
        preprocessed_images = np.array([preprocess_image(img) for img in images])
        
        # Si el modelo espera entrada con 3 canales, duplicar la imagen en 3 canales
        preprocessed_images = np.stack([preprocessed_images] * 3, axis=-1)
        yield preprocessed_images, labels

# Crear los generadores personalizados
train_data_gen_custom = custom_generator(train_data_gen)
val_data_gen_custom = custom_generator(val_data_gen)
test_data_gen_custom = custom_generator(test_data_gen)


train_data_gen_inf = infinite_generator(train_data_gen_custom)
val_data_gen_inf = infinite_generator(val_data_gen_custom)

history = model.fit(
    train_data_gen_inf,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=epochs,
    validation_data=val_data_gen_inf,
    validation_steps=int(np.ceil(total_validation / float(BATCH_SIZE))),
    verbose= 1
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)
print(epochs_range)

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
