## Juan David Galan Vargas - 202111470
##Prueba para cluster HYPATIA
##Identificación de cúmulos abiertos distantes a partir de diagramas magnitud-color

# Importar
import tensorflow as tf
#import os
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K
from keras.layers import Input
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import csv
import pydot
#from tensorflow.keras.utils import plot_model
#import tensorflow.keras.backend as K
#from sklearn.metrics import f1_score
import matplotlib.colors as mcolors

# Direcciones: Necesarias de Ajustar al subir diagramas a HYPATIA

cumulos_folder_shell1=r"C:\Users\juang\Documents\Tareas JD\Universidad\Materias\8 Semestre\Proyecto Teorico\Proyecto Teorico 20242\Gaia-DR3-Proyecto-Teorico-20242\muestra_entrenamiento\shell1\cumulos"
nocumulos_folder_shell1=r"C:\Users\juang\Documents\Tareas JD\Universidad\Materias\8 Semestre\Proyecto Teorico\Proyecto Teorico 20242\Gaia-DR3-Proyecto-Teorico-20242\muestra_entrenamiento\shell1\no-cumulos"

cumulos_folder_shell2=r"C:\Users\juang\Documents\Tareas JD\Universidad\Materias\8 Semestre\Proyecto Teorico\Proyecto Teorico 20242\Gaia-DR3-Proyecto-Teorico-20242\muestra_entrenamiento\shell2\cumulos"
nocumulos_folder_shell2=r"C:\Users\juang\Documents\Tareas JD\Universidad\Materias\8 Semestre\Proyecto Teorico\Proyecto Teorico 20242\Gaia-DR3-Proyecto-Teorico-20242\muestra_entrenamiento\shell2\no-cumulos"

shell1=r"C:\Users\juang\Documents\Tareas JD\Universidad\Materias\8 Semestre\Proyecto Teorico\Proyecto Teorico 20242\Gaia-DR3-Proyecto-Teorico-20242\muestra_entrenamiento\shell1"
shell2=r"C:\Users\juang\Documents\Tareas JD\Universidad\Materias\8 Semestre\Proyecto Teorico\Proyecto Teorico 20242\Gaia-DR3-Proyecto-Teorico-20242\muestra_entrenamiento\shell2"

clasificar_shell1=r"C:\Users\juang\Documents\Tareas JD\Universidad\Materias\8 Semestre\Proyecto Teorico\Proyecto Teorico 20242\Gaia-DR3-Proyecto-Teorico-20242\data_v2_diagramas\shell1\prueba"
clasificar_shell2=r"C:\Users\juang\Documents\Tareas JD\Universidad\Materias\8 Semestre\Proyecto Teorico\Proyecto Teorico 20242\Gaia-DR3-Proyecto-Teorico-20242\data_v2_diagramas\shell2\prueba"

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    vertical_flip=False,
)
train_data_dir = shell1

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(253, 385),
    batch_size=32,
    class_mode='binary',
    classes=['no-cumulos', 'cumulos'],
    shuffle=True,
    #subset='training'  # Training set (80%)
)

def get_model():
    base_model = InceptionV3(input_shape = (253,385,3), weights='imagenet', include_top=False)
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    #x = keras.layers.Dense(512, activation='relu')(x)
    #x = keras.layers.BatchNormalization()(x)
    predictions = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False #Originalmente False, True en prueba
        
    for layer in base_model.layers[-150:]:
        layer.trainable = False
    return model
model = get_model()

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]
model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss='binary_crossentropy',
    metrics=METRICS)

# Define the log file path
log_file = r"C:\Users\juang\Documents\Tareas JD\Universidad\Materias\9 Semestre\TESIS\modelos\training_metrics.txt"

class MetricsLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.first_epoch = True  # Flag to track first epoch

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Open file in append mode
        with open(log_file, mode="a", newline="") as f:
            writer = csv.writer(f)

            # If first epoch, write headers
            if self.first_epoch:
                writer.writerow(["Epoch"] + list(logs.keys()))
                self.first_epoch = False  # Prevent rewriting headers

            # Write epoch number and metric values
            writer.writerow([epoch + 1] + [logs[key] for key in logs.keys()])

# Train the model and log metrics
model.fit(
    train_generator,
    epochs=15,
    callbacks=[MetricsLogger()]
)

#Ajustar dirección y nombre
modelo_path=r"C:\Users\juang\Documents\Tareas JD\Universidad\Materias\9 Semestre\TESIS\modelos\modelo1_prueba3.keras"
model.save(modelo_path,zipped=True)


