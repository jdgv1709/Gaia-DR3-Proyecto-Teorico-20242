# Importar
import tensorflow as tf
import os
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
from sklearn.model_selection import train_test_split
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
import sys



shell1=r"/hpcfs/home/fisica/j.galanv/muestra entrenamiento/shell1/"
shell2=r"/hpcfs/home/fisica/j.galanv/muestra entrenamiento/shell2/"


# Data generators
train_data_dir=shell2
# Step 1: Gather image filepaths and labels
filepaths = []
labels = []

for class_name in ['no', 'si']:
    class_dir = os.path.join(train_data_dir, class_name)
    for fname in os.listdir(class_dir):
        filepaths.append(os.path.join(class_dir, fname))
        labels.append(class_name)

df = pd.DataFrame({'filename': filepaths, 'class': labels})

# Step 2: Custom class-imbalanced validation split
# Desired validation fraction and class proportions
validation_fraction = 0.15  # 15% of the full dataset
validation_no_ratio = 0.8
validation_si_ratio = 0.2

total_val_samples = int(len(df) * validation_fraction)
val_no_count = int(total_val_samples * validation_no_ratio)
val_si_count = total_val_samples - val_no_count  # to ensure the total matches

df_no = df[df['class'] == 'no']
df_si = df[df['class'] == 'si']

val_no = df_no.sample(n=val_no_count, random_state=42)
val_si = df_si.sample(n=val_si_count, random_state=42)

val_df = pd.concat([val_no, val_si])
train_df = df.drop(val_df.index)


# Step 3: Use flow_from_dataframe
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    vertical_flip=False
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='class',
    target_size=(253, 385),
    class_mode='binary',
    batch_size=64,
    shuffle=True
)

validation_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col='filename',
    y_col='class',
    target_size=(253, 385),
    class_mode='binary',
    batch_size=64,
    shuffle=False  # don't shuffle validation
)


def get_model():
    capas=0
    base_model = InceptionV3(input_shape=(253, 385, 3), weights='imagenet', include_top=False)
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = keras.layers.Dense(2048, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    predictions = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
        capas+=1
    for layer in base_model.layers[-158:]:
        layer.trainable = True #Ajustar segun modelo
    print(capas)
    return model

model = get_model()
    

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall')]

model.compile(optimizer=keras.optimizers.Adam(9.521711513431419e-05), loss='binary_crossentropy', metrics=METRICS)

# Define the log file path
log_file = r"/hpcfs/home/fisica/j.galanv/modelos/metricas_modelo_final_1cond_2shell.txt"

def compute_f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

def compute_f1_macro(tp, tn, fp, fn):
    precision_pos = tp / (tp + fp)
    recall_pos = tp / (tp + fn)
    f1_pos = compute_f1(precision_pos, recall_pos)

    precision_neg = tn / (tn + fn)
    recall_neg = tn / (tn + fp)
    f1_neg = compute_f1(precision_neg, recall_neg)

    return (f1_pos + f1_neg) / 2
    
def compute_p4(tp, tn, fp, fn):
    numerator= 4* tp* tn
    denominator= 4* tp* tn + (tp+tn) * (fp*fn)
    return numerator/denominator

def compute_MCC(tp,tn,fp,fn):

    numerator = tp * tn - fp * fn
    denominator = tf.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / denominator


# Callback
class F1DerivativeConvergence(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.p4_scores=[]
        self.MCC_scores=[]
        self.f1_scores = []
        self.f1_macro_scores = []
        self.first_epoch = True
        self.convergence_reason = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        precision = logs.get('precision', 0)
        recall = logs.get('recall', 0)
        tp = logs.get('tp', 0)
        tn = logs.get('tn', 0)
        fp = logs.get('fp', 0)
        fn = logs.get('fn', 0)
        MCC= compute_MCC(tp,tn,fp,fn)
        p4 = compute_p4(tp,tn,fp,fn)
        f1 = compute_f1(precision, recall)
        f1_macro = compute_f1_macro(tp, tn, fp, fn)
        self.MCC_scores.append(MCC.numpy)
        self.p4_scores.append(p4)
        self.f1_scores.append(f1)
        self.f1_macro_scores.append(f1_macro)

        stop = False
        f1_deriv = f1_range = f1_macro_deriv = f1_macro_range = None
        

        # Condition 1: F1 stability by differences
        if len(self.f1_scores) >= 5:
            diffs = [abs(self.f1_scores[i] - self.f1_scores[i - 1]) for i in range(-1, -5, -1)]
            if all(diff < 5.820350451570645e-05 for diff in diffs):
                self.convergence_reason.append( "F1 stability (<0.001 over 5 epochs)")
                stop=True

        # Condition 2: F1-macro stability by differences
        if not stop and len(self.f1_macro_scores) >= 5:
            macro_diffs = [abs(self.f1_macro_scores[i] - self.f1_macro_scores[i - 1]) for i in range(-1, -5, -1)]
            if all(diff < 1.7500748127406587e-05 for diff in macro_diffs):
                self.convergence_reason.append("F1 Macro stability (<0.001 over 5 epochs)")
                stop=True


        # Condition 3: F1 high-order finite difference derivative
        if not stop and len(self.f1_scores) >= 6:
            f1_vals = self.f1_scores[-6:]  # Get F1(i-5) to F1(i)
            f1_deriv = (- (1/5) * f1_vals[0] +(5/4) * f1_vals[1] - (10/3) * f1_vals[2] + 5 * f1_vals[3] - 5 * f1_vals[4] +(137/60) * f1_vals[5] )
            if abs(f1_deriv) < 2.663145690996998e-06:
                self.convergence_reason.append("F1 high-order finite derivative (<0.000005)")
                stop=True



        # Condition 4: F1-macro finite difference derivative

        if not stop and len(self.f1_macro_scores) >= 6:
            f1_macro_vals = self.f1_macro_scores[-6:]  # Get F1(i-5) to F1(i)
            f1_macro_deriv = (- (1/5) * f1_macro_vals[0] +(5/4) * f1_macro_vals[1] - (10/3) * f1_macro_vals[2] + 5 * f1_macro_vals[3] - 5 * f1_macro_vals[4] +(137/60) * f1_macro_vals[5] )
            if abs(f1_macro_deriv) < 9.123428311346003e-05:
                self.convergence_reason.append("F1 Macro high-order finite derivative (<0.000005)")
                stop=True


        # Logging
        with open(log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            if self.first_epoch:
                writer.writerow([
                    "Epoch", *logs.keys(),"P4","MCC",
                    "F1_score", "F1_macro"
                ])
                self.first_epoch = False
            writer.writerow([
                epoch + 1,
                *[logs.get(k, "") for k in logs.keys()],
                p4,
                MCC.numpy,
                f1,
                f1_macro
            ])

        if stop:
            with open(log_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Convergence achieved:", self.convergence_reason])
            self.model.stop_training = True


# Train
model.fit(train_generator, epochs=800, validation_data=validation_generator, callbacks=[F1DerivativeConvergence()]
)

# Save the model
modelo_path = r"/hpcfs/home/fisica/j.galanv/modelos/modelo_final_1cond_2shell.keras"
model.save(modelo_path, zipped=True)