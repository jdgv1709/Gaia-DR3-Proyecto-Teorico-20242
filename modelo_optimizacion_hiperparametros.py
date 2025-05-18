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
import optuna


# Paths
shell1 = "/hpcfs/home/fisica/j.galanv/muestra entrenamiento/shell1/"
shell2 = "/hpcfs/home/fisica/j.galanv/muestra entrenamiento/shell2/"
log_file = "/hpcfs/home/fisica/j.galanv/modelos/metrics_optuna_1cond_intento2.csv"
best_model_path = "/hpcfs/home/fisica/j.galanv/modelos/optuna_1cond_intento2.keras"

# Prepare dataframe
filepaths = []
labels = []
for class_name in ['no', 'si']:
    class_dir = os.path.join(shell2, class_name)
    for fname in os.listdir(class_dir):
        filepaths.append(os.path.join(class_dir, fname))
        labels.append(class_name)
df = pd.DataFrame({'filename': filepaths, 'class': labels})

validation_fraction = 0.15
validation_no_ratio = 0.8
validation_si_ratio = 0.2
total_val_samples = int(len(df) * validation_fraction)
val_no_count = int(total_val_samples * validation_no_ratio)
val_si_count = total_val_samples - val_no_count
df_no = df[df['class'] == 'no']
df_si = df[df['class'] == 'si']
val_no = df_no.sample(n=val_no_count, random_state=42)
val_si = df_si.sample(n=val_si_count, random_state=42)
val_df = pd.concat([val_no, val_si])
train_df = df.drop(val_df.index)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Metrics helpers
def compute_f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall + 1e-6)

def compute_f1_macro(tp, tn, fp, fn):
    precision_pos = tp / (tp + fp + 1e-6)
    recall_pos = tp / (tp + fn + 1e-6)
    f1_pos = compute_f1(precision_pos, recall_pos)
    precision_neg = tn / (tn + fn + 1e-6)
    recall_neg = tn / (tn + fp + 1e-6)
    f1_neg = compute_f1(precision_neg, recall_neg)
    return (f1_pos + f1_neg) / 2

def compute_p4(tp, tn, fp, fn):
    numerator = 4 * tp * tn
    denominator = 4 * tp * tn + (tp + tn) * (fp * fn + 1e-6)
    return numerator / denominator

def compute_MCC(tp, tn, fp, fn):
    numerator = tp * tn - fp * fn
    denominator = tf.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-6)
    return numerator / denominator

# Convergence callback
class F1ConvergenceCallback(tf.keras.callbacks.Callback):
    def __init__(self, f1_diff_thresh, f1_macro_diff_thresh, f1_deriv_thresh, f1_macro_deriv_thresh):
        super().__init__()
        self.f1_scores, self.f1_macro_scores, self.p4_scores, self.MCC_scores = [], [], [], []
        self.f1_diff_thresh = f1_diff_thresh
        self.f1_macro_diff_thresh = f1_macro_diff_thresh
        self.f1_deriv_thresh = f1_deriv_thresh
        self.f1_macro_deriv_thresh = f1_macro_deriv_thresh

    def on_epoch_end(self, epoch, logs=None):
        tp = logs.get('tp', 0)
        tn = logs.get('tn', 0)
        fp = logs.get('fp', 0)
        fn = logs.get('fn', 0)
        precision = logs.get('precision', 0)
        recall = logs.get('recall', 0)

        f1 = compute_f1(precision, recall)
        f1_macro = compute_f1_macro(tp, tn, fp, fn)
        self.f1_scores.append(f1)
        self.f1_macro_scores.append(f1_macro)

        stop = False
        if len(self.f1_scores) >= 5:
            diffs = [abs(self.f1_scores[i] - self.f1_scores[i - 1]) for i in range(-1, -5, -1)]
            if all(diff < self.f1_diff_thresh for diff in diffs):
                stop = True

        if not stop and len(self.f1_macro_scores) >= 5:
            diffs = [abs(self.f1_macro_scores[i] - self.f1_macro_scores[i - 1]) for i in range(-1, -5, -1)]
            if all(diff < self.f1_macro_diff_thresh for diff in diffs):
                stop = True

        if not stop and len(self.f1_scores) >= 6:
            vals = self.f1_scores[-6:]
            deriv = (- (1/5)*vals[0] + (5/4)*vals[1] - (10/3)*vals[2] + 5*vals[3] - 5*vals[4] + (137/60)*vals[5])
            if abs(deriv) < self.f1_deriv_thresh:
                stop = True

        if not stop and len(self.f1_macro_scores) >= 6:
            vals = self.f1_macro_scores[-6:]
            deriv = (- (1/5)*vals[0] + (5/4)*vals[1] - (10/3)*vals[2] + 5*vals[3] - 5*vals[4] + (137/60)*vals[5])
            if abs(deriv) < self.f1_macro_deriv_thresh:
                stop = True

        if stop:
            self.model.stop_training = True

# Optuna objective
best_f1 = 0
best_params = None
best_epoch = 0

def objective(trial):
    global best_f1, best_params, best_epoch

    # Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dense_units = trial.suggest_int("dense_units", 256, 2048, step=128)
    trainable_layers = trial.suggest_int("trainable_layers", 50, 200)
    f1_diff_thresh = trial.suggest_float("f1_diff_thresh", 0.00001, 0.001)
    f1_macro_diff_thresh = trial.suggest_float("f1_macro_diff_thresh", 0.00001, 0.001)
    f1_deriv_thresh = trial.suggest_float("f1_deriv_thresh", 0.000001, 0.0001)
    f1_macro_deriv_thresh = trial.suggest_float("f1_macro_deriv_thresh", 0.000001, 0.0001)

    # Generators
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
    train_generator = train_datagen.flow_from_dataframe(
        train_df, x_col='filename', y_col='class',
        target_size=(253, 385), class_mode='binary',
        batch_size=batch_size, shuffle=True
    )
    val_generator = val_datagen.flow_from_dataframe(
        val_df, x_col='filename', y_col='class',
        target_size=(253, 385), class_mode='binary',
        batch_size=batch_size, shuffle=False
    )

    # Model
    base_model = InceptionV3(input_shape=(253, 385, 3), weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[-trainable_layers:]:
        layer.trainable = True
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(dense_units, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)

    METRICS = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy', metrics=METRICS)

    callback = F1ConvergenceCallback(f1_diff_thresh, f1_macro_diff_thresh, f1_deriv_thresh, f1_macro_deriv_thresh)
    history = model.fit(train_generator, validation_data=val_generator, epochs=800, callbacks=[callback], verbose=0)

    # Compute best F1
    best = 0
    for i, logs in enumerate(history.history['val_precision']):
        precision = history.history['val_precision'][i]
        recall = history.history['val_recall'][i]
        f1 = compute_f1(precision, recall)
        if f1 > best:
            best = f1
            best_epoch = i + 1

    if best > best_f1:
        best_f1 = best
        best_params = trial.params
        model.save(best_model_path)

    # Log to CSV
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([trial.number, best, best_epoch] + list(trial.params.values()))

    return best

# Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=35)

# Save parameter importance
importances = optuna.importance.get_param_importances(study)
print("Parameter importance:", importances)

print("Best F1 Score:", best_f1)
print("Best Params:", best_params)
print("Best Epoch:", best_epoch)
