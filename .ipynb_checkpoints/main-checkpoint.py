import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import layers, models, optimizers, callbacks, Model, Input
from tensorflow.keras.utils import to_categorical
import random
import os

from sklearn.metrics import accuracy_score
import tensorflow as tf
import datetime

tf.config.optimizer.set_jit(False)

from src.data_loader import BloodMNISTDataLoader
from src.NN_model import ResNet18
from src.accuracy_by_data_size import AccuracyByDataSize
from src.self_trainig_model import SelfTrainingModel


# === GPU MEMORY CONTROL ==
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# === Download data === #
loader = BloodMNISTDataLoader()
loader.show_sample_images(n=10, dataset="test")
loader.plot_label_histograms()
plt.show()

print('ok')
X_train, y_train = loader.X_train, loader.y_train
X_val, y_val = loader.X_val, loader.y_val
X_test, y_test = loader.X_test, loader.y_test

# === Plot accuracy vs data size === #
'''
run_name = "_test"
accuracy_model = AccuracyByDataSize(X_train, y_train, X_val, y_val, X_test, y_test, ResNet18, min_epochs=35, max_epochs=80, repetitions=1, run_name = run_name) # CHANGE RUN_NAME !!! #
#sizes = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
sizes = [0.1, 0.25]
means, stds = accuracy_model.plot_accuracy_vs_data_size(sizes)

with open(f"logs/accuracy_results/{run_name}.txt", "w") as f:
    f.write("size\tmean_accuracy\tstd_accuracy\n")  
    for size, mean, std in zip(sizes, means, stds):
        f.write(f"{size:.2f}\t{mean:.3f}\t{std:.3f}\n")
'''


# === Semi-supervised leraning ==== #
ratio = 0.50
run_name = "test"
trainer = SelfTrainingModel(model_NN=lambda: ResNet18(input_shape=(28, 28, 3), num_classes=8),
                            num_classes=8, 
                            initial_ratio=ratio, 
                            min_epochs=35, 
                            max_epochs=80, 
                            run_name = run_name # CHANGE RUN_NAME !!! #
                           ) 
final_model = trainer.train(X_train, y_train, X_val, y_val)

y_probs = final_model.predict(X_test)
y_pred = np.argmax(y_probs, axis=1)
y_true = np.argmax(y_test, axis=1)


report = classification_report(y_true, y_pred, digits=4)
with open(f"logs/classification_report/cr_{run_name}.txt", "w") as f:
    f.write(report)

conf_matrix = confusion_matrix(y_true, y_pred)
with open(f"logs/confusion_matrix/cm_{run_name}.txt", "w") as f:
    f.write(np.array2string(conf_matrix))

log_dir = f"logs/final_metrics/{run_name}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(log_dir)
with writer.as_default():
    tf.summary.scalar("Test Accuracy", accuracy_score(y_true, y_pred), step=1)
    tf.summary.scalar("Train Ratio", ratio, step=1)

print(classification_report(y_true, y_pred, digits = 4))
print(confusion_matrix(y_true, y_pred))
