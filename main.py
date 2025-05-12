import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import layers, models, optimizers, callbacks, Model, Input
from tensorflow.keras.utils import to_categorical

from src.data_loader import BloodMNISTDataLoader
from src.NN_model import ResNet18
from src.accuracy_by_data_size import AccuracyByDataSize
from src.self_trainig_model import SelfTrainingModel
import random
import os

loader = BloodMNISTDataLoader()
loader.show_sample_images(n=10, dataset="test")
loader.plot_label_histograms()
plt.show()

print('ok')
# Dane dostępne jako np.
X_train, y_train = loader.X_train, loader.y_train
X_val, y_val = loader.X_val, loader.y_val
X_test, y_test = loader.X_test, loader.y_test

'''
accuracy_model = AccuracyByDataSize(X_train, y_train, X_val, y_val, X_test, y_test, ResNet18, min_epochs=35, max_epochs=80, repetitions=5)
sizes = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
#sizes = [0.05]
means, stds = accuracy_model.plot_accuracy_vs_data_size(sizes)
'''


trainer = SelfTrainingModel(model_NN=ResNet18, num_classes=8, initial_ratio=0.1, min_epochs=35, max_epochs=80)
final_model = trainer.train(X_train, y_train, X_val, y_val)

y_probs = final_model.predict(X_test)
y_pred = np.argmax(y_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
