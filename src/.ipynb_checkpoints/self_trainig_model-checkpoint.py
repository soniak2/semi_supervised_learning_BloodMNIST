import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import entropy
import gc

from src.callbacks import get_callbacks

class SelfTrainingModel:
    def __init__(self, model_NN, num_classes, initial_ratio = 0.05, threshold_strategy = 'f1', batch_size = 128, min_epochs=35, max_epochs=80, run_name = "_"):
        self.model_NN = model_NN
        self.num_classes = num_classes
        self.initial_ratio = initial_ratio
        self.threshold_strategy = threshold_strategy
        self.batch_size = batch_size
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.thresholds = None
        self.run_name = run_name

        np.random.seed(42)
        tf.random.set_seed(42)

    def split_subset_and_rest(self, X, y, subset_size):
        total_size = len(X)
        indices = np.random.permutation(total_size)
        subset_idx, rest_idx = indices[:subset_size], indices[subset_size:]

        return X[subset_idx], y[subset_idx], X[rest_idx], y[rest_idx]

    def initial_balanced_split(self, X, y, subset_size):
        y_classes = np.argmax(y, axis=1)
        samples_per_class = subset_size // self.num_classes
        selected_indices = []
    
        for cls in range(self.num_classes):
            cls_indices = np.where(y_classes == cls)[0]
            if len(cls_indices) < samples_per_class:
                raise ValueError(f"Not enough samples for class {cls} to create balanced subset.")
            np.random.shuffle(cls_indices)
            selected_indices.extend(cls_indices[:samples_per_class])
    
        selected_indices = np.array(selected_indices)
        rest_indices = np.setdiff1d(np.arange(len(X)), selected_indices)

        for cls in range(self.num_classes):
            selected_class_count = np.sum(y_classes[selected_indices] == cls)
            rest_class_count = np.sum(y_classes[rest_indices] == cls)
            print(f"After split, Class {cls}: {selected_class_count} in subset, {rest_class_count} in the rest.")
    
        return X[selected_indices], y[selected_indices], X[rest_indices], y[rest_indices]

    
    def find_best_thresholds_per_class(self, y_true, y_probs):
        thresholds = []
        for i in range(self.num_classes):
            y_true_binary = (y_true == i).astype(int)
            y_score = y_probs[:, i]
            precision, recall, thresh = precision_recall_curve(y_true_binary, y_score)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            max_f1_idx = np.argmax(f1)
            best_thresh = thresh[max_f1_idx] if max_f1_idx < len(thresh) else 1.0
            thresholds.append(best_thresh)
        #thresholds = np.clip(thresholds, 0.3, 0.8)
        print("Best thresholds ", thresholds)
        return thresholds
        

    def pseudo_label(self, model, X_unlabeled):
        y_probability = model.predict(X_unlabeled)
        X_confident, y_confident, X_unconfident = [], [], []
        #for i, probs in enumerate(y_probability[:10]):
        #    print(f"Sample {i}: {probs}")
        
        '''
        for i, probability in enumerate(y_probability):
            above_threshold = probability >= self.thresholds
            if np.any(above_threshold):
                pred_class = np.argmax(probability * above_threshold)
                X_confident.append(X_unlabeled[i])
                y_confident.append(tf.keras.utils.to_categorical(pred_class, num_classes=self.num_classes))
            else:
                X_unconfident.append(X_unlabeled[i])
        '''
        '''
        for i, probability in enumerate(y_probability):
            pred_class = np.argmax(probability)
            if probability[pred_class] >= self.thresholds[pred_class]:
                X_confident.append(X_unlabeled[i])
                y_confident.append(tf.keras.utils.to_categorical(pred_class, num_classes=self.num_classes))
            else:
                X_unconfident.append(X_unlabeled[i])
        '''

        for i, probs in enumerate(y_probability):
            ent = entropy(probs)
            if ent < 1.0:  # entropia niska -> wysoka pewność
                pred_class = np.argmax(probs)
                if probs[pred_class] >= self.thresholds[pred_class]:
                    X_confident.append(X_unlabeled[i])
                    y_confident.append(tf.keras.utils.to_categorical(pred_class, num_classes=self.num_classes))
                else:
                    X_unconfident.append(X_unlabeled[i])
            else:
                X_unconfident.append(X_unlabeled[i])

        return np.array(X_confident), np.array(y_confident), np.array(X_unconfident)

    def calculate_epochs(self, size):
        return int(self.min_epochs + (self.max_epochs - self.min_epochs) * (1 - size))

    def train(self, X_train, y_train, X_val, y_val):
        model = None
        subset_size = int(len(X_train) * self.initial_ratio)
        X_labeled, y_labeled, X_unlabeled, y_unlabeled = self.split_subset_and_rest(X_train, y_train, subset_size)
        #X_labeled, y_labeled, X_unlabeled, y_unlabeled = self.initial_balanced_split(X_train, y_train, subset_size)
        
        print("X_labeled shape :", X_labeled.shape)
        print("X_unlabeled shape :", X_unlabeled.shape)

        round_num = 1
        while len(X_unlabeled) > 100: # 150 # 200
            print(f"\n=== Iteration {round_num}: Training on {len(X_labeled)} labeled samples ===")

            y_labeled_classes = np.argmax(y_labeled, axis=1) 
            class_weights_array = compute_class_weight(class_weight='balanced',
                                                       classes=np.arange(self.num_classes),
                                                       y=y_labeled_classes).astype(np.float32)
            
            class_weights = dict(enumerate(class_weights_array))
            print("Class weights:", class_weights)

            callbacks = get_callbacks(folder_name = "fit", run_name = self.run_name)

            used_ratio = len(X_labeled) / len(X_train)
            no_of_epochs = self.calculate_epochs(used_ratio)
            print(no_of_epochs)

            
            
            if model is not None:
                del model
            tf.keras.backend.clear_session()
            gc.collect()
            
            model = self.model_NN()
            history = model.fit(X_labeled, y_labeled,
                                  epochs=no_of_epochs,
                                  batch_size=self.batch_size,
                                  validation_data=(X_val, y_val),
                                  callbacks=callbacks,
                                  class_weight=class_weights,
                                  verbose=False)

            val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
            print(f"=== Validation accuracy after iteration {round_num}: {val_accuracy:.4f} ===")

            y_val_labels = np.argmax(y_val, axis=1)
            y_probs = model.predict(X_val)
            self.thresholds = self.find_best_thresholds_per_class(y_val_labels, y_probs)

            subset_ratio = self.initial_ratio if self.initial_ratio > 0.20 else 0.2
            current_subset_size = min(len(X_unlabeled), int(len(X_train) * subset_ratio))
            X_subset, _, X_unlabeled, _ = self.split_subset_and_rest(X_unlabeled, y_unlabeled, current_subset_size)
 
            X_confident, y_confident, X_remaining = self.pseudo_label(model, X_subset)
            print(f"=== Added {len(X_confident)} confident samples, {len(X_subset) - len(X_confident)} remain unlabeled. ===")

            if len(X_confident) > 0:
                X_labeled = np.concatenate([X_labeled, X_confident], axis=0)
                y_labeled = np.concatenate([y_labeled, y_confident], axis=0)
            if len(X_remaining) > 0:
                X_unlabeled = np.concatenate([X_unlabeled, X_remaining], axis=0)

            round_num += 1

        print("Training completed")
        return model