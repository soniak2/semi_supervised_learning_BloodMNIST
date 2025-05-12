import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau


class AccuracyByDataSize:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, model_class, min_epochs=35, max_epochs=100, repetitions=5):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.model_class = model_class
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.repetitions = repetitions

    def get_random_subset(self, dataset, size):
        X, y = dataset
        indices = np.random.choice(X.shape[0], size=int(size * X.shape[0]), replace=False)
        X_subset = X[indices]
        y_subset = y[indices]
        return X_subset, y_subset

    def train_model_on_random_subset(self, size):
        X_train_subset, y_train_subset = self.get_random_subset((self.X_train, self.y_train), size)
        X_val_subset, y_val_subset = self.get_random_subset((self.X_val, self.y_val), size)
        X_test_subset, y_test_subset = self.get_random_subset((self.X_test, self.y_test), size)

        print("Subset shape", X_train_subset.shape, y_train_subset.shape)

        reduce_lr = ReduceLROnPlateau(
                                        monitor='val_loss', 
                                        factor=0.1, 
                                        patience=5, 
                                        min_lr=1e-6,
                                        verbose = False
                                        )


        model = self.model_class()
        history = model.fit(X_train_subset, y_train_subset, 
                            epochs=self.calculate_epochs(size),
                            batch_size=128, 
                            validation_data=(self.X_val, self.y_val), # X_val_subset, y_val_subset
                            callbacks= [reduce_lr],
                            verbose=False)
        
        print("Number of epochs :", self.calculate_epochs(size))
        
        test_loss, test_acc = model.evaluate(X_test_subset, y_test_subset)
        print(f"Test accuracy: {test_acc:.4f}")
        return test_acc

    def calculate_epochs(self, size):
        return int(self.min_epochs + (self.max_epochs - self.min_epochs) * (1 - size))

    def calculate_accuracy_by_size(self, sizes):
        means = []
        stds = []
        for size in sizes:
            print(f"=== Training on {int(size * 100)}% of data ===")
            accuracies_for_size = []
            for _ in range(self.repetitions):
                accuracy = self.train_model_on_random_subset(size)
                accuracies_for_size.append(accuracy)
            mean_accuracy = np.mean(accuracies_for_size)
            std_accuracy = np.std(accuracies_for_size)
            means.append(mean_accuracy)
            stds.append(std_accuracy)
        return means, stds


    def plot_accuracy_vs_data_size(self, sizes):
        means, stds = self.calculate_accuracy_by_size(sizes)
        percentages = [int(size * 100) for size in sizes]

        
        plt.figure(figsize=(6, 4))
        plt.errorbar(percentages, means, yerr=stds, fmt='-o', capsize=5, elinewidth=2)

        for x, y in zip(percentages, means):
            plt.text(x + 5, y + 0.01, f"{y:.3f}", ha='center', fontsize=10, color='black')
            
        #plt.title("Accuracy vs Data Size", fontsize=18)
        plt.xlabel("Percentage of data used [%]", fontsize=18)
        plt.ylabel("Accuracy - test", fontsize=18)
        plt.ylim((0, 1.0))
        plt.xlim((0, 105))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('accuracy_vs_data_size.png')
        print('Plot has been saved as accuracy_vs_data_size.png')
        plt.show()

        print("means: ", means)
        print("stds: ", stds)

        return means, stds
        