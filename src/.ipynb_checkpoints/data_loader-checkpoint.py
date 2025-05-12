import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from medmnist import BloodMNIST

class BloodMNISTDataLoader:
    def __init__(self):
        self.train_dataset = BloodMNIST(split="train", download=True, transform=None)
        self.val_dataset = BloodMNIST(split="val", download=True, transform=None)
        self.test_dataset = BloodMNIST(split="test", download=True, transform=None)

        self.X_train, self.y_train = self._prepare_data(self.train_dataset)
        self.X_val, self.y_val = self._prepare_data(self.val_dataset)
        self.X_test, self.y_test = self._prepare_data(self.test_dataset)

        print(f"X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")
        print(f"X_val shape: {self.X_val.shape}, y_val shape: {self.y_val.shape}")
        print(f"X_test shape: {self.X_test.shape}, y_test shape: {self.y_test.shape}")

    def _prepare_data(self, dataset):
        X = dataset.imgs.astype("float32") / 255.0
        X = X.reshape(-1, 28, 28, 3)
        y = dataset.labels.squeeze()
        y = to_categorical(y, num_classes=8)
        return X, y

    def show_sample_images(self, n = 10, dataset = "test"):
        if dataset == "train":
            X, y = self.X_train, self.y_train
        elif dataset == "val":
            X, y = self.X_val, self.y_val
        else:
            X, y = self.X_test, self.y_test

        plt.figure(figsize=(n, 2))
        for i in range(n):
            plt.subplot(1, n, i + 1)
            plt.imshow(X[i])
            label = np.argmax(y[i])
            plt.title(f"Label: {label}")
            plt.axis("off")
        plt.tight_layout()
        plt.savefig('sample_images.png')
        plt.show()

    def plot_label_histograms(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, (name, y) in enumerate([
            ("Training set", self.y_train),
            ("Validation set", self.y_val),
            ("Test set", self.y_test)
        ]):
            labels = np.argmax(y, axis=1)
            unique, counts = np.unique(labels, return_counts=True)
            axes[idx].bar(unique, counts, color=['blue', 'green', 'red'][idx])
            axes[idx].set_title(name)
            axes[idx].set_xlabel("Class")
            axes[idx].set_ylabel("Count")

        plt.tight_layout()
        plt.savefig('label_histogram.png')
        plt.show()