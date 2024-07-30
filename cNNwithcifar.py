import tensorflow as tf
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix

class CNN:
    def __init__(self, x_train, x_test, y_train, y_test, epochs):
        self.x_train = x_train / 255.0  # Normalizing
        self.x_test = x_test / 255.0    # Normalizing
        self.y_train = y_train
        self.y_test = y_test
        self.epochs = epochs
        self.NUM_CATEGORIES = len(np.unique(y_test))
    
    def get_model(self):
        # vgg modelling but input shape is (32,32,3)
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding='same', input_shape=(32, 32, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation="relu"), 
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.NUM_CATEGORIES, activation="softmax")
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                      loss="sparse_categorical_crossentropy", 
                      metrics=["accuracy"])

        return model
    
    def modelling(self):
        model = self.get_model()
        model.summary()

        # data augmentation
        data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True,
            rotation_range=10, zoom_range=0.1)

        train_generator = data_generator.flow(self.x_train, self.y_train, batch_size=32)
        validation_generator = tf.keras.preprocessing.image.ImageDataGenerator().flow(self.x_test, self.y_test, batch_size=32)

        steps_per_epoch = self.x_train.shape[0] // 32
        validation_steps = self.x_test.shape[0] // 32

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        hist = model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=self.epochs, 
                         validation_data=validation_generator, validation_steps=validation_steps,
                         callbacks=[early_stopping])

        print("Train score:", model.evaluate(self.x_train, self.y_train))
        print("Test score:", model.evaluate(self.x_test, self.y_test))
        p = model.predict(self.x_test)
        cm = confusion_matrix(self.y_test, p.argmax(axis=1))

        self.plotGraphics(hist)
        self.plotConfusion(cm, list(range(10)))
        self.plot_misclassified_examples(p.argmax(axis=1), self.y_test, self.x_test)
        model.save("cifar10models.h5")  

    def plotGraphics(self, history):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plotConfusion(self, cm, classes):
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xticks(np.arange(len(classes)), classes, rotation=45)
        plt.yticks(np.arange(len(classes)), classes)
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def plot_misclassified_examples(self, p_test, y_test, x_test):
        misclassified_idx = np.where(p_test != y_test)[0]
        for i in range(len(misclassified_idx)):
            plt.figure()
            plt.imshow(x_test[misclassified_idx[i]])
            plt.title(f"True label: {y_test[misclassified_idx[i]]} Predicted: {p_test[misclassified_idx[i]]}")
            plt.show()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = y_train.flatten()
y_test = y_test.flatten()

obj = CNN(x_train, x_test, y_train, y_test, epochs=50)
obj.modelling()
