# ==========================================
# DEEP LEARNING PROJECT
# IMAGE CLASSIFICATION USING TENSORFLOW
# ==========================================

# IMPORT LIBRARIES

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# LOAD DATASET
# ==========================================

print("Loading CIFAR-10 dataset...")

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

print("Dataset Loaded Successfully!")

# Normalize images
train_images = train_images / 255.0
test_images = test_images / 255.0


# ==========================================
# CLASS NAMES
# ==========================================

class_names = [
    'Airplane',
    'Automobile',
    'Bird',
    'Cat',
    'Deer',
    'Dog',
    'Frog',
    'Horse',
    'Ship',
    'Truck'
]


# ==========================================
# VISUALIZE SAMPLE IMAGES
# ==========================================

print("Displaying sample images...")

plt.figure(figsize=(10,10))

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.xticks([])
    plt.yticks([])

    plt.imshow(train_images[i])

    plt.xlabel(class_names[train_labels[i][0]])

plt.show()


# ==========================================
# BUILD CNN MODEL
# ==========================================

print("Building CNN model...")

model = models.Sequential()

# Convolution Layer 1
model.add(layers.Conv2D(
    32,
    (3,3),
    activation='relu',
    input_shape=(32,32,3)
))

# Pooling Layer
model.add(layers.MaxPooling2D((2,2)))

# Convolution Layer 2
model.add(layers.Conv2D(
    64,
    (3,3),
    activation='relu'
))

# Pooling Layer
model.add(layers.MaxPooling2D((2,2)))

# Convolution Layer 3
model.add(layers.Conv2D(
    64,
    (3,3),
    activation='relu'
))

# Flatten Layer
model.add(layers.Flatten())

# Dense Layer
model.add(layers.Dense(
    64,
    activation='relu'
))

# Output Layer
model.add(layers.Dense(10))

print("CNN Model Built Successfully!")


# ==========================================
# MODEL SUMMARY
# ==========================================

print("\nModel Summary:")
model.summary()


# ==========================================
# COMPILE MODEL
# ==========================================

print("\nCompiling model...")

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ),
    metrics=['accuracy']
)

print("Model Compiled Successfully!")


# ==========================================
# TRAIN MODEL
# ==========================================

print("\nTraining model...")

history = model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(test_images, test_labels)
)

print("Model Training Completed!")


# ==========================================
# EVALUATE MODEL
# ==========================================

print("\nEvaluating model...")

test_loss, test_acc = model.evaluate(
    test_images,
    test_labels,
    verbose=2
)

print("\nTest Accuracy:", test_acc)


# ==========================================
# VISUALIZE ACCURACY GRAPH
# ==========================================

print("\nDisplaying Accuracy Graph...")

plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(['Training Accuracy', 'Validation Accuracy'])

plt.show()


# ==========================================
# MAKE PREDICTIONS
# ==========================================

print("\nMaking Predictions...")

predictions = model.predict(test_images)

# Show one prediction
index = 5

predicted_label = np.argmax(predictions[index])

print("\nPredicted Class:")
print(class_names[predicted_label])

print("\nActual Class:")
print(class_names[test_labels[index][0]])


# ==========================================
# DISPLAY PREDICTED IMAGE
# ==========================================

plt.figure(figsize=(5,5))

plt.imshow(test_images[index])

plt.title(
    f"Predicted: {class_names[predicted_label]}"
)

plt.show()


# ==========================================
# SAVE MODEL
# ==========================================

print("\nSaving model...")

model.save("model/image_classifier_model.h5")

print("Model Saved Successfully!")


# ==========================================
# PROJECT COMPLETED
# ==========================================

print("\nDEEP LEARNING PROJECT COMPLETED SUCCESSFULLY!")
