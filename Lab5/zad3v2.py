import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

# Optimize TensorFlow performance
tf.config.optimizer.set_jit(True)  # Enable XLA compilation
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load and prepare data
data_dir = "dogs-cats-mini"
filenames = [f for f in os.listdir(
    data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
categories = ['dog' if f.split('.')[0] == 'dog' else 'cat' for f in filenames]

df = pd.DataFrame({'filename': filenames, 'category': categories}).sample(
    frac=1).reset_index(drop=True)

# Split data (80% train, 20% validation)
train_df, validate_df = train_test_split(df, test_size=0.2, random_state=42)

# Optimized image generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True  # Only most effective augmentation
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Larger batch size for better GPU utilization
batch_size = 64
target_size = (128, 128)  # Smaller images for faster processing

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=data_dir,
    x_col='filename',
    y_col='category',
    target_size=target_size,
    class_mode='binary',
    batch_size=batch_size
)

validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    directory=data_dir,
    x_col='filename',
    y_col='category',
    target_size=target_size,
    class_mode='binary',
    batch_size=batch_size,
    shuffle=False
)

# Simplified model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(*target_size, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Early stopping to prevent unnecessary epochs
early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=3, restore_best_weights=True)

history = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    callbacks=[
        ModelCheckpoint('best_model.h5', monitor='val_accuracy',
                        save_best_only=True),
        early_stopping
    ],
    verbose=1
)

# Visualization
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Curves')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Curves')

plt.savefig('training_curves.png', bbox_inches='tight')
plt.close()

# Evaluation
best_model = load_model('best_model.h5')
loss, accuracy = best_model.evaluate(validation_generator)
print(f"\nValidation Accuracy: {accuracy*100:.2f}%")

# Sample predictions
plt.figure(figsize=(15, 15))
sample_files = validate_df['filename'].sample(9).values

for i, filename in enumerate(sample_files):
    img_path = os.path.join(data_dir, filename)
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img)/255.0
    pred_prob = best_model.predict(np.expand_dims(img_array, axis=0))[0][0]
    pred = "DOG" if pred_prob > 0.5 else "CAT"
    confidence = max(pred_prob, 1-pred_prob)  # Get the confidence score

    plt.subplot(3, 3, i+1)
    plt.imshow(img)
    plt.title(f"Pred: {pred}\n({confidence:.2f})")
    plt.axis('off')

plt.savefig('sample_predictions.png', bbox_inches='tight')
plt.close()