import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import History

# Load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess data
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
original_test_labels = np.argmax(test_labels, axis=1) # Save original labels for confusion matrix

# Define model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = History()
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2,
          callbacks=[history])

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# Predict on test images
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Confusion matrix
cm = confusion_matrix(original_test_labels, predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plotting training and validation accuracy and loss
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.tight_layout()
plt.show()

# Display 25 images from the test set with their predicted labels
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i].reshape(28,28), cmap=plt.cm.binary)
    plt.xlabel(predicted_labels[i])
plt.show()

#1. reshape zmienia z tablicy 2x2 na tablice 4x4 
#   1 na końcu oznacza jeden kanał (skala szarości),
#   float32 zamienia piksele na liczby zmiennoprzecinkowe,
#   /255 normalizuje na zakres 0-1


#2. warstwa wejściowa ->
#  -> Conv2D wyłapuje wzorce, relu zamienia <0 na 0 i
#  zwraca mapy cech 
#  -> MaxPooling2D wybiera maksymalna wartość
#  z każdego okna 2x2 i redukuje danych o połowe
#  -> Flatten Przekształca 3D tensor w 1D wektor
#  -> Dense Łączy się ze wszystkimi wejściami i robi na nich ReLU
#  -> Warstwa wyjściowa zamienia wartości w prawdopodobieństwa
#  i zwraca prawdopodobienstwa dla kazdej cyfry

#3. najwięcej jest pomyleń 4 z 9 i 7 z 9

#4. krzywe wskazują na przeuczenie, ponieważ świetnie radzi sobie
#  z danymi treningowymi ale gorzej na nowych danych

#5. checkpoint = ModelCheckpoint(
#     filepath='mnist_model_epoch{epoch:02d}_val_acc{val_accuracy:.4f}.h5',  # Nazwa pliku z numerem epoki i dokładnością
#     monitor='val_accuracy',        # Metryka do monitorowania
#     save_best_only=True,           # Zapisuj tylko gdy jest poprawa
#     mode='max',                    # Maksymalizuj dokładność (dla straty użyj 'min')
#     verbose=1                      # Pokaż informacje o zapisie
# )


# model.fit(
#     train_images, 
#     train_labels, 
#     epochs=5, 
#     batch_size=64, 
#     validation_split=0.2,
#     callbacks=[history, checkpoint]  # Dodaj checkpoint do listy callbacków
# )

