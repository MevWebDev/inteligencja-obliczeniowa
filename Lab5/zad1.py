import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

# Load the iris dataset
iris = load_iris()
x = iris.data
y = iris.target

# Preprocess the data
# Scale the features
scaler = StandardScaler() #Normaliuzje dane poprzez Z-score z = (x - μ) / σ
X_scaled = scaler.fit_transform(x)

# Encode the labels
encoder = OneHotEncoder(sparse_output=False) #zamienia nazwy na liczby np
#[[1, 0, 0],  # Class 0 (Setosa)
#[0, 1, 0],  # Class 1 (Versicolor)
# [0, 0, 1],  # Class 2 (Virginica)
# [0, 1, 0],  # Class 1 (Versicolor)
# [1, 0, 0],  # Class 0 (Setosa)
#...]
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Load the pre-trained model
#tanh lepiej
#selu najlepiej

model = Sequential([
    Dense(64,activation='selu', input_shape=(X_train.shape[1],)),
    Dense(64,activation='selu'),
    Dense(y_encoded.shape[1], activation='softmax') #ma 3 neurony każdy do rodzaju irysa
    # i są w kształtach jak OneHotEncoder
])

# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
#ten compiler dał najwieksze wyniki

# Continue training the model for 10 more epochs
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2,)
#im mniejszy batch size tym te krzywe są bardizej rozjechane góra dół

# Save the updated model
model.save('updated_iris_model.h5')

# Evaluate the updated model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Plot the learning curve
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model


# Plot and save the model architecture
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#a. Normaliuzje dane poprzez Z-score z = (x - μ) / σ
#b. zamienia nazwy na liczby np
    #[[1, 0, 0],  # Class 0 (Setosa)
    #[0, 1, 0],  # Class 1 (Versicolor)
    # [0, 0, 1],  # Class 2 (Virginica)
    # [0, 1, 0],  # Class 1 (Versicolor)
    # [1, 0, 0],  # Class 0 (Setosa)
    #...]
#c. ma 3 neurony wyjściowe każdy do rodzaju irysa
    # i są w kształtach jak OneHotEncoder

#d. #tanh lepiej, selu najlepiej

#e. najlepiej poradził sobie optimize rmsprop

#f. im mniejszy batch size tym większy rozstrzał wykresu, im wiekszy batch tym szybciej

#g można powiedzieć, że model radzi sobie świetnie bo często daje 100% accuracy

