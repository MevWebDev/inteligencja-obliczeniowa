import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
df = pd.read_csv("iris.csv")

# Split the data into features and target
X = df.iloc[:, :-1].values  # Features (all columns except the last one)
y = df.iloc[:, -1].values   # Target (last column)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=292499)

# Dictionary to store classifiers and their results
classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=292499),
    "3NN": KNeighborsClassifier(n_neighbors=3),
    "5NN": KNeighborsClassifier(n_neighbors=5),
    "11NN": KNeighborsClassifier(n_neighbors=11),
    "Naive Bayes": GaussianNB()
}

# Dictionary to store results
results = {}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    # Train the classifier
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"\n{name} Classifier:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    
    plt.figure(figsize=(8, 6))
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix ({name})')
    plt.savefig(f'./Lab3/plots/confusion_matrix_{name.replace(" ", "_").lower()}.png')
    plt.close()

# Compare accuracies
plt.figure(figsize=(12, 6))
plt.bar(results.keys(), results.values())
plt.ylim([0, 1])
plt.ylabel('Accuracy')
plt.title('Classifier Comparison')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, (key, value) in enumerate(results.items()):
    plt.text(i, value + 0.02, f'{value:.4f}', ha='center')
plt.savefig('./Lab3/plots/classifier_comparison.png')
plt.close()
print("\nClassifier comparison plot saved as 'classifier_comparison.png'")

# Print summary of results
print("\nSummary of Classifier Accuracies:")
for name, accuracy in results.items():
    print(f"{name}: {accuracy:.4f}")

# Find the best classifier
best_classifier = max(results, key=results.get)
print(f"\nThe best classifier is: {best_classifier} with accuracy: {results[best_classifier]:.4f}")