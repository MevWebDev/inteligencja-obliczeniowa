import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Load the dataset
data = pd.read_csv("diabetes.csv")
randomstate = 292499
hidden_layer = (6,3)


# Assuming the target variable is in the last column
X = data.iloc[:, :-1].values  # All columns except last one (features)
y = data.iloc[:, -1].values   # Last column (target)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=randomstate
)

# Compare different activation functions
activations = ['identity', 'logistic', 'tanh', 'relu']
results = {}

for activation in activations:
    # Create and train model with this activation
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer, 
        max_iter=500, 
        random_state=randomstate,
        activation=activation
    )
    mlp.fit(X_train, y_train)
    
    # Test performance
    predictions = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    results[f"NN: {activation}"] = accuracy
    
    print(f"\nActivation: {activation}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print("""
|TN  FP|
|FN  TP|
""")
    print(confusion_matrix(y_test, predictions))

    

# Plot comparison of activations
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values())
plt.title('Accuracy by Activation Function')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig('activation_comparison.png')

dtc = DecisionTreeClassifier(random_state=292499)
dtc.fit(X_train, y_train)
dtc_pred = dtc.predict(X_test)
dtc_accuracy = accuracy_score(y_test, dtc_pred)
results["Decision Tree"] = dtc_accuracy
print("Decision Tree accuracy:", dtc_accuracy)

knf = KNeighborsClassifier(n_neighbors=11)
knf.fit(X_train, y_train)
knf_pred = knf.predict(X_test)
knf_accuracy = accuracy_score(y_test, knf_pred)
results["KNF-5"] = knf_accuracy
print("K-11 neighbours accuracy:", knf_accuracy)

naive = GaussianNB()
naive.fit(X_train, y_train)
naive_pred = naive.predict(X_test)
naive_accuracy = accuracy_score(y_test, naive_pred)
results["Naive Bayes"] = naive_accuracy
print("Naive bayes neighbours accuracy:", naive_accuracy)

results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values())
plt.title('Accuracy by Activation Function')
plt.ylabel('Accuracy')
plt.ylim(min(results.values()) - 0.05, max(results.values()) + 0.05)
plt.savefig('activation_comparison.png')


# Więcej jest false negative 
# czyli że ktoś ma diabetes a pokazało że nie ma
# i uważam że takie są gorsze ponieważ więcej zła można wyrządzić
# na niestwierdzeniu posiadanej choroby niz na stwierdzeniu 
# nieposiadanej
