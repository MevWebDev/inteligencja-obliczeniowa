from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

iris = load_iris()
randomstate = 292499

# Show how strings are converted to numbers
print("Original iris species names:", iris.target_names)
print("Corresponding numeric values:")
for i, name in enumerate(iris.target_names):
    print(f"{name}: {i}")

from sklearn.model_selection import train_test_split
datasets = train_test_split(iris.data, iris.target,
                            test_size=0.3, random_state=randomstate)

train_data, test_data, train_labels, test_labels = datasets



mlpd = MLPClassifier(hidden_layer_sizes=(2), max_iter=10000,random_state=randomstate)
mlpd.fit(train_data, train_labels)

mlpf = MLPClassifier(hidden_layer_sizes=(3), max_iter=10000,random_state=randomstate)
mlpf.fit(train_data, train_labels)

mlpg = MLPClassifier(hidden_layer_sizes=(3,3), max_iter=10000,random_state=randomstate)
mlpg.fit(train_data, train_labels)



print("Network with 2 hidden nodes")

predictions_test = mlpd.predict(test_data)
print(accuracy_score(predictions_test, test_labels))

print("Network with 3 hidden nodes")

predictions_test = mlpf.predict(test_data)
print(accuracy_score(predictions_test, test_labels))

print("Network with 3 + 3 hidden nodes")

predictions_test = mlpg.predict(test_data)
print(accuracy_score(predictions_test, test_labels))







