import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("iris.csv")

# Dzielimy dane na features i targets (czyli wartosci i co ma wyjsc)
X = df.iloc[:,:-1].values #wszystkie kolumny oprocz ostatniej
y = df.iloc[:,-1].values #tylko ostatnie

#Dziele dane na testowe i treningowe
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=292499)

#Tworze drzewi decyzyjne
clf = DecisionTreeClassifier(random_state=292295)

#Ucze drzewo na treningowych danych
clf.fit(X_train, y_train)

#Sprawdzam na testowych danych
y_pred = clf.predict(X_test)

#Sprawdzam dokładność
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#Print feature importance
feature_names = df.columns[:-1]
print("\nFeature importance:")
for name, importance in zip(feature_names, clf.feature_importances_):
    print(f"{name}: {importance:.4f}")

# Visualize the decision tree (if the tree is not too large)
plt.figure(figsize=(15, 10))
tree.plot_tree(clf, 
               feature_names=feature_names,
               class_names=np.unique(y),
               filled=True)
plt.savefig('decision_tree.png', dpi=300)
print("\nDecision tree visualization saved as 'decision_tree.png'")

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

