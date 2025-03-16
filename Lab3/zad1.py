import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris.csv")

(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=292499)

def classify_iris(sl,sw,pl,pw):
    if pl < 2 and pw < 1:
        return ("Setosa")
    elif sl >5 and pw > 1.5:
        return ("Virginica")
    else:
        return ("Versicolor")
    
good_predictions = 0
len = test_set.shape[0]



for i in range(len):
    if classify_iris(test_set[i][0],test_set[i][1],test_set[i][2],test_set[i][3]) == test_set[i][-1]:
        good_predictions += 1
    else:
        print(classify_iris(test_set[i][0],test_set[i][1],test_set[i][2],test_set[i][3]), test_set[i][-1])





train_df = pd.DataFrame(train_set, columns=df.columns)
sorted_train_df = train_df.sort_values(by='variety')
print("Sorted training data:")
print(sorted_train_df.values)
# print(train_set)

print(good_predictions)
print(good_predictions/len*100, "%")