import pandas as pd
df = pd.read_csv("./Lab2/iris.csv")
print(df)

#wszystkie wiersze, kolumna nr0
print(df.values[:,0])

#wiersze od 5 do 10, wszystkie kolumny
print(df.values[5:11, :])

#dane w komórce [1,4]
print(df.values[1,4])

