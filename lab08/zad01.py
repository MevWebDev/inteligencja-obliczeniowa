import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


# pip install wordcloud matplotlib numpy pillow pandas nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt_tab")

with open("ArticleZad1.txt", "r", encoding="utf-8") as file:
    text = file.read()

# b) tokenizacja
tokens = word_tokenize(text.lower())  # zamiana na małe litery
print(f"Liczba słów po tokenizacji: {len(tokens)}")

# c) usuniecie stopwords
stop_words = set(stopwords.words("english"))
filtered_tokens = [word for word in tokens if word.isalpha()
                   and word not in stop_words]
print(f"Liczba słów po usunięciu stop-words: {len(filtered_tokens)}")

# czeste slowa
BagofWords = Counter(filtered_tokens)
print("\nBag of Words (Top 20):")
for word, count in BagofWords.most_common(30):
    print(f"{word}: {count}")


custom_stopwords = []
new_stopwords = ["said", "showing", "also", "like"]
custom_stopwords.extend(new_stopwords)

stop_words.update(custom_stopwords)
filtered_tokens_custom = [
    word for word in filtered_tokens if word not in custom_stopwords]

# Sprawdzenie liczby słów po dodaniu custom stop-words
print(
    f"\nLiczba słów po dodaniu custom stop-words: {len(filtered_tokens_custom)}")

# e) Lemmatyzacja
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(
    word) for word in filtered_tokens]
print(f"Liczba słów po lematyzacji: {len(lemmatized_tokens)}")


# f) Wektor słów (częstotliwość) + wykres
word_counts = Counter(lemmatized_tokens)
top_10_words = word_counts.most_common(10)

# Przygotowanie danych do wykresu
words = [word[0] for word in top_10_words]
counts = [word[1] for word in top_10_words]

# Rysowanie wykresu
plt.figure(figsize=(10, 6))
plt.bar(words, counts, color="skyblue")
plt.title("10 najczęściej występujących słów w artykule")
plt.xlabel("Słowa")
plt.ylabel("Liczba wystąpień")
plt.xticks(rotation=45)
plt.show()

# Wyświetlenie wektora słów
print("\nPrzykładowy wektor słów (częstotliwość):")
for word, count in list(word_counts.items())[:20]:
    print(f"{word}: {count}")

wordcloud = WordCloud(
    width=800, 
    height=400,
    background_color='white',
    stopwords=STOPWORDS, 
    max_words=100
).generate(text)

# Display the WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Hide axes
plt.show()

# Save to file
wordcloud.to_file("wordcloud.png")