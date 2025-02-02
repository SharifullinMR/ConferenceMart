import matplotlib.pyplot as plt
from collections import Counter

# Загрузка ключевых слов из файлов
with open("keywords_BertTopicLemma.txt", "r", encoding="utf-8") as f:
    bertopic_keywords = [keyword.strip() for line in f for keyword in line.split(",")]

with open("keywords_KeyBertLemma+stopslova.txt", "r", encoding="utf-8") as f:
    keybert_keywords = [keyword.strip() for line in f for keyword in line.split(",")]

# Подсчёт встречаемости ключевых слов
bertopic_counter = Counter(bertopic_keywords)
keybert_counter = Counter(keybert_keywords)
print(bertopic_counter)


# Отбираем топ-10 ключевых слов по встречаемости
top_bertopic = bertopic_counter.most_common(10)
top_keybert = keybert_counter.most_common(10)

# Построение диаграммы для BertTopic
plt.figure("Top 10 Keywords (BertTopic)")
plt.barh([kw for kw, _ in reversed(top_bertopic)], 
         [freq for _, freq in reversed(top_bertopic)], color='skyblue')
plt.title("Top 10 Keywords (BertTopic)")
plt.xlabel("Frequency")
plt.tight_layout()  # Для корректного отображения

# Построение диаграммы для KeyBert
plt.figure("Top 10 Keywords (KeyBert)")
plt.barh([kw for kw, _ in reversed(top_keybert)], 
         [freq for _, freq in reversed(top_keybert)], color='lightgreen')
plt.title("Top 10 Keywords (KeyBert)")
plt.xlabel("Frequency")
plt.tight_layout()  # Для корректного отображения

# Отображение окон
plt.show()


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Создание текста из ключевых слов для каждого метода
bertopic_text = " ".join(bertopic_keywords)
keybert_text = " ".join(keybert_keywords)

# Создание облаков слов
bertopic_wordcloud = WordCloud(width=800, height=400, background_color="white").generate(bertopic_text)
keybert_wordcloud = WordCloud(width=800, height=400, background_color="white").generate(keybert_text)

# Построение облака слов для BertTopic
plt.figure("Word Cloud (BertTopic)", figsize=(10, 5))
plt.imshow(bertopic_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud (BertTopic)")

# Построение облака слов для KeyBert
plt.figure("Word Cloud (KeyBert)", figsize=(10, 5))
plt.imshow(keybert_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud (KeyBert)")

# Отображение окон
plt.show()
