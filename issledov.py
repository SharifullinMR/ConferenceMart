import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Загружаем JSON файл в DataFrame
df = pd.read_json('KaggleDatasetArxiv.json', lines=True)
df.index = df.index + 1
print(df.head(10))

# Настраиваем векторизатор для TF-IDF с биграммами и триграммами
vectorizer_model = TfidfVectorizer(ngram_range=(1, 3), stop_words='english')

# Инициализируем эмбеддинговую модель и BERTopic с KeyBERTInspired
embedding_model = SentenceTransformer('all-MPNet-base-v2')
representation_model = KeyBERTInspired()
topic_model = BERTopic(
    embedding_model=embedding_model,
    representation_model=representation_model,
    vectorizer_model=vectorizer_model
)

# Преобразуем весь текст в список
texts = df['text'].tolist()

# Извлечение тем и вероятностей
topics, probabilities = topic_model.fit_transform(texts)

# Извлечение ключевых слов для каждой темы
all_keywords = []
for i, topic in enumerate(topics):
    keywords = topic_model.get_topic(topic)
    keyword_list = [kw[0] for kw in keywords]
    
    # Сохраняем ключевые слова в список
    all_keywords.append(', '.join(keyword_list))
    print(f"Ключевые слова для текста {i+1}: {', '.join(keyword_list)}")

# Сохраняем результаты в файл
output_file = "keywords_extracted1.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines("\n".join(all_keywords))

print(f"Ключевые слова сохранены в файл {output_file}")

# получение эмбеддингов
embeddings = embedding_model.encode(texts)

import json

# Сохраняем текст и его эмбеддинг как пары
embeddings_data = [{"text": text, "embedding": emb.tolist()} for text, emb in zip(texts, embeddings)]

# Записываем в JSON
with open("embeddings_with_texts.json", "w", encoding="utf-8") as f:
    json.dump(embeddings_data, f, ensure_ascii=False, indent=4)

topic_model.visualize_topics().write_html("topic_visualization.html")