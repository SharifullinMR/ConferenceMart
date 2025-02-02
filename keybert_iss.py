from keybert import KeyBERT
import pandas as pd

# Загружаем JSON файл в DataFrame
df = pd.read_json('KaggleDatasetArxiv.json', lines=True)
df.index = df.index + 1
print(df.head(10))

# KeyBERT
kw_model = KeyBERT(model='roberta-base')
keybert_keywords = []
for i, text in enumerate(df['text']):
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        use_mmr=True,
        diversity = 0.6
    )
    print(keywords)
    keywords_only = [kw[0] for kw in keywords]  # Оставляем только ключевые слова
    unique_keywords = list(set(keywords_only))  # Удаление дубликатов
    keybert_keywords.append(unique_keywords)
    print(f"\nKeyBERT Keywords for text {i + 1}:")
    print(unique_keywords)


# Сохранение результатов KeyBERT в файл
keybert_output_file_path = 'C:/Users/Marsohodik/Desktop/pad/confer/Keybert_keywords.txt'
with open(keybert_output_file_path, 'w', encoding='utf-8') as f:
    for keywords in keybert_keywords:
        f.write(', '.join(keywords) + '\n')

print(f"KeyBERT Keywords сохранены в файл: {keybert_output_file_path}")