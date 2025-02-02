import os
os.chdir('C:/Users/Marsohodik/Desktop/pad/confer')
import nltk
from nltk.corpus import stopwords
import re

# Загрузка стоп-слов и модели лемматизации
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = set(stopwords.words('english'))

import spacy

# Загрузка модели spaCy
nlp = spacy.load("en_core_web_sm")

# Путь к файлу с ключевыми словами
input_file_path = 'C:/Users/Marsohodik/Desktop/pad/confer/Keybert_keywords.txt'
output_file_path = 'C:/Users/Marsohodik/Desktop/pad/confer/keywords_KeyBertLemma.txt'

# Функция для лемматизации набора ключевых слов
def lemmatize_keywords(input_path, output_path):
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                # Разделяем строку на фразы (предполагается, что они разделены запятыми)
                phrases = [phrase.strip() for phrase in line.strip().split(',')]
                
                # Лемматизация каждой фразы
                updated_phrases = []
                for phrase in phrases:
                    doc = nlp(phrase.lower())
                    # Лемматизируем каждое слово в фразе и собираем обратно в строку
                    lemmatized_phrase = " ".join([token.lemma_ for token in doc])
                    # Если лемматизация изменила фразу, заменяем ее и выводим изменения
                    if lemmatized_phrase != phrase.lower():
                        print(f"Изменение: '{phrase}' -> '{lemmatized_phrase}'")
                    updated_phrases.append(lemmatized_phrase if lemmatized_phrase != phrase.lower() else phrase)

                # Записываем результат в выходной файл
                outfile.write(", ".join(updated_phrases) + "\n")

        print(f"Ключевые слова сохранены с учетом изменений в файл: {output_path}")

    except FileNotFoundError:
        print(f"Ошибка: Файл {input_path} не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

# Запуск функции
lemmatize_keywords(input_file_path, output_file_path)
