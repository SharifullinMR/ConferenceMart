import json
import ast
import pandas as pd
# Укажите путь к вашему файлу JSON
file_path = 'arxivData.json'

# Функция для извлечения данных
def process_data(data):
    processed_data = []
    for item in data:
        # Преобразуем строки в списки
        authors = ast.literal_eval(item['author'])  # Преобразуем строку в список
        tags = ast.literal_eval(item['tag'])  # Преобразуем строку в список

        # Извлекаем имена авторов
        author_names = [author['name'] for author in authors]
        
        # Извлекаем термины из тегов
        tag_terms = [tag['term'] for tag in tags]

        # Формируем итоговый словарь
        processed_item = {
            'title': item['title'],
            'summary': item['summary'],
            'author': ', '.join(author_names),  # Объединяем имена авторов в строку
            'tag': ', '.join(tag_terms)  # Объединяем теги в строку
        }
        processed_data.append(processed_item)
    
    return processed_data

# Чтение JSON-файла
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Преобразуем данные
    processed_data = process_data(data)

    # Создаем DataFrame
    df = pd.DataFrame(processed_data)
     # Устанавливаем индексы с 1
    df.index = df.index + 1
    # Выводим DataFrame
    print(df)
    
except FileNotFoundError:
    print(f"Файл {file_path} не найден.")
except json.JSONDecodeError as e:
    print(f"Ошибка чтения JSON: {e}")

# Выводим количество строк в DataFrame
print(f"Количество строк в DataFrame: {len(df)}")


# Выводим отдельно столбец summary
print("\nСтолбец 'author':")
print(df['author'])

df.to_json('KaggleDatasetArxiv.json', orient='records', lines=True, force_ascii=False)

print("DataFrame успешно сохранен как KaggleDatasetArxiv.json")

