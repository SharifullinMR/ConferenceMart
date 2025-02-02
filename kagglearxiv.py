import pandas as pd

# Загружаем JSON файл в DataFrame
df = pd.read_json('KaggleDatasetArxiv.json', lines=True)
df.index = df.index + 1
# Выводим DataFrame на экран
print(df.head(10))
# Выводим отдельно столбец summary
print("\nСтолбец 'author':")
print(df['author'])

# Объединяем столбцы 'title' и 'summary' в новый столбец 'text'
df['text'] = df['title'] + '. ' + df['summary']

# Выводим результат
print(df[['title', 'summary', 'text']].head(10))
df.to_json('KaggleDatasetArxiv.json', orient='records', lines=True, force_ascii=False)

print("DataFrame успешно сохранен как KaggleDatasetArxiv.json")
