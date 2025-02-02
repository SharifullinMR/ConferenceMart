# Чтение исходного файла и удаление слов с 2d и 3d
input_file = "keywords_KeyBertLemma+stopslova.txt"
output_file = "keywords_KeyBertLemma+stopslova.txt"

with open(input_file, "r", encoding="utf-8") as f:
    updated_keywords = []
    for line in f:
        filtered_keywords = [keyword.strip() for keyword in line.split(",") if "2d" not in keyword.lower() and "3d" not in keyword.lower()]
        if filtered_keywords:
            updated_keywords.append(", ".join(filtered_keywords))

# Сохранение результата в новый файл
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(updated_keywords))

print(f"Файл успешно обработан и сохранен в {output_file}.")
