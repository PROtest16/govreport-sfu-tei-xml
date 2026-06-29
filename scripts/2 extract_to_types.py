import lxml.etree as ET
import csv
import os
from collections import Counter

def extract_entities(working_dir, input_filename, output_filename):
    # Формируем полные пути к файлам
    input_path = os.path.join(working_dir, input_filename)
    output_path = os.path.join(working_dir, output_filename)
    
    # Проверка существования файла
    if not os.path.exists(input_path):
        print(f"Ошибка: Файл не найден по пути {input_path}")
        return

    # Регистрируем пространство имен TEI
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    
    try:
        parser = ET.XMLParser(recover=True)
        tree = ET.parse(input_path, parser)
        root = tree.getroot()
    except Exception as e:
        print(f"Ошибка при чтении XML: {e}")
        return

    entities_data = []
    tags_to_find = ['placeName', 'orgName', 'persName']
    
    for tag in tags_to_find:
        elements = root.xpath(f"//tei:{tag}", namespaces=ns)
        for elem in elements:
            # Очищаем текст от лишних пробелов
            text = " ".join(elem.text.split()) if elem.text else "[Нет текста]"
            ref = elem.get('ref', '') 
            entities_data.append((tag, text, ref))

    # Считаем частоту упоминаний
    stats = Counter(entities_data)

    # Записываем в CSV
    try:
        with open(output_path, mode='w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['Тег', 'Текст сущности', 'REF (ID)', 'Частота', 'Type (заполнить)', 'Subtype (заполнить)'])
            
            # Сортировка: сначала по Тегу, потом по убыванию частоты
            sorted_entities = sorted(stats.items(), key=lambda x: (x[0][0], -x[1]))
            
            for (tag, text, ref), count in sorted_entities:
                writer.writerow([tag, text, ref, count, '', ''])
        
        print(f"Успешно!")
        print(f"Обработан файл: {input_path}")
        print(f"Результат сохранен в: {output_path}")
        print(f"Найдено уникальных сущностей: {len(stats)}")
        
    except Exception as e:
        print(f"Ошибка при записи CSV: {e}")

# --- НАСТРОЙКИ ПУТЕЙ ---
# Используем префикс r перед строкой, чтобы Windows-путь с бэкслешами \ читался корректно
working_directory = r"C:\Users\vladimir\Desktop\TEI\govreport-sfu-tei-xml\tei_reports_with_tables_formation\1851_report"
input_file = "1851_report_tei_with_csv_links.xml"
output_file = "entities_list_1851.csv"

# ЗАПУСК
extract_entities(working_directory, input_file, output_file)