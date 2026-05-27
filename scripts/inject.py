import lxml.etree as ET
import csv
import os

def inject_entities_back(working_dir, xml_input, csv_input, xml_output):
    # Полные пути
    xml_input_path = os.path.join(working_dir, xml_input)
    csv_input_path = os.path.join(working_dir, csv_input)
    xml_output_path = os.path.join(working_dir, xml_output)

    if not os.path.exists(csv_input_path):
        print(f"Ошибка: Не нашел CSV файл: {csv_input_path}")
        return

    # 1. Читаем CSV и создаем базу знаний (dictionary)
    # Ключ: (Тег, Текст), Значение: (Type, Subtype)
    mapping = {}
    with open(csv_input_path, mode='r', encoding='utf-8-sig') as f:
        # Используем разделитель ТОЧКА С ЗАПЯТОЙ, как в прошлом скрипте
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            tag = row['Тег']
            text = row['Текст сущности']
            t_type = row.get('Type (заполнить)', '').strip()
            t_subtype = row.get('Subtype (заполнить)', '').strip()
            
            if t_type or t_subtype:
                mapping[(tag, text)] = (t_type, t_subtype)

    print(f"Загружено правил из CSV: {len(mapping)}")

    # 2. Парсим XML
    parser = ET.XMLParser(remove_blank_text=False)
    tree = ET.parse(xml_input_path, parser)
    root = tree.getroot()
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

    # 3. Идем по XML и обновляем теги
    count_updated = 0
    tags_to_check = ['placeName', 'orgName', 'persName']

    for tag_name in tags_to_check:
        elements = root.xpath(f"//tei:{tag_name}", namespaces=ns)
        for elem in elements:
            # Важно: нормализуем текст так же, как при выгрузке
            text = " ".join(elem.text.split()) if elem.text else "[Нет текста]"
            
            if (tag_name, text) in mapping:
                t_type, t_subtype = mapping[(tag_name, text)]
                
                if t_type:
                    elem.set('type', t_type)
                if t_subtype:
                    elem.set('subtype', t_subtype)
                
                count_updated += 1

    # 4. Сохраняем результат
    # Используем метод, который максимально сохраняет форматирование
    tree.write(xml_output_path, encoding="utf-8", xml_declaration=True)
    
    print(f"--- УСПЕХ ---")
    print(f"Обновлено вхождений в тексте: {count_updated}")
    print(f"Файл сохранен: {xml_output_path}")

# --- НАСТРОЙКИ ПУТЕЙ ---
working_directory = r"C:\Users\vladimir\Desktop\TEI\govreport-sfu-tei-xml\tei_reports_with_tables_formation\1851_report"
input_xml = "1851_report_tei_with_csv_links.xml" 
input_csv = "entities_list_1851.csv" # Тот файл, который ты заполнил
output_xml = "1851_final_marked.xml" # Итоговый файл

# ЗАПУСК
inject_entities_back(working_directory, input_xml, input_csv, output_xml)