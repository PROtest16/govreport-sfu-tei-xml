#!/usr/bin/env python3
"""
Скрипт для извлечения всех сущностей из standOff секций TEI XML файлов
Енисейских губернских отчётов
"""

import xml.etree.ElementTree as ET
import json
import os
from pathlib import Path
from collections import defaultdict

# Namespace для TEI
TEI_NS = {'tei': 'http://www.tei-c.org/ns/1.0'}

def extract_entities_from_file(xml_path):
    """Извлекает сущности из одного XML файла"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        entities = {
            'persons': [],
            'places': [],
            'organizations': []
        }
        
        # Ищем standOff секцию
        standoff = root.find('.//tei:standOff', TEI_NS)
        if standoff is None:
            return entities
        
        # Извлекаем персон
        list_person = standoff.find('.//tei:listPerson', TEI_NS)
        if list_person is not None:
            for person in list_person.findall('tei:person', TEI_NS):
                person_id = person.get('{http://www.w3.org/XML/1998/namespace}id')
                pers_name = person.find('tei:persName', TEI_NS)
                if pers_name is not None:
                    entities['persons'].append({
                        'id': person_id,
                        'name': pers_name.text or ''
                    })
        
        # Извлекаем места
        list_place = standoff.find('.//tei:listPlace', TEI_NS)
        if list_place is not None:
            for place in list_place.findall('tei:place', TEI_NS):
                place_id = place.get('{http://www.w3.org/XML/1998/namespace}id')
                place_name = place.find('tei:placeName', TEI_NS)
                if place_name is not None:
                    entities['places'].append({
                        'id': place_id,
                        'name': place_name.text or ''
                    })
        
        # Извлекаем организации
        list_org = standoff.find('.//tei:listOrg', TEI_NS)
        if list_org is not None:
            for org in list_org.findall('tei:org', TEI_NS):
                org_id = org.get('{http://www.w3.org/XML/1998/namespace}id')
                org_name = org.find('tei:orgName', TEI_NS)
                if org_name is not None:
                    entities['organizations'].append({
                        'id': org_id,
                        'name': org_name.text or ''
                    })
        
        return entities
        
    except Exception as e:
        print(f"Ошибка при обработке {xml_path}: {e}")
        return None

def main():
    # Путь к директории с отчётами
    reports_dir = Path('/home/vladimir/.openclaw/workspace/govreport-sfu-tei-xml/tei_reports_with_tables_formation')
    
    # Собираем все сущности
    all_entities = {
        'persons': [],
        'places': [],
        'organizations': [],
        'metadata': {
            'total_files': 0,
            'processed_files': 0,
            'files': []
        }
    }
    
    # Проходим по всем XML файлам
    xml_files = list(reports_dir.rglob('*.xml'))
    all_entities['metadata']['total_files'] = len(xml_files)
    
    for xml_file in xml_files:
        print(f"Обрабатываю: {xml_file.name}")
        entities = extract_entities_from_file(xml_file)
        
        if entities:
            # Добавляем информацию об источнике к каждой сущности
            source_info = {
                'file': xml_file.name,
                'path': str(xml_file.relative_to(reports_dir))
            }
            
            for person in entities['persons']:
                person['source'] = source_info
                all_entities['persons'].append(person)
            
            for place in entities['places']:
                place['source'] = source_info
                all_entities['places'].append(place)
            
            for org in entities['organizations']:
                org['source'] = source_info
                all_entities['organizations'].append(org)
            
            all_entities['metadata']['processed_files'] += 1
            all_entities['metadata']['files'].append(source_info)
    
    # Сохраняем результат
    output_file = Path('/home/vladimir/.openclaw/workspace/entities_extracted.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_entities, f, ensure_ascii=False, indent=2)
    
    # Статистика
    print(f"\n{'='*60}")
    print(f"Обработано файлов: {all_entities['metadata']['processed_files']}/{all_entities['metadata']['total_files']}")
    print(f"Извлечено персон: {len(all_entities['persons'])}")
    print(f"Извлечено мест: {len(all_entities['places'])}")
    print(f"Извлечено организаций: {len(all_entities['organizations'])}")
    print(f"\nРезультат сохранён в: {output_file}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
