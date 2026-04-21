#!/usr/bin/env python3
"""
Скрипт для дедупликации сущностей - объединяет одинаковые имена
и собирает все источники для каждой уникальной сущности
"""

import json
from collections import defaultdict

def deduplicate_entities(entities):
    """Объединяет сущности с одинаковыми именами"""
    name_map = defaultdict(lambda: {'sources': [], 'ids': []})
    
    for entity in entities:
        name = entity['name']
        name_map[name]['sources'].append(entity['source'])
        name_map[name]['ids'].append(entity['id'])
    
    # Создаём список уникальных сущностей
    unique_entities = []
    for name, data in sorted(name_map.items()):
        unique_entities.append({
            'name': name,
            'ids': list(set(data['ids'])),  # уникальные ID
            'sources': data['sources'],
            'count': len(data['sources'])
        })
    
    return unique_entities

def main():
    # Загружаем исходные данные
    with open('/home/vladimir/.openclaw/workspace/entities_extracted.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("Дедупликация сущностей...")
    
    # Дедуплицируем только места и организации, персон оставляем как есть
    deduplicated = {
        'persons': data['persons'],  # Оставляем как есть - разные люди могут иметь одинаковые имена
        'places': deduplicate_entities(data['places']),
        'organizations': deduplicate_entities(data['organizations']),
        'metadata': {
            'original_counts': {
                'persons': len(data['persons']),
                'places': len(data['places']),
                'organizations': len(data['organizations'])
            },
            'unique_counts': {},
            'source_files': data['metadata']['files']
        }
    }
    
    # Обновляем счётчики
    deduplicated['metadata']['unique_counts'] = {
        'persons': len(deduplicated['persons']),
        'places': len(deduplicated['places']),
        'organizations': len(deduplicated['organizations'])
    }
    
    # Сохраняем результат
    output_file = '/home/vladimir/.openclaw/workspace/entities_deduplicated.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(deduplicated, f, ensure_ascii=False, indent=2)
    
    # Статистика
    print(f"\n{'='*60}")
    print("РЕЗУЛЬТАТЫ ДЕДУПЛИКАЦИИ:")
    print(f"{'='*60}")
    
    for entity_type in ['persons', 'places', 'organizations']:
        original = deduplicated['metadata']['original_counts'][entity_type]
        unique = deduplicated['metadata']['unique_counts'][entity_type]
        removed = original - unique
        print(f"{entity_type.upper():20} {original:3} → {unique:3} (удалено дубликатов: {removed})")
    
    print(f"\n{'='*60}")
    print(f"Результат сохранён в: {output_file}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
