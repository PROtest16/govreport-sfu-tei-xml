from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

try:
    from natasha import DatesExtractor, Doc, MorphVocab, NewsEmbedding, NewsNERTagger, Segmenter
except Exception:  # pragma: no cover - optional dependency in runtime
    DatesExtractor = None
    Doc = None
    MorphVocab = None
    NewsEmbedding = None
    NewsNERTagger = None
    Segmenter = None

TEI_NS = "http://www.tei-c.org/ns/1.0"
XML_NS = "http://www.w3.org/XML/1998/namespace"
ET.register_namespace("", TEI_NS)

NS = f"{{{TEI_NS}}}"
XML_ID = f"{{{XML_NS}}}id"
YEAR_MIN = 1800
YEAR_MAX = 1917

PAGE_BREAK_RE = re.compile(r"\[\[PAGE_BREAK:(\d+)\]\]")
YEAR_RE = re.compile(rf"(?<!\d)((?:18\d{{2}}|19(?:0\d|1[0-7])))(?!\d)\s*(?:г\.|года|год|году)")
FULL_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+"
    r"(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+"
    r"((?:18\d{2}|19(?:0\d|1[0-7])))\s*(?:г\.|года)?",
    flags=re.I,
)
MONTHS = {
    "января": "01",
    "февраля": "02",
    "марта": "03",
    "апреля": "04",
    "мая": "05",
    "июня": "06",
    "июля": "07",
    "августа": "08",
    "сентября": "09",
    "октября": "10",
    "ноября": "11",
    "декабря": "12",
}


@dataclass(frozen=True)
class EntityDef:
    xml_id: str
    kind: str
    headword: str
    patterns: tuple[str, ...]


@dataclass(frozen=True)
class DocumentConfig:
    report_id: int
    source_txt: Path
    output_xml: Path
    title: str
    archive_note: str
    pdf_url: str
    pdf_original_url: str | None
    report_url: str
    source: str
    text_type: str
    page_count: int


@dataclass(frozen=True)
class DynamicEntity:
    xml_id: str
    kind: str
    headword: str


@dataclass(frozen=True)
class EntityMatch:
    start: int
    end: int
    kind: str
    xml_id: str
    headword: str
    subtype: str = "heuristic"
    priority: int = 1


@dataclass(frozen=True)
class DateMatch:
    start: int
    end: int
    when: str
    subtype: str
    priority: int


@dataclass(frozen=True)
class NatashaModels:
    segmenter: Any
    ner_tagger: Any
    dates_extractor: Any


PLACE_ENTITY_DEFS: tuple[EntityDef, ...] = (
    EntityDef(
        "place_krasnoyarsk",
        "placeName",
        "Красноярск",
        (
            r"г\.\s*Красноярск(?:е|а|у|ом|ий|ого|ому|им|ая|ой|ую|ою)?",
            r"Красноярск(?:е|а|у|ом|ий|ого|ому|им|ая|ой|ую|ою)?",
        ),
    ),
    EntityDef("place_eniseysk", "placeName", "Енисейск", (r"г\.\s*Енисейск(?:е|а|у|ом)?", r"Енисейск(?:е|а|у|ом)?")),
    EntityDef("place_kansk", "placeName", "Канск", (r"Канск(?:е|а|у|ом|ий|ого|ому|им|ая|ой|ую|ою)?",)),
    EntityDef("place_achinsk", "placeName", "Ачинск", (r"Ачинск(?:е|а|у|ом|ий|ого|ому|им|ая|ой|ую|ою)?",)),
    EntityDef("place_minusinsk", "placeName", "Минусинск", (r"Минусинск(?:е|а|у|ом|ий|ого|ому|им|ая|ой|ую|ою)?",)),
    EntityDef("place_turukhansk", "placeName", "Туруханск", (r"Туруханск(?:е|а|у|ом)?",)),
    EntityDef("place_enisei_province", "placeName", "Енисейская губерния", (r"Енисейск(?:ая|ой|ую|ою)\s+губерни(?:я|и|е|ю|ей)",)),
    EntityDef("place_tomsk_province", "placeName", "Томская губерния", (r"Томск(?:ая|ой|ую|ою)\s+губерни(?:я|и|е|ю|ей)",)),
    EntityDef("place_irkutsk_province", "placeName", "Иркутская губерния", (r"Иркутск(?:ая|ой|ую|ою)\s+губерни(?:я|и|е|ю|ей)",)),
    EntityDef("place_eastern_siberia", "placeName", "Восточная Сибирь", (r"Восточн(?:ая|ой|ую|ею)\s+Сибир(?:ь|и|ью)",)),
    EntityDef("place_turukhansk_region", "placeName", "Туруханский край", (r"Туруханск(?:ий|ого|ому|им)\s+кра(?:й|я|ю|ем)",)),
    EntityDef("place_enisei_okrug", "placeName", "Енисейский округ", (r"Енисейск(?:ий|ого|ому|им|ом)\s+округ(?:а|е|у|ом)?",)),
    EntityDef("place_krasnoyarsk_okrug", "placeName", "Красноярский округ", (r"Красноярск(?:ий|ого|ому|им|ом)\s+округ(?:а|е|у|ом)?",)),
    EntityDef("place_kansk_okrug", "placeName", "Канский округ", (r"Канск(?:ий|ого|ому|им|ом)\s+округ(?:а|е|у|ом)?",)),
    EntityDef("place_achinsk_okrug", "placeName", "Ачинский округ", (r"Ачинск(?:ий|ого|ому|им|ом)\s+округ(?:а|е|у|ом)?",)),
    EntityDef("place_minusinsk_okrug", "placeName", "Минусинский округ", (r"Минусинск(?:ий|ого|ому|им|ом)\s+округ(?:а|е|у|ом)?",)),
    EntityDef("place_troitskoe", "placeName", "Троицкое", (r"сел(?:о|е)\s+Троицк(?:ое|ом)",)),
)

ORG_ENTITY_DEFS: tuple[EntityDef, ...] = (
    EntityDef("org_enisei_general_governance", "orgName", "Енисейское Общее Губернское Управление", (r"Енисейск(?:ое|ого|ому|им)\s+Общ(?:ее|его|ему|им)\s+Губернск(?:ое|ого|ому|им)\s+Управлени(?:е|я|ю|ем)",)),
    EntityDef("org_police_department", "orgName", "Департамент полиции", (r"Департамент[ауеом]?\s+полиции",)),
    EntityDef("org_gubernskoe_pravlenie", "orgName", "Губернское правление", (r"Губернск(?:ое|ого|ому|им)\s+правлени(?:е|я|ю|ем)",)),
    EntityDef("org_prikaz_public_welfare", "orgName", "Приказ общественного призрения", (r"Приказ[ауеом]?\s+общественн(?:ого|ому|ым)\s+призрени(?:я|ю|ем)",)),
    EntityDef("org_medical_board", "orgName", "Врачебная управа", (r"Врачебн(?:ая|ой|ую|ою)\s+управ(?:а|ы|е|у|ой)",)),
    EntityDef("org_recruit_board", "orgName", "Рекрутское присутствие", (r"Рекрутск(?:ое|ого|ому|им)\s+присутстви(?:е|я|ю|ем)",)),
    EntityDef("org_public_library", "orgName", "Красноярская губернская публичная библиотека", (r"Красноярск(?:ая|ой|ую|ою)\s+губернск(?:ая|ой|ую|ою)\s+публичн(?:ая|ой|ую|ою)\s+библиотек(?:а|и|е|у|ой)",)),
    EntityDef("org_irkutsk_gymnasium", "orgName", "Иркутская гимназия", (r"Иркутск(?:ая|ой|ую|ою)\s+гимнази(?:я|и|е|ю|ей)",)),
    EntityDef("org_cossack_regiment", "orgName", "Енисейский конный казачий полк", (r"Енисейск(?:ому|ий|ого|им)\s+конн(?:ому|ый|ого|ым)\s+казачь(?:ему|ий|его|им)\s+полк(?:у|а|ом)?",)),
    EntityDef("org_private_board", "orgName", "Красноярская Частная Управа", (r"Красноярск(?:ой|ая|ую|ою)\s+Частн(?:ой|ая|ую|ою)\s+Управ(?:е|а|у|ой)",)),
    EntityDef("org_house_of_correction", "orgName", "Смирительный дом", (r"Смирительн(?:ого|ый|ом|ому)\s+дом(?:а|у|ом)?",)),
    EntityDef("org_troitsk_salt_works", "orgName", "Троицкий солеваренный завод", (r"Троицк(?:ому|ий|ого|им)\s+солеваренн(?:ому|ый|ого|ым)\s+завод(?:у|а|ом)?",)),
    EntityDef("org_statistical_committee", "orgName", "Енисейский губернский статистический комитет", (r"Енисейск(?:ий|ого|ому|им)\s+Губернск(?:ий|ого|ому|им)\s+Статистическ(?:ий|ого|ому|им)\s+комитет",)),
)

ENTITY_DEFS = PLACE_ENTITY_DEFS + ORG_ENTITY_DEFS

PERSON_TITLE_PATTERN = re.compile(
    r"\b(?:губернатор|генерал-губернатор|министр|советник|исправник|секретарь|доктор|лекарь|"
    r"полковник|майор|генерал-майор|статский советник|титулярный советник|купец|мещанин|"
    r"рядовой|унтер-офицер|фельдфебель|цирюльник|почтальон|председатель|начальник)\s+"
    r"([А-ЯЁ][а-яё-]+(?:\s+[А-ЯЁ][а-яё-]+){0,2})"
)
FULL_NAME_WITH_PATRONYMIC_PATTERN = re.compile(
    r"\b([А-ЯЁ][а-яё-]+\s+[А-ЯЁ][а-яё-]+(?:ович|евич|ична|инична)\s+[А-ЯЁ][а-яё-]+)\b"
)
SURNAME_SUFFIXES = (
    "ов", "ова", "ев", "ева", "ин", "ина", "ын", "ына", "ский", "ская",
    "цкий", "цкая", "ко", "ич", "ович", "евич", "енко", "ук", "юк",
)
PERSON_STOPWORDS = {
    "Восточной", "Сибири", "Енисейской", "губернии", "Губернского", "Губернское",
    "Общее", "Управление", "Правление", "Комитет", "Министерство", "Департамент",
    "Приказ", "Управа", "Суд", "Полиция", "Красноярска", "Красноярск", "Енисейска",
    "Енисейск", "Канска", "Канск", "Ачинска", "Ачинск", "Минусинска", "Минусинск",
    "Туруханска", "Туруханск",
}
TABLE_HEADWORDS = ("ведомость", "табель", "опись", "содержание", "итого", "всего", "№")
PERSON_TITLE_WORDS = {word.lower() for title in (
    "губернатор",
    "генерал-губернатор",
    "министр",
    "советник",
    "исправник",
    "секретарь",
    "доктор",
    "лекарь",
    "полковник",
    "майор",
    "генерал-майор",
    "статский советник",
    "титулярный советник",
    "купец",
    "мещанин",
    "рядовой",
    "унтер-офицер",
    "фельдфебель",
    "цирюльник",
    "почтальон",
    "председатель",
    "начальник",
) for word in title.split()}
PERSON_PATRO_SUFFIXES = ("ович", "евич", "ична", "инична")
PERSON_NOISE_WORDS = {
    "вашему",
    "высокопревосходительству",
    "господину",
    "императорскому",
    "величеству",
    "министру",
    "внутренних",
    "дел",
    "отделения",
    "стола",
    "отделению",
    "столу",
}
PERSON_CONTEXT_WORDS = {
    "округа",
    "округом",
    "округе",
    "города",
    "городе",
    "суда",
    "суду",
    "судом",
    "совета",
    "совету",
    "управления",
    "управлению",
    "управы",
    "правления",
    "палаты",
    "комитета",
    "отделения",
    "стола",
    "председателя",
    "председателю",
    "начальника",
    "начальнику",
    "губернатора",
    "губернатору",
    "советника",
    "советнику",
    "лекаря",
    "доктора",
}
PLACE_CONTEXT_RE_1 = re.compile(
    r"^(?:г\.|город(?:е|а|ом|у)?|село|селе|села|дер\.|деревн(?:я|е|и)|станиц(?:а|е|и)|"
    r"улус(?:е|а)?|слобод(?:а|е|ы)|волост(?:ь|и|ью))\s+[А-ЯЁ][А-Яа-яё.-]+(?:\s+[А-ЯЁ][А-Яа-яё.-]+){0,2}$"
)
PLACE_CONTEXT_RE_2 = re.compile(
    r"^[А-ЯЁ][а-яё-]+(?:ской|ская|ского|скую|ском|скою|ский|ским|ские|ских)\s+"
    r"(?:губерни(?:я|и|е|ю|ей)|округ(?:а|е|у|ом)?|кра(?:й|я|е|ем)|област(?:ь|и|ью))$"
)
PLACE_NOISE_WORDS = {"начальника", "министра", "господину", "вашему", "величеству", "отделения", "стола"}
ORG_ROOT_RE = re.compile(
    r"(правлен|управ|суд|полиц|приказ|комитет|министер|департамент|дум|совет|комисси|палат|"
    r"архив|библиотек|гимнази|присутств|казначейств|канцеляр|больниц)"
)
ORG_LAST_TOKEN_RE = re.compile(
    r"^(?:правлени[еяюем]|управлени[еяюем]|управ[аеуыой]|суд(?:а|е|у|ом)?|полици(?:я|и|ю|ей)|"
    r"приказ(?:а|у|ом)?|комитет(?:а|е|ом)?|министерств(?:о|а)|департамент(?:а|е|ом)?|"
    r"дум(?:а|ы|е)|совет(?:а|е|ом)?|комисси(?:я|и|ю|ей)|палат(?:а|ы|е)|архив(?:а|е|у|ом)?|"
    r"библиотек(?:а|и|е|у|ой)|гимнази(?:я|и|е|ю|ей)|присутстви(?:е|я|ю|ем)|казначейств(?:о|а|е)|"
    r"канцеляри(?:я|и|ю|ей)|больниц(?:а|ы|е|у|ой))$"
)
ORG_STOP_TAILS = {
    "для",
    "дел",
    "привить",
    "неоконченных",
    "отделения",
    "отделение",
    "стол",
    "стола",
}
ADJECTIVAL_PLACE_TOKEN_RE = re.compile(
    r"\b[А-ЯЁ][а-яё-]{3,}(?:ский|ского|скому|ским|ском|ская|ской|скую|скою)\b"
)
PLACE_CONTEXT_CUE_RE = re.compile(
    r"(?:по\s+уездам?|уезд(?:а|е|ов|ам|ом|ы)?|волост(?:ь|и|ям|ях|ью)?|"
    r"округ(?:а|е|у|ом|и)?|губерни(?:я|и|е|ю|ей)|кра(?:й|я|е|ем)|област(?:ь|и|ью)|"
    r"г\.\s*[А-ЯЁ]|город(?:а|е|ом|у)?|село|деревн(?:я|е|и)|станиц(?:а|е|и))",
    flags=re.I,
)
PLACE_RIGHT_CONTEXT_RE = re.compile(
    r"^\s*(?:уезд(?:а|е|ов|ам|ом|ы)?|волост(?:ь|и|ям|ях|ью)?|"
    r"округ(?:а|е|у|ом|и)?|губерни(?:я|и|е|ю|ей)|кра(?:й|я|е|ем)|област(?:ь|и|ью))\b",
    flags=re.I,
)
PLACE_LIST_TAIL_RE = re.compile(r"(?:[\s,]|(?:\bи\b)|(?:\bили\b)|(?:\bотчасти\b))+$", flags=re.I)
SK_CASE_ENDINGS = (
    "",
    "а",
    "у",
    "е",
    "ом",
    "ий",
    "ого",
    "ому",
    "им",
    "ая",
    "ой",
    "ую",
    "ою",
    "ое",
    "ые",
    "ых",
    "ыми",
    "ие",
    "их",
    "ими",
)
SK_ADJECTIVAL_SUFFIXES = (
    "ский",
    "ского",
    "скому",
    "ским",
    "ском",
    "ская",
    "ской",
    "скую",
    "скою",
    "ское",
    "ские",
    "ских",
    "скими",
)
RUS_FIRST_NAMES = {
    "Александр", "Алексей", "Андрей", "Антон", "Аркадий", "Афанасий", "Василий", "Виктор",
    "Владимир", "Всеволод", "Гавриил", "Георгий", "Григорий", "Даниил", "Дмитрий", "Евгений",
    "Егор", "Елисей", "Захар", "Иван", "Игорь", "Илья", "Кирилл", "Константин", "Лев", "Леонид",
    "Максим", "Матвей", "Михаил", "Никита", "Николай", "Павел", "Петр", "Роман", "Семен",
    "Сергей", "Степан", "Тимофей", "Федор", "Яков", "Авдотья", "Аграфена", "Александра",
    "Анастасья", "Анна", "Варвара", "Екатерина", "Елена", "Лукерья", "Мария", "Надежда", "Пелагея",
}
NATASHA_MODELS: NatashaModels | None = None


def tei(tag: str) -> str:
    return f"{NS}{tag}"


def slugify(text: str) -> str:
    text = text.lower()
    text = text.replace("ё", "e")
    text = re.sub(r"[^0-9a-zа-я]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def compact_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def year_in_range(value: int) -> bool:
    return YEAR_MIN <= value <= YEAR_MAX


def preceded_by_number_sign(text: str, start: int) -> bool:
    return "№" in text[max(0, start - 3):start]


def get_natasha_models(enabled: bool) -> NatashaModels | None:
    global NATASHA_MODELS
    if not enabled or Segmenter is None:
        return None
    if NATASHA_MODELS is None:
        segmenter = Segmenter()
        embedding = NewsEmbedding()
        ner_tagger = NewsNERTagger(embedding)
        morph_vocab = MorphVocab()
        dates_extractor = DatesExtractor(morph_vocab)
        NATASHA_MODELS = NatashaModels(segmenter=segmenter, ner_tagger=ner_tagger, dates_extractor=dates_extractor)
    return NATASHA_MODELS


def is_valid_person_surface(surface: str) -> bool:
    cleaned = compact_spaces(surface.strip(" ,.;:()[]«»\"'"))
    if len(cleaned) < 3 or len(cleaned) > 80:
        return False
    if any(ch.isdigit() for ch in cleaned):
        return False
    tokens = re.findall(r"[А-ЯЁ][а-яё-]+", cleaned)
    if not tokens:
        return False
    while tokens and tokens[0].lower() in PERSON_TITLE_WORDS:
        tokens.pop(0)
    if not tokens:
        return False
    if any(token.lower() in PERSON_NOISE_WORDS for token in tokens):
        return False
    if len(tokens) > 3:
        return False
    if len(tokens) == 1:
        return len(tokens[0]) >= 5 and tokens[0].lower().endswith(SURNAME_SUFFIXES)
    if len(tokens) == 2:
        if tokens[0] not in RUS_FIRST_NAMES:
            return False
        return tokens[1].lower().endswith(SURNAME_SUFFIXES)
    if tokens[0] not in RUS_FIRST_NAMES:
        return False
    return tokens[1].lower().endswith(PERSON_PATRO_SUFFIXES) and tokens[2].lower().endswith(SURNAME_SUFFIXES)


def is_valid_org_surface(surface: str) -> bool:
    cleaned = compact_spaces(surface.strip(" ,.;:()[]«»\"'"))
    if len(cleaned) < 5 or len(cleaned) > 120:
        return False
    words = cleaned.split()
    if len(words) < 2 or len(words) > 8:
        return False
    lowered = cleaned.lower()
    if lowered.startswith("дела "):
        return False
    if cleaned.isupper() and len(words) > 6:
        return False
    if not ORG_ROOT_RE.search(lowered):
        return False
    last_token = re.findall(r"[А-Яа-яЁё-]+", cleaned)[-1].lower()
    if last_token in ORG_STOP_TAILS:
        return False
    if not ORG_LAST_TOKEN_RE.fullmatch(last_token):
        return False
    return True


def is_valid_place_surface(surface: str, place_forms: set[str]) -> bool:
    cleaned = compact_spaces(surface.strip(" ,.;:()[]«»\"'"))
    if len(cleaned) < 2 or len(cleaned) > 80:
        return False
    if any(ch.isdigit() for ch in cleaned):
        return False
    tokens = re.findall(r"[А-ЯЁ][а-яё-]+", cleaned)
    if any(token.lower() in PLACE_NOISE_WORDS for token in tokens):
        return False
    if cleaned in place_forms:
        return True
    if PLACE_CONTEXT_RE_1.match(cleaned):
        return True
    if PLACE_CONTEXT_RE_2.match(cleaned):
        return True
    return False


def extract_toponym_stems(headword: str) -> set[str]:
    stems: set[str] = set()
    for token in re.findall(r"[А-ЯЁ][а-яё-]+", headword):
        if token.endswith("ск"):
            stems.add(token)
            continue
        for suffix in SK_ADJECTIVAL_SUFFIXES:
            if token.endswith(suffix) and len(token) > len(suffix) + 1:
                stems.add(token[: -len(suffix)] + "ск")
                break
    return stems


def generate_toponym_forms(headword: str) -> set[str]:
    forms: set[str] = set()
    for stem in extract_toponym_stems(headword):
        for ending in SK_CASE_ENDINGS:
            forms.add(f"{stem}{ending}")
    return forms


def generate_toponym_patterns(headword: str) -> tuple[str, ...]:
    patterns: list[str] = []
    endings = "|".join(re.escape(ending) for ending in SK_CASE_ENDINGS if ending)
    for stem in extract_toponym_stems(headword):
        patterns.append(rf"{re.escape(stem)}(?:{endings})?")
    return tuple(patterns)


def collect_place_forms(raw_text: str) -> set[str]:
    forms = {entity.headword for entity in PLACE_ENTITY_DEFS}
    for entity in PLACE_ENTITY_DEFS:
        forms.update(generate_toponym_forms(entity.headword))
    patterns = (
        r"\b(?:г\.|город(?:е|а|ом|у)?|село|селе|села|дер\.|деревн(?:я|е|и)|станиц(?:а|е|и)|"
        r"улус(?:е|а)?|слобод(?:а|е|ы)|волост(?:ь|и|ью))\s+[А-ЯЁ][А-Яа-яё.-]+(?:\s+[А-ЯЁ][А-Яа-яё.-]+){0,2}",
        r"\b[А-ЯЁ][а-яё-]+(?:ской|ская|ского|скую|ском|скою|ский|ским|ские|ских)\s+"
        r"(?:губерни(?:я|и|е|ю|ей)|округ(?:а|е|у|ом)?|кра(?:й|я|е|ем)|област(?:ь|и|ью))",
    )
    sample = raw_text[:250_000]
    for pattern in patterns:
        for match in re.finditer(pattern, sample):
            candidate = compact_spaces(match.group(0).strip(" ,.;:()[]«»\"'"))
            if len(candidate) > 40:
                continue
            if len(candidate.split()) > 4:
                continue
            if any(ch.isdigit() for ch in candidate):
                continue
            forms.add(candidate)
    return forms


def load_document_configs(manifest_path: Path, output_dir: Path) -> list[DocumentConfig]:
    items = json.loads(manifest_path.read_text(encoding="utf-8"))
    configs: list[DocumentConfig] = []
    for item in items:
        report_id = int(item.get("report_id", item.get("id")))
        title = item["title"]
        year_match = re.match(r"(\d{4})", title)
        stem = slugify(f"{year_match.group(1) if year_match else report_id}_{title}")[:120]
        configs.append(
            DocumentConfig(
                report_id=report_id,
                source_txt=Path(item["text_path"]),
                output_xml=output_dir / f"{stem}.tei.xml",
                title=title,
                archive_note=item.get("imprint", ""),
                pdf_url=item.get("pdf_url", ""),
                pdf_original_url=item.get("pdf_original_url"),
                report_url=item.get("report_url", ""),
                source=item.get("source", ""),
                text_type=item.get("text_type", ""),
                page_count=int(item.get("page_count", 0)),
            )
        )
    return configs


def split_pages(raw_text: str) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    current_page = 1
    cursor = 0
    for match in PAGE_BREAK_RE.finditer(raw_text):
        pages.append((current_page, raw_text[cursor:match.start()]))
        current_page = int(match.group(1))
        cursor = match.end()
    pages.append((current_page, raw_text[cursor:]))
    return pages


def normalize_block(block: str) -> str:
    block = block.replace("\r\n", "\n").replace("\r", "\n")
    block = re.sub(r"(?<=\w)\s*-\s*\n\s*(?=\w)", "", block)
    block = re.sub(r"(?<=[А-Яа-яЁёA-Za-z])\s+-\s+(?=[А-Яа-яЁёA-Za-z])", "", block)
    block = re.sub(
        r"\b([А-ЯЁ][а-яё]{3,})\s+(ск(?:ий|ого|ому|им|ом|ая|ой|ую|ою|ое|ие|их|ими|ою))\b",
        r"\1\2",
        block,
    )
    block = re.sub(r"(?<=\S)\n(?=\S)", " ", block)
    lines = [line.strip() for line in block.splitlines()]
    lines = [line for line in lines if line and line != "_______________" and not re.fullmatch(r"\d+", line)]
    cleaned = " ".join(lines)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def normalize_line(line: str) -> str:
    line = line.replace("\r", "").strip()
    line = re.sub(r"(?<=[А-Яа-яЁёA-Za-z])\s+-\s+(?=[А-Яа-яЁёA-Za-z])", "", line)
    line = re.sub(
        r"\b([А-ЯЁ][а-яё]{3,})\s+(ск(?:ий|ого|ому|им|ом|ая|ой|ую|ою|ое|ие|их|ими|ою))\b",
        r"\1\2",
        line,
    )
    line = re.sub(r"\s+", " ", line)
    return line


def iter_paragraphs(page_text: str) -> Iterable[str]:
    for block in re.split(r"\n\s*\n", page_text):
        cleaned = normalize_block(block)
        if cleaned:
            yield cleaned


def split_table_cells(line: str) -> list[str]:
    line = normalize_line(line)
    if not line:
        return []
    explicit = [part.strip() for part in re.split(r"\s{2,}", line) if part.strip()]
    if len(explicit) >= 2:
        return explicit
    # Text-first table parsing: split a row into a textual head and a stable numeric tail.
    match = re.search(r"((?:\s+(?:\d+(?:[.,/]\d+)?|[–-])){2,})\s*$", line)
    if match:
        head = line[:match.start(1)].strip(" -–")
        tail = re.findall(r"\d+(?:[.,/]\d+)?|[–-]", match.group(1))
        if head and len(tail) >= 2:
            return [head, *tail]
    return [line]


def is_probable_table_block(lines: list[str]) -> bool:
    if len(lines) < 5:
        return False
    normalized = [normalize_line(line) for line in lines if normalize_line(line)]
    if len(normalized) < 3:
        return False
    header_text = " ".join(normalized[:3]).lower()
    numeric_rows = 0
    segmented_rows = 0
    short_lines = 0
    for line in normalized:
        digit_groups = len(re.findall(r"\d+(?:[.,/]\d+)?", line))
        if digit_groups >= 2:
            numeric_rows += 1
        if len(split_table_cells(line)) >= 3:
            segmented_rows += 1
        if len(line) <= 90:
            short_lines += 1
    if any(word in header_text for word in TABLE_HEADWORDS) and segmented_rows >= 2 and numeric_rows >= 2:
        return True
    if short_lines < max(3, len(normalized) // 2):
        return False
    return segmented_rows >= max(5, len(normalized) // 2)


def build_table_rows(lines: list[str]) -> list[tuple[str, list[str]]]:
    normalized = [normalize_line(line) for line in lines if normalize_line(line) and not re.fullmatch(r"\d{1,3}", normalize_line(line))]
    row_candidates = [split_table_cells(line) for line in normalized]
    widths = [len(cells) for cells in row_candidates if len(cells) >= 2]
    target_width = Counter(widths).most_common(1)[0][0] if widths else 1

    rows: list[tuple[str, list[str]]] = []
    current_row: list[str] | None = None
    header_done = False

    for line, cells in zip(normalized, row_candidates):
        if len(cells) >= target_width and target_width >= 2:
            padded = cells + [""] * (target_width - len(cells))
            if current_row is not None:
                rows.append(("data", current_row))
            current_row = padded
            header_done = True
            continue

        if not header_done:
            rows.append(("label", [line]))
            continue

        if current_row is None:
            rows.append(("label", [line]))
            continue

        current_row[0] = f"{current_row[0]} {line}".strip()

    if current_row is not None:
        rows.append(("data", current_row))
    return rows


def iter_page_elements(page_text: str) -> Iterable[tuple[str, list[str] | str]]:
    blocks = [block for block in re.split(r"\n\s*\n", page_text) if block.strip()]
    for block in blocks:
        lines = [line for line in block.splitlines() if normalize_line(line)]
        if is_probable_table_block(lines):
            yield ("table", lines)
        else:
            paragraph = normalize_block(block)
            if paragraph:
                yield ("p", paragraph)


def append_text(parent: ET.Element, last_child: ET.Element | None, text: str) -> None:
    if not text:
        return
    if last_child is None:
        parent.text = (parent.text or "") + text
    else:
        last_child.tail = (last_child.tail or "") + text


def canonicalize_person(surface: str) -> str | None:
    cleaned = compact_spaces(surface.strip(" ,.;:()[]«»\"'"))
    tokens = [token for token in cleaned.split() if token]
    if not tokens:
        return None
    while tokens and tokens[0].lower() in {
        "губернатор", "генерал-губернатор", "министр", "советник", "исправник", "секретарь",
        "доктор", "лекарь", "полковник", "майор", "генерал-майор", "статский", "титулярный",
        "купец", "мещанин", "рядовой", "унтер-офицер", "фельдфебель", "цирюльник",
        "почтальон", "председатель", "начальник",
    }:
        tokens.pop(0)
    if not tokens:
        return None
    if len(tokens) == 1:
        if tokens[0].lower().endswith(SURNAME_SUFFIXES):
            return tokens[0]
        return None
    if any(token in PERSON_STOPWORDS for token in tokens):
        return None
    if len(tokens) == 2:
        first, last = tokens
        if first in RUS_FIRST_NAMES and last.lower().endswith(SURNAME_SUFFIXES):
            return f"{first} {last}"
        if first.lower() in PERSON_CONTEXT_WORDS and last.lower().endswith(SURNAME_SUFFIXES):
            return last
        return None
    if len(tokens) == 3:
        first, middle, last = tokens
        if first in RUS_FIRST_NAMES and middle.lower().endswith(("ович", "евич", "ична", "инична")) and last.lower().endswith(SURNAME_SUFFIXES):
            return f"{first} {middle} {last}"
        if first.lower() in PERSON_CONTEXT_WORDS and middle.lower() in PERSON_CONTEXT_WORDS and last.lower().endswith(SURNAME_SUFFIXES):
            return last
        return None
    return None


def add_dynamic_entity(
    store: dict[str, DynamicEntity],
    kind: str,
    headword: str,
    prefix: str,
) -> DynamicEntity:
    key = f"{kind}:{headword}"
    if key not in store:
        xml_id = f"{prefix}_{slugify(headword)[:60]}"
        store[key] = DynamicEntity(xml_id=xml_id, kind=kind, headword=headword)
    return store[key]


def resolve_predefined_entity(kind: str, surface: str) -> EntityDef | None:
    lowered = compact_spaces(surface).lower()
    for entity in ENTITY_DEFS:
        if entity.kind != kind:
            continue
        if entity.headword.lower() == lowered:
            return entity
        for pattern in entity.patterns:
            if re.fullmatch(pattern, surface, flags=re.I):
                return entity
        if kind == "placeName":
            for pattern in generate_toponym_patterns(entity.headword):
                if re.fullmatch(pattern, surface, flags=re.I):
                    return entity
    return None


def has_place_context(paragraph_text: str, start: int, end: int) -> bool:
    left = paragraph_text[max(0, start - 180):start]
    right = paragraph_text[end:end + 40]
    if PLACE_CONTEXT_CUE_RE.search(left):
        return True
    if PLACE_RIGHT_CONTEXT_RE.match(right):
        return True
    if PLACE_LIST_TAIL_RE.search(left):
        extended_left = paragraph_text[max(0, start - 260):start]
        if PLACE_CONTEXT_CUE_RE.search(extended_left):
            return True
    return False


def collect_matches(
    paragraph_text: str,
    dynamic_entities: dict[str, DynamicEntity],
    natasha_models: NatashaModels | None,
    place_forms: set[str],
) -> list[EntityMatch]:
    matches: list[EntityMatch] = []

    for entity in ENTITY_DEFS:
        patterns = list(entity.patterns)
        if entity.kind == "placeName":
            patterns.extend(generate_toponym_patterns(entity.headword))
        for pattern in patterns:
            compiled = re.compile(rf"(?<!\w)({pattern})(?!\w)")
            for match in compiled.finditer(paragraph_text):
                matches.append(
                    EntityMatch(
                        start=match.start(1),
                        end=match.end(1),
                        kind=entity.kind,
                        xml_id=entity.xml_id,
                        headword=entity.headword,
                        subtype="heuristic-dictionary",
                        priority=0,
                    )
                )

    if natasha_models is not None and Doc is not None:
        doc = Doc(paragraph_text)
        doc.segment(natasha_models.segmenter)
        doc.tag_ner(natasha_models.ner_tagger)
        tag_map = {"PER": "persName", "ORG": "orgName", "LOC": "placeName"}
        for span in doc.spans:
            kind = tag_map.get(span.type)
            if kind is None:
                continue
            surface = paragraph_text[span.start:span.stop]
            if kind == "persName":
                compact_surface = compact_spaces(surface)
                if ADJECTIVAL_PLACE_TOKEN_RE.fullmatch(compact_surface) and has_place_context(
                    paragraph_text, span.start, span.stop
                ):
                    predefined = resolve_predefined_entity("placeName", compact_surface)
                    if predefined is None:
                        entity = add_dynamic_entity(dynamic_entities, "placeName", compact_surface, "place")
                    else:
                        entity = DynamicEntity(predefined.xml_id, predefined.kind, predefined.headword)
                    matches.append(
                        EntityMatch(
                            start=span.start,
                            end=span.stop,
                            kind=entity.kind,
                            xml_id=entity.xml_id,
                            headword=entity.headword,
                            subtype="natasha-context-place",
                            priority=1,
                        )
                    )
                    continue
                headword = canonicalize_person(surface)
                if not headword:
                    continue
                if not is_valid_person_surface(headword):
                    continue
                entity = add_dynamic_entity(dynamic_entities, "persName", headword, "person")
            elif kind == "orgName":
                if not is_valid_org_surface(surface):
                    continue
                predefined = resolve_predefined_entity("orgName", compact_spaces(surface))
                if predefined is None:
                    continue
                entity = DynamicEntity(predefined.xml_id, predefined.kind, predefined.headword)
            else:
                if not is_valid_place_surface(surface, place_forms):
                    continue
                predefined = resolve_predefined_entity("placeName", compact_spaces(surface))
                if predefined is None:
                    continue
                entity = DynamicEntity(predefined.xml_id, predefined.kind, predefined.headword)
            matches.append(
                EntityMatch(
                    start=span.start,
                    end=span.stop,
                    kind=entity.kind,
                    xml_id=entity.xml_id,
                    headword=entity.headword,
                    subtype="natasha",
                    priority=1,
                )
            )

    for match in ADJECTIVAL_PLACE_TOKEN_RE.finditer(paragraph_text):
        if not has_place_context(paragraph_text, match.start(), match.end()):
            continue
        surface = compact_spaces(match.group(0))
        predefined = resolve_predefined_entity("placeName", surface)
        if predefined is None:
            entity = add_dynamic_entity(dynamic_entities, "placeName", surface, "place")
        else:
            entity = DynamicEntity(predefined.xml_id, predefined.kind, predefined.headword)
        matches.append(
            EntityMatch(
                start=match.start(),
                end=match.end(),
                kind=entity.kind,
                xml_id=entity.xml_id,
                headword=entity.headword,
                subtype="context-place-fallback",
                priority=2,
            )
        )

    for match in PERSON_TITLE_PATTERN.finditer(paragraph_text):
        headword = canonicalize_person(match.group(1))
        if not headword or not is_valid_person_surface(headword):
            continue
        entity = add_dynamic_entity(dynamic_entities, "persName", headword, "person")
        matches.append(
            EntityMatch(
                start=match.start(1),
                end=match.end(1),
                kind=entity.kind,
                xml_id=entity.xml_id,
                headword=entity.headword,
                subtype="title-fallback",
                priority=2,
            )
        )

    for match in FULL_NAME_WITH_PATRONYMIC_PATTERN.finditer(paragraph_text):
        headword = canonicalize_person(match.group(1))
        if not headword or not is_valid_person_surface(headword):
            continue
        entity = add_dynamic_entity(dynamic_entities, "persName", headword, "person")
        matches.append(
            EntityMatch(
                start=match.start(1),
                end=match.end(1),
                kind=entity.kind,
                xml_id=entity.xml_id,
                headword=entity.headword,
                subtype="regex-fallback",
                priority=2,
            )
        )

    matches.sort(key=lambda item: (item.start, item.priority, -(item.end - item.start), item.xml_id))

    resolved: list[EntityMatch] = []
    occupied_until = -1
    for item in matches:
        if item.start < occupied_until:
            continue
        resolved.append(item)
        occupied_until = item.end
    return resolved


def collect_date_matches(
    paragraph_text: str,
    occupied_ranges: list[tuple[int, int]],
    natasha_models: NatashaModels | None,
) -> list[DateMatch]:
    matches: list[DateMatch] = []

    if natasha_models is not None:
        for match in natasha_models.dates_extractor(paragraph_text):
            fact = match.fact
            year = getattr(fact, "year", None)
            if year is None or not year_in_range(int(year)):
                continue
            if preceded_by_number_sign(paragraph_text, match.start):
                continue
            start, end = match.start, match.stop
            if any(not (end <= left or start >= right) for left, right in occupied_ranges):
                continue
            month = getattr(fact, "month", None)
            day = getattr(fact, "day", None)
            when = f"{int(year):04d}"
            if month and day:
                when = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
            matches.append(DateMatch(start=start, end=end, when=when, subtype="natasha", priority=0))

    for match in FULL_DATE_RE.finditer(paragraph_text):
        year = int(match.group(3))
        if not year_in_range(year):
            continue
        if preceded_by_number_sign(paragraph_text, match.start()):
            continue
        start, end = match.span(0)
        if any(not (end <= left or start >= right) for left, right in occupied_ranges):
            continue
        day = int(match.group(1))
        month = MONTHS[match.group(2).lower()]
        when = f"{year:04d}-{month}-{day:02d}"
        matches.append(DateMatch(start=start, end=end, when=when, subtype="regex-fallback", priority=1))

    for match in YEAR_RE.finditer(paragraph_text):
        year = int(match.group(1))
        if not year_in_range(year):
            continue
        if preceded_by_number_sign(paragraph_text, match.start()):
            continue
        start, end = match.span(0)
        if any(not (end <= left or start >= right) for left, right in occupied_ranges):
            continue
        matches.append(DateMatch(start=start, end=end, when=str(year), subtype="regex-fallback", priority=1))

    matches.sort(key=lambda item: (item.start, item.priority, -(item.end - item.start)))
    resolved: list[DateMatch] = []
    occupied_until = -1
    for item in matches:
        if item.start < occupied_until:
            continue
        resolved.append(item)
        occupied_until = item.end
    return resolved


def build_paragraph(
    paragraph_text: str,
    seen_entities: dict[str, DynamicEntity],
    natasha_models: NatashaModels | None,
    place_forms: set[str],
) -> ET.Element:
    paragraph = ET.Element(tei("p"))
    last_child: ET.Element | None = None
    cursor = 0

    occupied_ranges: list[tuple[int, int]] = []
    matches = collect_matches(paragraph_text, seen_entities, natasha_models, place_forms)

    for start, end, _surface in ((m.start, m.end, paragraph_text[m.start:m.end]) for m in matches):
        occupied_ranges.append((start, end))

    date_matches = collect_date_matches(paragraph_text, occupied_ranges, natasha_models)

    marks: list[tuple[int, int, str, str | None, dict[str, str], int]] = []
    for item in matches:
        marks.append(
            (
                item.start,
                item.end,
                item.kind,
                item.xml_id,
                {"type": "auto", "subtype": item.subtype},
                item.priority,
            )
        )
    for item in date_matches:
        marks.append(
            (
                item.start,
                item.end,
                "date",
                None,
                {"when": item.when, "type": "auto", "subtype": item.subtype},
                item.priority,
            )
        )
    marks.sort(key=lambda item: (item[0], item[5], -(item[1] - item[0]), item[2]))

    used_until = -1
    for start, end, kind, xml_id, attrs, _priority in marks:
        if start < used_until:
            continue
        append_text(paragraph, last_child, paragraph_text[cursor:start])
        child = ET.SubElement(paragraph, tei(kind))
        if kind != "date":
            child.set("ref", f"#{xml_id}")
        for key, value in attrs.items():
            child.set(key, value)
        child.text = paragraph_text[start:end]
        last_child = child
        cursor = end
        used_until = end

    append_text(paragraph, last_child, paragraph_text[cursor:])
    return paragraph


def append_entity_lists(root: ET.Element, seen_entities: dict[str, DynamicEntity]) -> None:
    ordered: dict[str, list[DynamicEntity]] = {"persName": [], "placeName": [], "orgName": []}
    by_kind_id: dict[str, set[str]] = {"persName": set(), "placeName": set(), "orgName": set()}

    for entity in ENTITY_DEFS:
        key = f"{entity.kind}:{entity.headword}"
        if key in seen_entities:
            dynamic = DynamicEntity(entity.xml_id, entity.kind, entity.headword)
            if dynamic.xml_id not in by_kind_id[dynamic.kind]:
                ordered[dynamic.kind].append(dynamic)
                by_kind_id[dynamic.kind].add(dynamic.xml_id)

    for dynamic in seen_entities.values():
        if dynamic.kind not in ordered:
            continue
        if dynamic.xml_id in by_kind_id[dynamic.kind]:
            continue
        ordered[dynamic.kind].append(dynamic)
        by_kind_id[dynamic.kind].add(dynamic.xml_id)

    if not any(ordered.values()):
        return

    stand_off = ET.SubElement(root, tei("standOff"))

    if ordered["persName"]:
        list_person = ET.SubElement(stand_off, tei("listPerson"))
        for entity in ordered["persName"]:
            node = ET.SubElement(list_person, tei("person"))
            node.set(XML_ID, entity.xml_id)
            ET.SubElement(node, tei("persName")).text = entity.headword

    if ordered["placeName"]:
        list_place = ET.SubElement(stand_off, tei("listPlace"))
        for entity in ordered["placeName"]:
            node = ET.SubElement(list_place, tei("place"))
            node.set(XML_ID, entity.xml_id)
            ET.SubElement(node, tei("placeName")).text = entity.headword

    if ordered["orgName"]:
        list_org = ET.SubElement(stand_off, tei("listOrg"))
        for entity in ordered["orgName"]:
            node = ET.SubElement(list_org, tei("org"))
            node.set(XML_ID, entity.xml_id)
            ET.SubElement(node, tei("orgName")).text = entity.headword


def build_document(config: DocumentConfig, natasha_models: NatashaModels | None) -> ET.ElementTree:
    raw_text = config.source_txt.read_text(encoding="utf-8")
    place_forms = collect_place_forms(raw_text)

    root = ET.Element(tei("TEI"))
    root.set(f"{{{XML_NS}}}lang", "ru")

    tei_header = ET.SubElement(root, tei("teiHeader"))
    file_desc = ET.SubElement(tei_header, tei("fileDesc"))

    title_stmt = ET.SubElement(file_desc, tei("titleStmt"))
    ET.SubElement(title_stmt, tei("title")).text = config.title

    publication_stmt = ET.SubElement(file_desc, tei("publicationStmt"))
    ET.SubElement(publication_stmt, tei("p")).text = "Prepared locally from OCR-layer PDF for TEI conversion."

    source_desc = ET.SubElement(file_desc, tei("sourceDesc"))
    source_bibl = ET.SubElement(source_desc, tei("bibl"))
    ET.SubElement(source_bibl, tei("title")).text = config.title
    ET.SubElement(source_bibl, tei("idno"), type="report-id").text = str(config.report_id)
    ET.SubElement(source_bibl, tei("idno"), type="portal-url").text = config.report_url
    ET.SubElement(source_bibl, tei("idno"), type="portal-base").text = "https://govreport.sfu-kras.ru"
    if config.pdf_url:
        ET.SubElement(source_bibl, tei("ref"), target=config.pdf_url, type="pdf").text = config.pdf_url
    if config.pdf_original_url:
        ET.SubElement(source_bibl, tei("ref"), target=config.pdf_original_url, type="pdf-original").text = config.pdf_original_url
    ET.SubElement(source_bibl, tei("note"), type="archive").text = config.archive_note
    ET.SubElement(source_bibl, tei("note"), type="source").text = config.source
    ET.SubElement(source_bibl, tei("note"), type="text-type").text = config.text_type
    ET.SubElement(source_bibl, tei("extent")).text = f"{config.page_count} pages"
    ET.SubElement(source_bibl, tei("note"), type="portal-metadata").text = (
        "Metadata fields were copied from the report page at govreport.sfu-kras.ru."
    )

    encoding_desc = ET.SubElement(tei_header, tei("encodingDesc"))
    ner_note = "strict Natasha + rule-based fallback" if natasha_models is not None else "rule-based fallback only"
    ET.SubElement(encoding_desc, tei("p")).text = (
        "OCR text was first extracted from PDF into plain text with explicit [[PAGE_BREAK:N]] markers. "
        "TEI was then built from the intermediate text with rule-based markup for persons, places, "
        "organizations, and years in the range 1800-1917. Tables were reconstructed from line structure and "
        f"repeated row patterns in the plain text layer. Named-entity mode: {ner_note}."
    )

    profile_desc = ET.SubElement(tei_header, tei("profileDesc"))
    lang_usage = ET.SubElement(profile_desc, tei("langUsage"))
    ET.SubElement(lang_usage, tei("language"), ident="ru").text = "Russian"

    revision_desc = ET.SubElement(tei_header, tei("revisionDesc"))
    ET.SubElement(revision_desc, tei("change")).text = "Initial rule-based TEI conversion from OCR text."

    text_el = ET.SubElement(root, tei("text"))
    body = ET.SubElement(text_el, tei("body"))

    seen_entities: dict[str, DynamicEntity] = {}
    for entity in ENTITY_DEFS:
        # Add predefined entities lazily only if they are referenced.
        pass

    for page_no, page_text in split_pages(raw_text):
        ET.SubElement(body, tei("pb"), n=str(page_no))
        for kind, payload in iter_page_elements(page_text):
            if kind == "p":
                body.append(build_paragraph(payload, seen_entities, natasha_models, place_forms))
                continue

            table = ET.SubElement(body, tei("table"), type="ocr-text", subtype="row-pattern")
            for role, cells in build_table_rows(payload):
                row = ET.SubElement(table, tei("row"))
                if role == "label":
                    row.set("role", "label")
                for cell_text in cells:
                    cell = ET.SubElement(row, tei("cell"))
                    if cell_text:
                        cell.append(build_paragraph(cell_text, seen_entities, natasha_models, place_forms))

    # Keep only predefined entities that were actually referenced.
    referenced_ids = {
        element.attrib["ref"][1:]
        for element in root.iter()
        if "ref" in element.attrib and element.attrib["ref"].startswith("#")
    }
    for entity in ENTITY_DEFS:
        key = f"{entity.kind}:{entity.headword}"
        if entity.xml_id in referenced_ids:
            seen_entities[key] = DynamicEntity(entity.xml_id, entity.kind, entity.headword)

    append_entity_lists(root, seen_entities)
    return ET.ElementTree(root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TEI XML from OCR plain text files with page break markers.")
    parser.add_argument("--manifest", default="data/govreport_manifest.json", help="Manifest JSON with text paths and metadata.")
    parser.add_argument("--output-dir", default="tei_reports", help="Directory for generated TEI files.")
    parser.add_argument("--limit", type=int, default=0, help="Build only the first N documents.")
    parser.add_argument(
        "--use-natasha",
        action="store_true",
        help="Enable strict Natasha NER for persName/orgName/placeName/date (with rule-based fallback).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = load_document_configs(Path(args.manifest), Path(args.output_dir))
    if args.limit:
        configs = configs[: args.limit]
    natasha_models = get_natasha_models(args.use_natasha)
    if args.use_natasha and natasha_models is None:
        print("[warn] Natasha is unavailable; fallback to rule-based mode.", flush=True)

    for config in configs:
        tree = build_document(config, natasha_models=natasha_models)
        config.output_xml.parent.mkdir(parents=True, exist_ok=True)
        ET.indent(tree, space="  ")
        tree.write(config.output_xml, encoding="utf-8", xml_declaration=True)
        print(f"[ok] {config.output_xml}")


if __name__ == "__main__":
    main()

