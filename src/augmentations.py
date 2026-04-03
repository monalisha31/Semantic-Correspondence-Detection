import random
import re
from typing import Dict

CLASS_SYNONYMS = {
    "person": ["individual", "human", "agent", "actor", "user"],
    "family": ["household", "group", "unit", "clan"],
    "address": ["location", "place", "site", "position"],
    "name": ["label", "title", "identifier", "designation"],
    "employee": ["worker", "staff", "member", "associate"],
    "company": ["organization", "firm", "enterprise", "corporation"],
    "product": ["item", "article", "goods", "commodity"],
    "order": ["purchase", "transaction", "request", "booking"],
    "customer": ["client", "buyer", "consumer", "patron"],
    "account": ["profile", "record", "entry", "registry"],
    "document": ["file", "record", "artifact", "resource"],
    "message": ["notification", "alert", "communication", "signal"],
    "event": ["occurrence", "incident", "action", "happening"],
    "task": ["job", "assignment", "activity", "work"],
    "project": ["initiative", "program", "venture", "undertaking"],
    "component": ["part", "element", "module", "unit"],
    "container": ["holder", "wrapper", "collection", "repository"],
    "node": ["vertex", "element", "point", "entity"],
    "edge": ["link", "connection", "arc", "relationship"],
    "parent": ["ancestor", "superior", "owner", "root"],
    "child": ["descendant", "subordinate", "offspring", "leaf"],
    "type": ["kind", "category", "class", "variety"],
    "value": ["amount", "quantity", "data", "content"],
    "status": ["state", "condition", "phase", "stage"],
    "description": ["detail", "summary", "info", "specification"],
    "date": ["timestamp", "time", "moment", "instant"],
    "id": ["identifier", "key", "code", "number"],
    "email": ["mail", "emailAddress", "contact"],
    "phone": ["telephone", "phoneNumber", "contact"],
    "age": ["years", "lifetime"],
    "price": ["cost", "amount", "fee", "charge"],
    "size": ["dimension", "magnitude", "extent", "capacity"],
    "color": ["colour", "hue", "shade", "tint"],
    "weight": ["mass", "heaviness", "load"],
}

TYPE_GENERALIZATIONS = {
    "EString": ["String", "EString", "string", "java.lang.String"],
    "EInt": ["Integer", "EInt", "int", "EInteger", "java.lang.Integer"],
    "EBoolean": ["Boolean", "EBoolean", "bool", "java.lang.Boolean"],
    "EFloat": ["Float", "EFloat", "float", "EDouble", "Number"],
    "EDouble": ["Double", "EDouble", "double", "EFloat", "Number"],
    "EDate": ["Date", "EDate", "Timestamp", "DateTime", "java.util.Date"],
    "ELong": ["Long", "ELong", "long", "EInt", "java.lang.Long"],
    "EByte": ["Byte", "EByte", "byte"],
    "EChar": ["Char", "EChar", "char", "Character"],
}

CONTAINMENT_SYNONYMS = {
    "Composition": ["Composition", "containment", "owns"],
    "Association": ["Association", "reference", "links"],
}


def camel_to_snake(name: str) -> str:
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def snake_to_camel(name: str) -> str:
    parts = name.split('_')
    return parts[0].lower() + ''.join(w.title() for w in parts[1:])


def get_synonym(word: str) -> str:
    lower = word.lower()
    for key, synonyms in CLASS_SYNONYMS.items():
        if lower == key or lower in synonyms:
            candidates = [key] + synonyms
            candidates = [c for c in candidates if c.lower() != lower]
            if candidates:
                chosen = random.choice(candidates)
                if word[0].isupper():
                    return chosen[0].upper() + chosen[1:]
                return chosen
    return word


def generalize_type(type_name: str) -> str:
    for canonical, variants in TYPE_GENERALIZATIONS.items():
        if type_name in variants or type_name == canonical:
            return random.choice(variants)
    return type_name


def case_variation(text: str, prob: float = 0.2) -> str:
    tokens = text.split()
    for i in range(len(tokens)):
        if random.random() < prob:
            if '_' in tokens[i] and not tokens[i].startswith('(') and not tokens[i].endswith(')'):
                tokens[i] = snake_to_camel(tokens[i])
            elif (any(c.isupper() for c in tokens[i][1:]) and
                  not tokens[i].startswith('E') and
                  tokens[i] not in ('Class:', 'Supertypes:', 'Attributes:', 'References:', 'None', 'None.')):
                tokens[i] = camel_to_snake(tokens[i])
    return " ".join(tokens)


def synonym_replace(text: str, prob: float = 0.2) -> str:
    m = re.match(r"(Class: )(\w+)(\..*)", text, re.DOTALL)
    if m and random.random() < prob:
        new_name = get_synonym(m.group(2))
        text = m.group(1) + new_name + m.group(3)

    def replace_attr_names(match):
        label = match.group(1)
        content = match.group(2)
        if content.strip() in ("None", "None."):
            return match.group(0)
        items = [i.strip() for i in content.split(',')]
        new_items = []
        for item in items:
            m2 = re.match(r"(\w+)\s*\((.*?)\)", item)
            if m2 and random.random() < prob:
                new_name = get_synonym(m2.group(1))
                new_items.append(f"{new_name} ({m2.group(2)})")
            else:
                new_items.append(item)
        return f"{label}: {', '.join(new_items)}"

    text = re.sub(r"(Attributes): (.*?)(?=\. [A-Z]|$|\.$)", replace_attr_names, text)
    return text


def type_generalize(text: str, prob: float = 0.2) -> str:
    def replace_types(match):
        prefix = match.group(1)
        type_name = match.group(2)
        suffix = match.group(3)
        if random.random() < prob:
            new_type = generalize_type(type_name)
            return f"{prefix}{new_type}{suffix}"
        return match.group(0)

    text = re.sub(r"(\w+ \()(\w+)(\))", replace_types, text)
    return text


def shuffle_structure_lists(text: str, prob: float = 0.3) -> str:
    if random.random() > prob:
        return text

    def process_section(match):
        label = match.group(1)
        content = match.group(2)
        if content.strip() in ("None", "None."):
            return f"{label}: {content}"
        items = [i.strip() for i in content.split(',')]
        if len(items) > 1:
            random.shuffle(items)
        return f"{label}: {', '.join(items)}"

    pattern = r"(Attributes|References): (.*?)(?=\. [A-Z]|$|\.$)"
    return re.sub(pattern, process_section, text)


def rename_structural_elements(text: str, prob: float = 0.3) -> str:
    if random.random() > prob:
        return text

    def process_identifiers(match):
        label = match.group(1)
        content = match.group(2)
        if content.strip() in ("None", "None."):
            return match.group(0)
        items = [i.strip() for i in content.split(',')]
        new_items = []
        for item in items:
            m = re.match(r"(\w+)(\s*(?:\(.*?\)|\->.*?))", item)
            if m and random.random() < 0.5:
                name = m.group(1)
                rest = m.group(2)
                if '_' in name:
                    name = snake_to_camel(name)
                elif any(c.isupper() for c in name[1:]):
                    name = camel_to_snake(name)
                new_items.append(f"{name}{rest}")
            else:
                new_items.append(item)
        return f"{label}: {', '.join(new_items)}"

    text = re.sub(r"(Attributes): (.*?)(?=\. [A-Z]|$|\.$)", process_identifiers, text)
    text = re.sub(r"(References): (.*?)(?=\. [A-Z]|$|\.$)", process_identifiers, text)
    return text


def delete_random_token(text: str, prob: float = 0.1) -> str:
    tokens = text.split()
    if len(tokens) <= 3:
        return text
    protected = {'Class:', 'Supertypes:', 'Attributes:', 'References:'}
    new_tokens = [t for t in tokens if t in protected or random.random() > prob]
    if not new_tokens:
        return text
    return " ".join(new_tokens)


def drop_optional_feature(text: str, prob: float = 0.15) -> str:
    if random.random() > prob:
        return text

    def drop_from_section(match):
        label = match.group(1)
        content = match.group(2)
        if content.strip() in ("None", "None."):
            return match.group(0)
        items = [i.strip() for i in content.split(',')]
        if len(items) <= 1:
            return match.group(0)
        drop_idx = random.randint(0, len(items) - 1)
        items.pop(drop_idx)
        return f"{label}: {', '.join(items)}"

    if random.random() < 0.5:
        text = re.sub(r"(Attributes): (.*?)(?=\. [A-Z]|$|\.$)", drop_from_section, text)
    else:
        text = re.sub(r"(References): (.*?)(?=\. [A-Z]|$|\.$)", drop_from_section, text)
    return text


def augment(text: str, config: dict = None) -> str:
    if config is None:
        config = {}

    text = synonym_replace(text, prob=config.get('synonym_replace_prob', 0.2))
    text = type_generalize(text, prob=config.get('type_generalize_prob', 0.2))
    text = shuffle_structure_lists(text, prob=config.get('shuffle_lists_prob', 0.3))
    text = case_variation(text, prob=config.get('case_variation_prob', 0.2))
    text = rename_structural_elements(text, prob=config.get('rename_elements_prob', 0.3))
    text = drop_optional_feature(text, prob=config.get('drop_optional_prob', 0.15))
    text = delete_random_token(text, prob=config.get('delete_token_prob', 0.1))

    return text


def transform_class_struct(cls_obj: dict, transform_type: str) -> dict:
    import copy
    cls = copy.deepcopy(cls_obj)

    if transform_type == "rename":
        cls['name'] = get_synonym(cls['name'])
        for attr in cls.get('attributes', []):
            attr['name'] = get_synonym(attr['name'])
        for ref in cls.get('references', []):
            ref['name'] = get_synonym(ref['name'])

    elif transform_type == "reorder":
        random.shuffle(cls.get('attributes', []))
        random.shuffle(cls.get('references', []))

    elif transform_type == "type_abstract":
        for attr in cls.get('attributes', []):
            attr['type'] = generalize_type(attr['type'])

    elif transform_type == "add_optional":
        optional_attrs = [
            {"name": "description", "type": "EString"},
            {"name": "createdAt", "type": "EDate"},
            {"name": "isActive", "type": "EBoolean"},
            {"name": "priority", "type": "EInt"},
            {"name": "label", "type": "EString"},
        ]
        cls.setdefault('attributes', []).append(random.choice(optional_attrs))

    elif transform_type == "flatten_inherit":
        cls['supertypes'] = []

    elif transform_type == "case_switch":
        cls['name'] = camel_to_snake(cls['name']) if any(
            c.isupper() for c in cls['name'][1:]
        ) else snake_to_camel(cls['name'])
        for attr in cls.get('attributes', []):
            attr['name'] = camel_to_snake(attr['name']) if any(
                c.isupper() for c in attr['name'][1:]
            ) else snake_to_camel(attr['name'])

    elif transform_type == "combined":
        transforms = random.sample(
            ["rename", "reorder", "type_abstract", "case_switch"],
            k=random.randint(2, 3)
        )
        for t in transforms:
            cls = transform_class_struct(cls, t)

    return cls
