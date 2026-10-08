"""Off-the-shelf detector: spaCy NER + regexes (Presidio-like default coverage)."""
import re, spacy
from common import resolve
MAP = {'PERSON': 'PERSON', 'ORG': 'ORG', 'GPE': 'LOC', 'LOC': 'LOC', 'FAC': 'LOC', 'NORP': 'DEM',
       'DATE': 'DATETIME', 'TIME': 'DATETIME', 'MONEY': 'QUANTITY', 'QUANTITY': 'QUANTITY', 'PERCENT': 'QUANTITY'}
REGEX = [(re.compile(r"\b\d{1,6}/\d{2}\b"), 'CODE'),
         (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), 'CODE'),
         (re.compile(r"\+?\d[\d ()-]{7,}\d"), 'CODE')]
_nlp = None
def nlp(model):
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(model, exclude=['lemmatizer', 'parser', 'tagger', 'attribute_ruler', 'senter'])
        _nlp.max_length = 3_000_000
    return _nlp
def detect_many(texts, model='en_core_web_lg', batch=16):
    out = []
    for doc, text in zip(nlp(model).pipe(texts, batch_size=batch), texts):
        ms = [dict(start=e.start_char, end=e.end_char, type=MAP[e.label_]) for e in doc.ents if e.label_ in MAP]
        for rx, t in REGEX:
            ms += [dict(start=m.start(), end=m.end(), type=t) for m in rx.finditer(text)]
        ms = resolve(ms)
        for m in ms:
            m['surface'] = text[m['start']:m['end']]
            m['ent'] = (m['type'], ' '.join(m['surface'].lower().split()))  # within-doc grouping by surface
        out.append(ms)
    return out
