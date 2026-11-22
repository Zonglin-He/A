"""Deterministic metric-value extraction for SpatialRGPT quantitative answers."""
import math
import re
import unicodedata


NUMBER = r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[+-]?\.\d+'
UNITS = {
    'meters': 1, 'meter': 1, 'metres': 1, 'metre': 1, 'm': 1,
    'centimeters': .01, 'centimeter': .01, 'centimetres': .01, 'centimetre': .01, 'cm': .01,
    'millimeters': .001, 'millimeter': .001, 'millimetres': .001, 'millimetre': .001, 'mm': .001,
    'kilometers': 1000, 'kilometer': 1000, 'kilometres': 1000, 'kilometre': 1000, 'km': 1000,
    'inches': .0254, 'inch': .0254, 'in': .0254, '"': .0254,
    'feet': .3048, 'foot': .3048, 'ft': .3048, "'": .3048,
    'yards': .9144, 'yard': .9144, 'yd': .9144,
}
UNIT = '|'.join(re.escape(text) for text in sorted(UNITS,key=len,reverse=True))
RANGE = re.compile(rf'(?<![\w.])({NUMBER})\s*(?:-|–|—|to|and)\s*({NUMBER})\s*({UNIT})(?!\w)')
TOKEN = re.compile(rf'(?<![\w.])({NUMBER})\s*({UNIT})(?!\w)')
WORDS = dict(zero=0,one=1,two=2,three=3,four=4,five=5,six=6,seven=7,eight=8,nine=9,ten=10)
WORD_VALUE = re.compile(rf'\b({"|".join(WORDS)})\s+({UNIT})(?!\w)')


def valid(value):
    return value if value is not None and math.isfinite(value) and value>0 else None


def nfloat(number):
    return float(number.replace(',',''))


def metric(text, *, bare_meters=False):
    if not isinstance(text,str):
        return None
    text = unicodedata.normalize('NFKC', text.casefold()).replace('’',"'").replace('′',"'").replace('″','"')
    text = re.sub(r'<(?:eos|pad|end_of_turn)>','',text).strip()
    work = list(text)
    values = []
    for m in RANGE.finditer(text):
        low, high, unit = m.groups()
        values.append((nfloat(low)+nfloat(high))/2*UNITS[unit])
        work[m.start():m.end()] = ' '*(m.end()-m.start())
    remaining = ''.join(work)
    matches = list(TOKEN.finditer(remaining))
    k = 0
    while k < len(matches):
        m = matches[k]
        v, u = m.groups()
        value = nfloat(v)*UNITS[u]
        if k+1 < len(matches) and UNITS[u] == .3048:
            nxt = matches[k+1]
            between = remaining[m.end():nxt.start()].strip()
            if UNITS[nxt[2]] == .0254 and between in ('','and',',','-'):
                value += nfloat(nxt[1])*.0254
                k += 1
        values.append(value)
        k += 1
    if not values:
        values = [WORDS[m[1]]*UNITS[m[2]] for m in WORD_VALUE.finditer(remaining)]
    if not values and bare_meters:
        exact = re.fullmatch(rf'(?:about |approximately |~\s*)?({NUMBER})[.\s]*',text)
        return valid(nfloat(exact[1])) if exact else None
    if not values or min(values)<=0:
        return None
    # Permit equivalent rounded unit renderings, without silently choosing conflicting values.
    first = values[0]
    if any(abs(first-v)>max(.005,.02*first) for v in values[1:]):
        return None
    return valid(first)
