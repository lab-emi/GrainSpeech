"""
Text-to-sequence using the reduced symbol set (symbols_exp).
Drop-in replacement for text/__init__.py for the experimental model.
"""

import re
from text import cleaners
from text.symbols_exp import symbols


_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")


def text_to_sequence(text, cleaner_names):
    sequence = []
    while len(text):
        m = _curly_re.match(text)
        if not m:
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)
    return sequence


def sequence_to_text(sequence):
    result = ""
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            if len(s) > 1 and s[0] == "@":
                s = "{%s}" % s[1:]
            result += s
    return result.replace("}{", " ")


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(syms):
    return [_symbol_to_id[s] for s in syms if s in _symbol_to_id and s != "_" and s != "~"]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(["@" + s for s in text.split()])
