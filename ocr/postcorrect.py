"""Optional lexicon post-correction for handwriting lines.

Conservative on purpose: only lower-case alphabetic words of four or more letters that are not in the
English frequency list are changed, and only to a word one edit away. Names, numbers, codes and
anything with capitals or punctuation inside are left alone. Uses the frequency dictionary shipped
with symspellpy (MIT licence); no network.
"""
from __future__ import annotations

import os
import re

_SYM = None
_WORD = re.compile(r"^([^A-Za-z]*)([a-z]{4,})([^A-Za-z]*)$")


def _sym():
    global _SYM
    if _SYM is None:
        import symspellpy
        _SYM = symspellpy.SymSpell(max_dictionary_edit_distance=1, prefix_length=7)
        path = os.path.join(os.path.dirname(symspellpy.__file__), "frequency_dictionary_en_82_765.txt")
        _SYM.load_dictionary(path, term_index=0, count_index=1)
    return _SYM


def correct_line(text: str) -> str:
    """The line with out-of-lexicon lower-case words replaced by their closest lexicon word."""
    import symspellpy
    sym, out = _sym(), []
    for tok in text.split(" "):
        m = _WORD.match(tok)
        if m and m.group(2) not in sym.words:
            hits = sym.lookup(m.group(2), symspellpy.Verbosity.TOP, max_edit_distance=1)
            if hits:
                tok = m.group(1) + hits[0].term + m.group(3)
        out.append(tok)
    return " ".join(out)
