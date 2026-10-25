"""Sentence -> ARPAbet word/phone structure via g2p_en (the demo path; the
evaluated pipeline uses speechocean762's own canonical phones).
Moved out of app.py so the Gradio demo and the local app share it."""
import re


def text_to_words(text: str, g2p):
    """Sentence -> (words, unresolved).

    `words` is [{'text': str, 'phones': [ARPAbet...]}, ...] for every word
    g2p_en resolved cleanly. `unresolved` lists the raw word substrings it
    didn't: either the regex word count didn't match g2p's own tokenization
    (numerals, unusual punctuation -- can't safely pair them up positionally)
    or a specific word came back with zero phones. Returning this instead of
    silently dropping mismatches means the caller can reject the whole
    sentence explicitly, rather than quietly scoring a shorter sentence than
    the one the speaker was shown.
    """
    phones_flat = g2p(text)
    groups, current = [], []
    for tok in phones_flat:
        if tok == " ":
            if current:
                groups.append(current)
                current = []
        elif re.match(r"^[A-Z]+[0-2]?$", tok):
            current.append(tok)
    if current:
        groups.append(current)

    word_texts = re.findall(r"[A-Za-z']+", text)
    if len(word_texts) != len(groups):
        return [], word_texts  # tokenization mismatch -- can't safely zip these up

    words, unresolved = [], []
    for w, phones in zip(word_texts, groups):
        if phones:
            words.append({"text": w.upper(), "phones": phones})
        else:
            unresolved.append(w)
    return words, unresolved
