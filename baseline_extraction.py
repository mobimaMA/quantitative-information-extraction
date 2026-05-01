import argparse
import re
from pathlib import Path

import spacy

QUANT_PATTERN = re.compile(
    r"""(?:
        \b\d+(?:\.\d+)?\s?%              
        |
        \b\d+(?:\.\d+)?\s?(?:mg|g|kg|ml|l|cm|mm|nm|um|μm|days?|weeks?|months?|years?)\b
        |
        \bn\s?=\s?\d+\b
        |
        \b\d+(?:\.\d+)?(?:\s?-\s?\d+(?:\.\d+)?)?\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        raise SystemExit(
            "spaCy English model not installed. Run: python -m spacy download en_core_web_sm"
        )


def extract_from_sentence(sent):
    text = sent.text
    matches = []
    for m in QUANT_PATTERN.finditer(text):
        start_char, end_char = m.span()
        quant_text = m.group().strip()
        span = sent.doc.char_span(sent.start_char + start_char, sent.start_char + end_char)
        left_context = ""
        right_context = ""

        if span is not None:
            left_tokens = []
            for token in sent:
                if token.i < span.start and token.pos_ in {"NOUN", "PROPN", "ADJ"}:
                    left_tokens.append(token.text)
            right_tokens = []
            for token in sent:
                if token.i >= span.end and token.pos_ in {"NOUN", "PROPN", "ADJ"}:
                    right_tokens.append(token.text)

            left_context = " ".join(left_tokens[-4:])
            right_context = " ".join(right_tokens[:4])

        matches.append(
            {
                "sentence": text,
                "quantity": quant_text,
                "left_context": left_context,
                "right_context": right_context,
            }
        )
    return matches


def extract_quantities(text):
    nlp = load_nlp()
    doc = nlp(text)
    outputs = []
    for sent in doc.sents:
        outputs.extend(extract_from_sentence(sent))
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_text_file", required=True, help="Plain text file to analyze")
    args = parser.parse_args()

    text = Path(args.input_text_file).read_text(encoding="utf-8")
    results = extract_quantities(text)

    if not results:
        print("No quantitative expressions found.")
        return

    for i, item in enumerate(results, start=1):
        print("=" * 80)
        print(f"Item {i}")
        print(f"Sentence     : {item['sentence']}")
        print(f"Quantity     : {item['quantity']}")
        print(f"Left context : {item['left_context']}")
        print(f"Right context: {item['right_context']}")


if __name__ == "__main__":
    main()
