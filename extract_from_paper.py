from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModelForTokenClassification.from_pretrained("outputs/scibert_quant_ie/checkpoint-120")

id2label = model.config.id2label

def merge_wordpieces(tokens):
    merged = []
    for tok in tokens:
        if tok.startswith("##") and merged:
            merged[-1] += tok[2:]
        elif tok not in ("[CLS]", "[SEP]", "[PAD]"):
            merged.append(tok)
    return merged

def extract_entities(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    pred_ids = torch.argmax(outputs.logits, dim=-1)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [id2label[i] for i in pred_ids]

    entities = []
    current_tokens = []
    current_label = None

    for tok, lab in zip(tokens, labels):
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue

        if lab.startswith("B-"):
            if current_tokens:
                entities.append((current_label, " ".join(merge_wordpieces(current_tokens))))
            current_label = lab[2:]
            current_tokens = [tok]

        elif lab.startswith("I-") and current_label == lab[2:]:
            current_tokens.append(tok)

        else:
            if current_tokens:
                entities.append((current_label, " ".join(merge_wordpieces(current_tokens))))
                current_tokens = []
                current_label = None

    if current_tokens:
        entities.append((current_label, " ".join(merge_wordpieces(current_tokens))))

    return entities

text = """
The response rate was 67% in 120 patients. Median PFS was 18 months.
"""

results = extract_entities(text)

print("Extracted entities:")
for label, value in results:
    print(f"{label}: {value}")