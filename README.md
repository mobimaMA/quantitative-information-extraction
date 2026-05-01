# Quantitative Information Extraction from Scientific Text
Starter package prepared for the project status through April 1, 2026.

## What is included
- `data_prep.py`: converts token-labeled JSONL into Hugging Face friendly splits
- `baseline_extraction.py`: regex/spaCy baseline for numeric expressions and nearby context
- `transformer_training.py`: fine-tuning script for a token-classification model
- `requirements.txt`: minimal dependencies
- `sample_data/`: tiny toy example for sanity checking

## Suggested workflow
1. Put your annotated data into JSONL format:
   - one document per line
   - fields: `tokens`, `labels`
2. Run:
   - `python data_prep.py --input data/annotated.jsonl --output_dir data/processed`
   - `python baseline_extraction.py --input_text_file sample_abstracts.txt`
   - `python transformer_training.py --train data/processed/train.jsonl --dev data/processed/dev.jsonl --test data/processed/test.jsonl`
3. Compare baseline and transformer outputs with precision / recall / F1.

## Recommended label set
Use BIO labels such as:
- B-QUANT
- I-QUANT
- B-UNIT
- I-UNIT
- B-METRIC
- I-METRIC
- O

You can expand this later to include:
- B-CONTEXT / I-CONTEXT
- B-SAMPLE_SIZE / I-SAMPLE_SIZE
- B-DURATION / I-DURATION

## Notes
- This package is a clean starting point, not a final polished experiment pipeline.
- The baseline favors transparency over complexity.
- The transformer script defaults to `allenai/scibert_scivocab_uncased`, which matches the scientific-text focus in the proposal.
