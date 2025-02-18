import os
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("/model")
model = AutoModelForSeq2SeqLM.from_pretrained("/model")

INPUT_TEXT = os.getenv("input_text", "Default input text").strip()

input_ids = tokenizer(
  f"Paraphrase: {INPUT_TEXT}",
  return_tensors="pt",
  max_length=128,
  truncation=True
).input_ids

outputs = model.generate(
  input_ids,
  num_return_sequences=3,
  max_length=128,
  do_sample=True,
  top_k=50,
  top_p=0.95,
  temperature=1.1
)

paraphrases = [
  tokenizer.decode(output, skip_special_tokens=True) for output in outputs
]

os.makedirs("/outputs", exist_ok=True)

with open("/outputs/result.json", "w") as f:
  json.dump({"input_text": INPUT_TEXT, "paraphrases": paraphrases}, f, indent=2)