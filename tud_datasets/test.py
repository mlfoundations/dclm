import torch
from transformers import BertTokenizer
from bert_model import BertForQualityRegression
from semantic_score import text_dict

model_path = "./quality_regression_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForQualityRegression.from_pretrained(model_path)

model.eval()

results = {}
for category, texts in text_dict.items():
    predicted_scores = []
    for text in texts:
          inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
          with torch.no_grad():
              outputs = model(**inputs)

          predicted_score = torch.sigmoid(outputs["logits"]).squeeze().item()
          predicted_scores.append(predicted_score)
    results[category] = predicted_scores

print(results)
