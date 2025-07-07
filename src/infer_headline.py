import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sheepdog import RobertaClassifier

import torch
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer

def load_model(checkpoint_path, device):
    model = RobertaClassifier(n_classes=4)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_headline(headline, model, tokenizer, device, max_len=512):
    encoding = tokenizer.encode_plus(
        headline,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        output, binary_output = model(input_ids, attention_mask)
        pred_4class = torch.argmax(output, dim=1).item()
        pred_binary = torch.argmax(binary_output, dim=1).item()
    return pred_4class, pred_binary

if __name__ == "__main__":
    checkpoint_path = "checkpoints/politifact_iter0.m"  # Update as needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = load_model(checkpoint_path, device)

    headline = input("Enter a headline to classify: ")
    pred_4class, pred_binary = predict_headline(headline, model, tokenizer, device)

    print(f"4-class prediction: {pred_4class}")
    print(f"Binary prediction: {pred_binary}")