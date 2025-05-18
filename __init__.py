import torch
import os
from torch import nn
from transformers import AutoTokenizer
import pdfplumber


def extract_text_from_pdf(pdf_path):
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"
    return all_text


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def tokenize_text(text, tokenizer, max_length=512):
    tokens = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return tokens


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def encode_text(text, tokenizer, max_length=512):
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return encoded["input_ids"]


import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Magi(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, num_classes=10, max_len=512):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):  # x: [batch_size, seq_len]
        seq_len = x.size(1)
        x = self.token_embedding(x) + self.pos_embedding[:, :seq_len, :]  # [batch, seq_len, d_model]

        x = x.permute(1, 0, 2)  # Transformer expects: [seq_len, batch_size, d_model]
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Mean-pool over sequence

        logits = self.classifier(x)
        return logits


vocab_size = tokenizer.vocab_size
model = Magi(vocab_size=vocab_size, num_classes=10).to(device)

text = extract_text_from_pdf("resume.pdf")
input_ids = encode_text(text, tokenizer).to(device)

logits = model(input_ids)
pred = torch.argmax(logits, dim=1)

print("Predicted class:", pred.item())

from transformers import AutoTokenizer, AutoModelForCausalLM

sample = {
    "input": "user prompt",
    "output": "model response",
}

msg = "This is a simple string to text if the phi-3 tokenizer is functioning properly before I implement it with Magi"
model_id = "microsoft/phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32  # or bfloat16 if you use GPU
)

model.to("cuda" if torch.cuda.is_available() else "cpu")

inputs = tokenizer(msg, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        temperature=0.7,
        top_p=0.9
    )

# Decode and print
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
