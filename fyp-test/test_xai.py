#!/usr/bin/env python
"""Test script for XAI token importance fix"""

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from view.xai import get_token_importance

# Load model
print("Loading model...")
model = DistilBertForSequenceClassification.from_pretrained('farzirfn/fake-news-distilbert')
tokenizer = DistilBertTokenizer.from_pretrained('farzirfn/fake-news-distilbert')
device = torch.device('cpu')
model.to(device)
model.eval()

# Test text
test_text = 'Breaking news: Scientists discover new species in Amazon rainforest'

# Get token importance
print("Calculating token importance...")
tokens, importances = get_token_importance(model, tokenizer, None, device, test_text)
print('✅ Token importance calculated successfully!')
print(f'\nTokens ({len(tokens)}): {tokens}')
print(f'\nImportances: {[f"{x:.4f}" for x in importances]}')
