import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# Sample Contract Texts
contracts = [
    "Vendor agrees to deliver goods within 10 days and maintain a quality score above 4.5.",
    "Vendor must comply with all safety regulations and deliver goods within 15 days.",
    "Vendor to provide monthly reports on quality and delivery timelines."
]

# Initialize NLP model
nlp = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Extract entities from contract texts
entities = [nlp(contract) for contract in contracts]
print("Extracted Entities:", entities)

# Plot entity extraction
entity_labels = []
entity_values = []
for entity in entities:
    for e in entity:
        entity_labels.append(e['word'])
        entity_values.append(e['score'])

plt.bar(entity_labels, entity_values, color='blue')
plt.xlabel('Entity Labels')
plt.ylabel('Confidence Score')
plt.title('NLP Contract Analytics')
plt.show()
