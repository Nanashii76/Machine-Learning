import pandas as pd
import numpy as np
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Verifica GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Carregamento dos dados
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

# 2. Limpeza básica
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

train_df['clean_text'] = train_df['text'].apply(clean_text)
test_df['clean_text'] = test_df['text'].apply(clean_text)

# 3. Tokenização
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 4. Dataset customizado
class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=128)
        self.labels = labels.tolist() if labels is not None else None

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

# 5. Split treino/validação
X_train, X_val, y_train, y_val = train_test_split(
    train_df['clean_text'], train_df['target'], test_size=0.2, stratify=train_df['target'], random_state=42
)

train_dataset = TextDataset(X_train, y_train)
val_dataset = TextDataset(X_val, y_val)
test_dataset = TextDataset(test_df['clean_text'])

# 6. Modelo
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# 7. Treinamento
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    # evaluation_strategy="epoch",
    save_strategy="no",
    logging_dir='./logs',
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"f1": f1_score(labels, preds)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# 8. Avaliação
eval_results = trainer.evaluate()
print(f"Validation F1-score: {eval_results['eval_f1']:.4f}")

# 9. Previsões no teste
test_preds_logits = trainer.predict(test_dataset).predictions
test_preds = np.argmax(test_preds_logits, axis=1)

# 10. Submissão
submission = pd.DataFrame({'id': test_df['id'], 'target': test_preds})
submission.to_csv('submission.csv', index=False)
