import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, AdamW, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# 1. Load dataset
df = pd.read_excel('/home/sabiha/course-6604/KDDTrain+.xlsx', nrows=10000)  # Adjust rows as needed

# 1. Load the test dataset
df_test = pd.read_excel('/home/sabiha/course-6604/KDDTest+.xlsx')


# 2. Preprocessing
df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
df = df.drop(['difficulty_score'], axis=1)  # Drop unused column

# 2. Preprocess test dataset
df_test['label'] = df_test['label'].apply(lambda x: 0 if x == 'normal' else 1)
df_test = df_test.drop(['difficulty_score'], axis=1)


# Convert rows to text
def row_to_text(row):
    return ', '.join([f"{col}: {row[col]}" for col in df.columns if col != 'label'])

df['text'] = df.apply(row_to_text, axis=1)
# Convert test rows to text
df_test['text'] = df_test.apply(lambda row: ', '.join([f"{col}: {row[col]}" for col in df_test.columns if col != 'label']), axis=1)


# Train-test split
# train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
train_texts = df['text']
train_labels = df['label']

# 3. Dataset class
class NetworkTrafficDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

# 4. Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
tokenizer.pad_token = tokenizer.eos_token  # GPT2 doesn't have a pad_token

model = AutoModelForSequenceClassification.from_pretrained('distilgpt2', num_labels=2)
model.config.pad_token_id = tokenizer.pad_token_id

# 5. DataLoaders
train_dataset = NetworkTrafficDataset(train_texts, train_labels, tokenizer)
# val_dataset = NetworkTrafficDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=4)

# 3. Prepare test DataLoader
test_dataset = NetworkTrafficDataset(df_test['text'], df_test['label'], tokenizer)
test_loader = DataLoader(test_dataset, batch_size=4)

# 6. Training Loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# epoch = 10
epochs = 10
for epoch in range(epochs):
    model.train()
    train_preds, train_true = [], []
    for batch in train_loader:
        batch = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Collect predictions for training accuracy
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        train_preds.extend(preds.cpu().numpy())
        train_true.extend(batch['labels'].cpu().numpy())

    # Training metrics
    train_acc = accuracy_score(train_true, train_preds)
    print(f"\nEpoch {epoch+1}: Train Accuracy = {train_acc:.4f}")


    # # Validation
    # model.eval()
    # val_preds, val_labels_list = [], []
    # with torch.no_grad():
    #     for batch in val_loader:
    #         batch = {key: val.to(device) for key, val in batch.items()}
    #         outputs = model(**batch)
    #         logits = outputs.logits
    #         preds = torch.argmax(logits, dim=-1)
    #         val_preds.extend(preds.cpu().numpy())
    #         val_labels_list.extend(batch['labels'].cpu().numpy())

    # acc = accuracy_score(val_labels_list, val_preds)
    # f1 = f1_score(val_labels_list, val_preds)
    # precision = precision_score(val_labels_list, val_preds)
    # recall = recall_score(val_labels_list, val_preds)

    # print(f"Epoch {epoch+1}: Accuracy={acc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

model.eval()
test_preds, test_labels_list = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        test_preds.extend(preds.cpu().numpy())
        test_labels_list.extend(batch['labels'].cpu().numpy())

# 8. Metrics
r2 = r2_score(test_labels_list, test_preds)
acc = accuracy_score(test_labels_list, test_preds)
f1 = f1_score(test_labels_list, test_preds)
recall = recall_score(test_labels_list, test_preds)
precision_anomaly = precision_score(test_labels_list, test_preds, pos_label=1)
precision_benign = precision_score(test_labels_list, test_preds, pos_label=0)
tn, fp, fn, tp = confusion_matrix(test_labels_list, test_preds).ravel()
fpr = fp / (fp + tn)

# 9. Results
print("\n--- Test Set Results ---")
print(f"Accuracy           : {acc:.4f}")
print(f"Test R² Score      : {r2:.4f}")
print(f"Precision (Benign) : {precision_benign:.4f}")
print(f"Precision (Anomaly): {precision_anomaly:.4f}")
print(f"Recall             : {recall:.4f}")
print(f"F1 Score           : {f1:.4f}")
print(f"FPR (False Positive Rate): {fpr:.6f}")