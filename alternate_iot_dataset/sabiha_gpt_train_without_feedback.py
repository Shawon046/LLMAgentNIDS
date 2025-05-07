# import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification , AdamW 
# from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

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

def without_feedback():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
        # Directory of training and test data
    # Load the full parquet dataset
    data_path = '/home/sabiha/course-6604/iot_network_flow_data_benign_mirai_bruteforce_spoofing.parquet'
    df_full = pd.read_parquet(data_path)
    # df_full = df_full.head(10000)
    # df_full = df_full[45000:60000]

    # print(df_full.head(10))

    # Label binarization: 0 = benign, 1 = attack
    df_full['label'] = df_full['label'].apply(lambda x: 0 if x.lower().startswith('benign') else 1)
    df_full = df_full.sample(frac=0.035, random_state=42)  # Sample 1.5% of the data

    # Train-test split (80%-20%)
    df, df_test = train_test_split(df_full, test_size=0.2, random_state=42, stratify=df_full['label'])

    # Create text column from all features except label
    df['text'] = df.drop(columns='label').astype(str).agg(', '.join, axis=1)
    df_test['text'] = df_test.drop(columns='label').astype(str).agg(', '.join, axis=1)


    # Quick overview
    # Overview
    print("Train shape:", df.shape)
    print("Test shape:", df_test.shape)
    print("\nTrain label distribution:\n", df['label'].value_counts(normalize=True).round(2))
    print("\nTest label distribution:\n", df_test['label'].value_counts(normalize=True).round(2))

    # Convert rows to text
    def row_to_text(row):
        return ', '.join([f"{col}: {row[col]}" for col in df.columns if col != 'label'])

    df['text'] = df.apply(row_to_text, axis=1)
    # Convert test rows to text
    df_test['text'] = df_test.apply(lambda row: ', '.join([f"{col}: {row[col]}" for col in df_test.columns if col != 'label']), axis=1)
    
    # Train-test split
    train_texts = df['text']
    train_labels = df['label']

    # 4. Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
    tokenizer.pad_token = tokenizer.eos_token  # GPT2 doesn't have a pad_token

    model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2)
    # model = GPT2ForSequenceClassification.from_pretrained('gpt2-xl', num_labels=2)
    model.config.pad_token_id = tokenizer.pad_token_id

    # 5. DataLoaders
    train_dataset = NetworkTrafficDataset(train_texts, train_labels, tokenizer)
    # val_dataset = NetworkTrafficDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=4)

    # 3. Prepare test DataLoader
    test_dataset = NetworkTrafficDataset(df_test['text'], df_test['label'], tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=4)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    scaler = torch.amp.GradScaler()  # AMP scaler
    losses = []

    # 6. Training Loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # epoch = 10
    epochs =3
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
        print(f"\nEpoch {epoch+1}: Train Accuracy = {train_acc:.4f} Loss = {loss.item():.4f}")
        losses.append(loss.item())
    # Save checkpoint
    torch.save(model.state_dict(), f"model_checkpoint_epoch_{epochs+1}.pt")

    plt.plot(losses, marker='o')
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"training_loss for {epochs} Epochs.png")
    plt.show()

    # model.eval()
    # test_preds, test_labels, prob_scores = [], [], []

    # with torch.no_grad():
    #     for batch in test_loader:
    #         batch = {k: v.to(device) for k, v in batch.items()}
    #         outputs = model(**batch)
    #         logits = outputs.logits
    #         probs = torch.nn.functional.softmax(logits, dim=-1)
    #         preds = torch.argmax(logits, dim=-1)
    #         test_preds.extend(preds.cpu().numpy())
    #         test_labels.extend(batch['labels'].cpu().numpy())
    #         prob_scores.extend(probs[:, 1].cpu().numpy())  # Probability of class 1 (Anomaly)

    # Compute metrics
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



if __name__ == "__main__":
    without_feedback()

