import pandas as pd
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import turtle
import nltk
from nltk.tokenize import sent_tokenize
import time

# Download NLTK sentence tokenizer data
try:
    sent_tokenize("This is a test. Is it?")
except LookupError:
    nltk.download('punkt')

# Data Preprocessing
#---------------------
def clean_text(text):
    if isinstance(text, str):
        sentences = sent_tokenize(text)
        meaningful = [s.strip() for s in sentences if len(s.strip().split()) >= 5]
        cleaned = ' '.join(meaningful)
        cleaned = re.sub(r'\W', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip().lower()
    return ''

# LIAR Dataset Specific Loading
def encode_liar_label(label):
    if label in ['true', 'mostly-true']:
        return 1.0  # True
    elif label in ['false', 'pants-fire']:
        return 0.0  # Fake
    else:
        return 0.5  # Uncertain/Mixed

# Added another dataset to make results accurate
def load_and_prepare_liar_dataset(train_path, valid_path, test_path):
    train_df = pd.read_csv(train_path, sep='\t', header=None, names=['id', 'label', 'statement', 'subject', 'speaker', 'speaker_job_title', 'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context'])
    train_df['text'] = train_df['statement'].apply(clean_text)
    train_df['is_true'] = train_df['label'].apply(encode_liar_label)
    train_df = train_df[['text', 'is_true']].dropna(subset=['text'])
    train_df['source'] = 'liar'  # Added a source identifier

    valid_df = pd.read_csv(valid_path, sep='\t', header=None, names=['id', 'label', 'statement', 'subject', 'speaker', 'speaker_job_title', 'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context'])
    valid_df['text'] = valid_df['statement'].apply(clean_text)
    valid_df['is_true'] = valid_df['label'].apply(encode_liar_label)
    valid_df = valid_df[['text', 'is_true']].dropna(subset=['text'])
    valid_df['source'] = 'liar'

    test_df = pd.read_csv(test_path, sep='\t', header=None, names=['id', 'label', 'statement', 'subject', 'speaker', 'speaker_job_title', 'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context'])
    test_df['text'] = test_df['statement'].apply(clean_text)
    test_df['is_true'] = test_df['label'].apply(encode_liar_label)
    test_df = test_df[['text', 'is_true']].dropna(subset=['text'])
    test_df['source'] = 'liar'

    print(f"LIAR Train dataset shape: {train_df.shape}")
    print(f"LIAR Valid dataset shape: {valid_df.shape}")
    print(f"LIAR Test dataset shape: {test_df.shape}")

    return train_df, valid_df, test_df  # Return test_df as well

# Previous Fake News Dataset Specific Loading
def load_and_prepare_previous_dataset(fake_news_path, true_news_path):
    fake_df = pd.read_csv(fake_news_path)
    true_df = pd.read_csv(true_news_path)

    fake_df['is_true'] = 0
    true_df['is_true'] = 1

    df = pd.concat([true_df, fake_df], ignore_index=True)
    df.dropna(subset=['text'], inplace=True)
    df['text'] = df['text'].apply(clean_text)
    df = df[['text', 'is_true']].dropna(subset=['text'])
    df['source'] = 'previous'  # Added a source identifier

    print(f"Previous dataset shape: {df.shape}")
    return df

# Combined Dataset Class
def simple_tokenize(text):
    return text.split()

class CombinedNewsDataset(Dataset):
    def __init__(self, dataframe, vocab=None, max_length=500):
        self.texts = dataframe['text'].tolist()
        self.labels = dataframe['is_true'].tolist()
        self.max_length = max_length

        if vocab is None:
            counter = Counter()
            for text in self.texts:
                tokens = simple_tokenize(text)
                counter.update(tokens)
            self.vocab = {word: index + 2 for index, (word, _) in enumerate(counter.most_common())}
            self.vocab['<PAD>'] = 0
            self.vocab['<UNK>'] = 1
        else:
            self.vocab = vocab

    def encode(self, text):
        tokens = simple_tokenize(text)
        token_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        return torch.tensor(token_ids[:self.max_length])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text_tensor = self.encode(self.texts[index])
        label = torch.tensor(self.labels[index], dtype=torch.float)
        return text_tensor, label

def collate_fn(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0).long()
    labels = torch.stack(labels)
    return texts_padded, labels

# LSTM Model Class
class LSTMClassify(nn.Module):
    def __init__(self, vocab_length, embedding_dim=128, hidden_dim=128, dropout_rate=0.5):
        super(LSTMClassify, self).__init__()
        self.embedding = nn.Embedding(vocab_length, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x, (h_n, _) = self.lstm(x)
        x = self.dropout(x)  # Apply dropout after LSTM output
        output = self.fc(h_n[-1])
        return self.sigmoid(output).squeeze()

# Train Function (Modified to calculate accuracy)
def train_model(train_loader, model, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    correct_predictions = 0  # Initialize counter for correct predictions
    total_predictions = 0
    with torch.no_grad():
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(texts)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate correct predictions
            predicted = (outputs >= 0.5).float()  # Threshold at 0.5
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    return total_loss / len(train_loader), correct_predictions, total_predictions

# Prediction Function
def predict_news(text, model, vocab, device):
    if not text.strip():
        return "No text to process."

    token_ids = [vocab.get(token, vocab['<UNK>']) for token in simple_tokenize(text)]
    if len(token_ids) == 0:
        return "Error: No valid tokens after cleaning."

    input_tensor = torch.tensor(token_ids[:500], dtype=torch.long).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        score = output.item()
        if score > 0.7:  # Adjusted threshold
            return f"TRUE ({score:.4f})"
        elif score < 0.3:  # Adjusted threshold
            return f"FAKE ({score:.4f})"
        else:
            return f"UNSURE ({score:.4f})"

# Function to evaluate the model on a given dataset
def evaluate_model(data_loader, model, device):
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for texts, labels in data_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            predicted = (outputs >= 0.5).float()
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
    return correct_predictions, total_predictions

# Turtle UI
def launch_turtle_ui(model, vocab, device):
    screen = turtle.Screen()
    screen.bgcolor("#d0f0fd")  # Soft blue background
    screen.title("Fake News Detector")
    screen.setup(width=700, height=500)

    pen = turtle.Turtle()
    pen.hideturtle()
    pen.penup()
    pen.goto(0, -50)

    while True:
        article_text = screen.textinput("Enter News Text",
                                       "Paste or type the news article text (or type 'exit' to quit):")
        if article_text is None or article_text.strip().lower() == "exit":
            break

        pen.clear()
        result = predict_news(article_text, model, vocab, device)
        pen.write(result, align="center", font=("Arial", 16, "bold"))
        time.sleep(5)

    screen.bye()

# Main Function (Modified to load and combine datasets)
def main():
    # Use GPU if available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    liar_train_file = "./dataset/train.tsv"
    liar_valid_file = "./dataset/valid.tsv"
    liar_test_file = "./dataset/test.tsv"  # Added test file
    previous_fake_file = "./dataset/Fake.csv"
    previous_true_file = "./dataset/True.csv"

    if not os.path.exists("lstm_combined_news.pth"):
        liar_train_df, liar_val_df, liar_test_df = load_and_prepare_liar_dataset(liar_train_file,
                                                                                 liar_valid_file,
                                                                                 liar_test_file)  # Load test data
        previous_df = load_and_prepare_previous_dataset(previous_fake_file, previous_true_file)

        # Combine the datasets
        combined_df = pd.concat([liar_train_df, previous_df], ignore_index=True)
        print(f"Combined dataset shape: {combined_df.shape}")

        # Create a custom PyTorch Dataset from the combined dataframe
        combined_dataset = CombinedNewsDataset(combined_df)
        print(combined_dataset)

        # Split the combined dataset into training and validation sets
        train_size = int(0.8 * len(combined_dataset))
        val_size = len(combined_dataset) - train_size
        train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size],
                                                   generator=torch.Generator().manual_seed(
                                                       42))  # Added seed for reproducibility

        # Create DataLoaders for training and validation datasets
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

        # Initialize the LSTM model and move it to the device (GPU or CPU)
        model = LSTMClassify(vocab_length=len(combined_dataset.vocab)).to(device)

        # Define the optimizer (Adam) and loss function (Binary Cross Entropy)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.BCELoss()

        # Train the model for 10 epochs
        total_correct = 0
        total_trained = 0
        for epoch in range(10):
            train_loss, correct, trained = train_model(train_loader, model, optimizer, loss_fn, device)
            total_correct += correct
            total_trained += trained
            print(f"Epoch {epoch + 1}/10 - Train Loss: {train_loss:.4f}")

        # Calculate and print training accuracy
        train_accuracy = total_correct / total_trained
        print(f"Training Accuracy: {train_accuracy:.4f}")

        # Evaluate on the validation set
        val_correct, val_total = evaluate_model(val_loader, model, device)
        val_accuracy = val_correct / val_total
        print(f"Validation Accuracy: {val_accuracy:.4f}")

        # Save the trained model weights
        torch.save({'model_state_dict': model.state_dict(), 'vocab': combined_dataset.vocab},
                   "lstm_combined_news.pth")
        print("Combined model training complete. Saved as 'lstm_combined_news.pth'.")
    else:
        print("Found pre-trained combined model, loading...")

    checkpoint = torch.load("lstm_combined_news.pth", map_location=device)
    vocab = checkpoint["vocab"]
    model = LSTMClassify(len(vocab)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    launch_turtle_ui(model, vocab, device)

if __name__ == "__main__":
    main()