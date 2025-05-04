import pandas as pd
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import threading 
from newspaper import Article
import turtle

# Data Preprocessing 
#---------------------
def clean_text(text):
    # Split text into sentences using punctuation (basic version)
    sentences = re.split(r'[.!?]', text)
    
    # Keep only meaningful sentences with at least 5 words
    meaningful = [s.strip() for s in sentences if len(s.strip().split()) >= 5]

    # Join and do basic cleanup
    cleaned = ' '.join(meaningful)
    cleaned = re.sub(r'\W', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip().lower()

def normalize_subject(subject):
    subject = subject.lower()
    if subject in ['news', 'worldnews']:
        return 'news'
    elif subject in ['politics', 'politicsNews']:
        return 'politics'
    else:
        return 'other'

def load_and_prepare_dataset(fake_news_path, true_news_path):
    fake_df = pd.read_csv(fake_news_path)
    true_df = pd.read_csv(true_news_path)

    # Add binary label
    fake_df['is_true'] = 0
    true_df['is_true'] = 1

    # Combine
    df = pd.concat([true_df, fake_df], ignore_index=True)

    # Drop rows with missing text
    df.dropna(subset=['text'], inplace=True)

    # Clean text
    df['text'] = df['text'].apply(clean_text)

    # Normalize and combine subject categories
    df['subject'] = df['subject'].apply(normalize_subject)

    # Label encode subject
    le = LabelEncoder()
    df['subject_encoded'] = le.fit_transform(df['subject'])

    print(f"Encoded subjects: {list(le.classes_)}")
    print(f"Final dataset shape: {df.shape}")
    
    return df[['text', 'subject_encoded', 'is_true']]

# Model Building
#-----------------
def simple_tokenize(text):
    return text.split()

# Custom Dataset Class
class NewsDataset(Dataset):
    def __init__(self, dataframe, vocab=None, max_length=500):
        self.texts = dataframe['text'].tolist()
        self.labels = dataframe['is_true'].tolist()
        self.subjects = dataframe['subject_encoded'].tolist()
        self.max_length = max_length

        # Build vocabulary if not provided
        if vocab is None:
            counter = Counter()
            for text in self.texts:
                tokens = simple_tokenize(text)
                counter.update(tokens)
            self.vocab = {word: index+2 for index, (word, _) in enumerate(counter.most_common())}
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
        subject = torch.tensor(self.subjects[index])
        return text_tensor, label, subject
    
def collate_fn(batch):
    texts, labels, subjects = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0).long()
    labels = torch.stack(labels)
    subjects = torch.stack(subjects)
    return texts_padded, labels, subjects

# LSTM Model Class
class LSTMClassify(nn.Module):
    def __init__(self, vocab_length, embedding_dim=128, hidden_dim=128):
        super(LSTMClassify, self).__init__()
        self.embedding = nn.Embedding(vocab_length, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.embedding(x) 
        _, (h_n, _) = self.lstm(x)
        output = self.fc(h_n[-1])
        return self.sigmoid(output).squeeze()

# Train Function
def train_model(train_loader, model, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for texts, labels, subjects in train_loader:
        texts, labels, subjects = texts.to(device), labels.to(device), subjects.to(device)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# Prediction Function
def predict_news(text, model, vocab, device):
    if not text.strip():
        return "No text to process."

    print("Text length:", len(text))
    print("Preview:", text[:300])  # Preview first 300 chars

    token_ids = [vocab.get(token, vocab['<UNK>']) for token in simple_tokenize(text)]
    if len(token_ids) == 0:
        return "Error: No valid tokens after cleaning."

    input_tensor = torch.tensor(token_ids[:500], dtype=torch.long).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        score = output.item()
        print("Model raw output:", score)
        if score > 0.8:
            return f"TRUE ({score:.4f})"
        elif score < 0.2:
            return f"FAKE ({score:.4f})"
        else:
            return f"UNSURE ({score:.4f})"
    # try:
    #     article = Article(url)
    #     article.download()
    #     article.parse()
    #     text = clean_text(article.text)

    #     # # Skip the first few sentences that may be image captions or navigation text
    #     # lines = text.split('\n')
    #     # cleaned_lines = [line for line in lines if len(line.split()) > 5 and 'caption' not in line and 'image' not in line]
    #     # text = ' '.join(cleaned_lines)

    #     print("Article length:", len(text))
    #     print("Preview:", text[:300])  # Preview first 300 chars

    #     token_ids = [vocab.get(token, vocab['<UNK>']) for token in simple_tokenize(text)]
    #     input_tensor = torch.tensor(token_ids[:500], dtype=torch.long).unsqueeze(0).to(device)

    #     model.eval()
    #     with torch.no_grad():
    #         output = model(input_tensor)
    #         print("Model raw output:", output.item())
    #         # prediction = torch.round(output)
    #         # return "TRUE" if prediction.item() > 0.5 else "FAKE"
    #         score = output.item()
    #         if score > 0.8:
    #             return f"TRUE ({score:.4f})"
    #         elif score < 0.2:
    #             return f"FAKE ({score:.4f})"
    #         else:
    #             return f"UNSURE ({score:.4f})"

    # except Exception as e:
    #     return f"Error {str(e)}"
    
# Turtle UI
def launch_turtle_ui(model, vocab, device):
    screen = turtle.Screen()
    screen.bgcolor("#d0f0fd")  # Soft blue background
    screen.title("Fake News Detector")
    screen.setup(width=700, height=500)

    article_text = turtle.textinput("Enter News Text", "Paste or type the news article text (or first few lines):")
    if article_text is None or article_text.strip() == "":
        print("No text entered. Exiting.")
        return

    # Wrap detection logic in a thread, but not the input prompt
    def run_detection():
        result = predict_news(article_text, model, vocab, device)
        display_result(result)

    def display_result(message):
        pen = turtle.Turtle()
        pen.hideturtle()
        pen.penup()
        pen.goto(0, -50)
        pen.write(message, align="center", font=("Arial", 16, "bold"))

    thread = threading.Thread(target=run_detection)
    thread.start()

    turtle.done()

# Main Function
def main():
    # Use GPU if available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists("lstm_fake_news.pth"):
        df_final = load_and_prepare_dataset("./dataset/Fake.csv", "./dataset/True.csv")
        print(df_final['is_true'].value_counts())

        # Create a custom PyTorch Dataset from the dataframe
        dataset = NewsDataset(df_final)
        print(dataset)
        # Split the dataset into 80% training and 20% validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        # Create DataLoaders for training and validation datasets
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        # Initialize the LSTM model and move it to the device (GPU or CPU)
        model = LSTMClassify(vocab_length=len(dataset.vocab)).to(device)
        # Define the optimizer (Adam) and loss function (Binary Cross Entropy)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.BCELoss()
        # Train the model for 10 epochs
        for epoch in range(10):
            train_loss = train_model(train_loader, model, optimizer, loss_fn, device)
            print(f"Epoch {epoch+1}/10 - Train Loss: {train_loss:.4f}")
        # Save the trained model weights
        torch.save({'model_state_dict': model.state_dict(), 'vocab': dataset.vocab}, "lstm_fake_news.pth")
        print("Model training complete. Saved as 'lstm_fake_news.pth'.")
    else:
        print("Found pre-trained model, loading...")    
        # print(df_final['is_true'].value_counts())


    checkpoint = torch.load("lstm_fake_news.pth", map_location=device)
    vocab = checkpoint["vocab"]
    model = LSTMClassify(len(vocab)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    launch_turtle_ui(model, vocab, device)

    # ================= Test on known TRUE sample ====================
    # df_test = load_and_prepare_dataset("./dataset/Fake.csv", "./dataset/True.csv")
    # true_sample = df_test[df_test['is_true'] == 1].iloc[0]
    # print("\nTesting model on known TRUE sample:")
    # print("Sample text snippet:", true_sample['text'][:300])

    # # Encode like in predict_news()
    # text = clean_text(true_sample['text'])
    # token_ids = [vocab.get(token, vocab['<UNK>']) for token in simple_tokenize(text)]
    # input_tensor = torch.tensor(token_ids[:500]).unsqueeze(0).to(device)

    # with torch.no_grad():
    #     output = model(input_tensor)
    #     score = output.item()
    #     print(f"Prediction score: {score:.4f}")
    #     print("Prediction:", "TRUE" if score > 0.5 else "FAKE")
    # # ===============================================================


if __name__ == "__main__":
    main()