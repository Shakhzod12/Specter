from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class SPECTERDataset(Dataset):
    def __init__(self, texts, citations, labels):
        self.texts = texts
        self.citations = citations
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        citation = self.citations[idx]
        label = self.labels[idx]

        # Tokenize text and citation
        input_ids = self.tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=512)
        citation_ids = self.tokenizer.encode(citation, add_special_tokens=True, truncation=True, max_length=128)

        # Create attention masks
        attention_mask = [1] * len(input_ids)
        citation_mask = [1] * len(citation_ids)

        # Pad input sequences
        input_padding_length = 512 - len(input_ids)
        citation_padding_length = 128 - len(citation_ids)
        input_ids += [0] * input_padding_length
        citation_ids += [0] * citation_padding_length
        attention_mask += [0] * input_padding_length
        citation_mask += [0] * citation_padding_length

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'citation_ids': torch.tensor(citation_ids),
            'citation_mask': torch.tensor(citation_mask),
            'label': torch.tensor(label)
        }

class SPECTER(nn.Module):
    def __init__(self, use_citation=True):
        super(SPECTER, self).__init__()
        self.use_citation = use_citation
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_classes)

        if self.use_citation:
            self.citation_fc = nn.Linear(768, 256)  # Adjust dimensions as needed

    def forward(self, input_ids, attention_mask, citation_ids=None, citation_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        if self.use_citation:
            citation_rep = self.bert(input_ids=citation_ids, attention_mask=citation_mask).pooler_output
            citation_rep = self.dropout(citation_rep)
            citation_representation = self.citation_fc(citation_rep)
            combined_representation = torch.cat((pooled_output, citation_representation), dim=1)
        else:
            combined_representation = pooled_output

        logits = self.classifier(combined_representation)
        return logits

# Prepare data for ablation study
texts = ["Example text 1", "Example text 2", "Example text 3"]
citations = ["Example citation 1", "Example citation 2", "Example citation 3"]
labels = [0, 1, 0]  # Example labels for classification task

# Create dataset for ablation study
dataset = SPECTERDataset(texts, citations, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize SPECTER model with and without citation
num_classes = 2  # Number of classes for classification task
specter_with_citation = SPECTER(use_citation=True)
specter_without_citation = SPECTER(use_citation=False)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_with_citation = torch.optim.Adam(specter_with_citation.parameters(), lr=1e-5)
optimizer_without_citation = torch.optim.Adam(specter_without_citation.parameters(), lr=1e-5)

# Training loop for ablation study
num_epochs = 3
for epoch in range(num_epochs):
    specter_with_citation.train()
    specter_without_citation.train()

    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        citation_ids = batch['citation_ids']
        citation_mask = batch['citation_mask']
        labels = batch['label']

        # Training with citation
        optimizer_with_citation.zero_grad()
        logits_with_citation = specter_with_citation(input_ids, attention_mask, citation_ids, citation_mask)
        loss_with_citation = criterion(logits_with_citation, labels)
        loss_with_citation.backward()
        optimizer_with_citation.step()

        # Training without citation
        optimizer_without_citation.zero_grad()
        logits_without_citation = specter_without_citation(input_ids, attention_mask)
        loss_without_citation = criterion(logits_without_citation, labels)
        loss_without_citation.backward()
        optimizer_without_citation.step()

    print(f"Epoch: {epoch+1} | Loss with Citation: {loss_with_citation.item()} | Loss without Citation: {loss_without_citation.item()}")
