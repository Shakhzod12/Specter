
from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# Load pre-trained tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
specter_model = BertModel.from_pretrained('allenai/specter')




class DomainDataset(Dataset):
    def __init__(self, texts, citations, labels=None):
        self.texts = texts
        self.citations = citations
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        citation = self.citations[idx]
        label = self.labels[idx] if self.labels is not None else None

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

        if label is not None:
            return {
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_mask),
                'citation_ids': torch.tensor(citation_ids),
                'citation_mask': torch.tensor(citation_mask),
                'label': torch.tensor(label)
            }
        else:
            return {
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_mask),
                'citation_ids': torch.tensor(citation_ids),
                'citation_mask': torch.tensor(citation_mask)
            }



class DomainAdaptationModel(nn.Module):
    def __init__(self, num_classes):
        super(DomainAdaptationModel, self).__init__()
        self.specter = specter_model
        self.dropout = nn.Dropout(0.1)
        self.citation_fc = nn.Linear(768, 256)  # Adjust dimensions as needed
        self.classifier = nn.Linear(1024, num_classes)  # Adjust dimensions as needed

    def forward(self, input_ids, attention_mask, citation_ids, citation_mask):
        specter_outputs = self.specter(input_ids=input_ids, attention_mask=attention_mask)
        specter_pooled_output = specter_outputs.pooler_output
        citation_rep = self.specter(input_ids=citation_ids, attention_mask=citation_mask).pooler_output
        citation_rep = self.dropout(citation_rep)
        citation_representation = self.citation_fc(citation_rep)
        combined_representation = torch.cat((specter_pooled_output, citation_representation), dim=1)
        logits = self.classifier(combined_representation)
        return logits




# Prepare source and target domain data
source_texts = ["Source domain text 1", "Source domain text 2", "Source domain text 3"]
source_citations = ["Source domain citation 1", "Source domain citation 2", "Source domain citation 3"]
source_labels = [0, 1, 0]  # Example labels for source domain

target_texts = ["Target domain text 1", "Target domain text 2", "Target domain text 3"]
target_citations = ["Target domain citation 1", "Target domain citation 2", "Target domain citation 3"]
target_labels = [1, 0, 1]  # Example labels for target domain

# Create source and target domain datasets
source_dataset = DomainDataset(source_texts, source_citations, source_labels)
target_dataset = DomainDataset(target_texts, target_citations, target_labels)

# Create source and target domain dataloaders
source_dataloader = DataLoader(source_dataset, batch_size=2, shuffle=True)
target_dataloader = DataLoader(target_dataset, batch_size=2, shuffle=True)

# Initialize domain adaptation model
num_classes = 2  # Number of classes for classification task
domain_model = DomainAdaptationModel(num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(domain_model.parameters(), lr=1e-5)

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    domain_model.train()

    # Train on source domain
    for batch in source_dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        citation_ids = batch['citation_ids']
        citation_mask = batch['citation_mask']
        labels = batch['label']

        # Forward pass
        logits = domain_model(input_ids, attention_mask, citation_ids, citation_mask)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Perform domain adaptation on target domain
    for batch in target_dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        citation_ids = batch['citation_ids']
        citation_mask = batch['citation_mask']

        # Forward pass (without labels)
        logits = domain_model(input_ids, attention_mask, citation_ids, citation_mask)

        # Compute target domain loss (without labels)
        target_loss = -logits.max(dim=1).values.mean()

        # Backward pass and optimization
        optimizer.zero_grad()
        target_loss.backward()
        optimizer.step()

    # Evaluation on target domain
    domain_model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in target_dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            citation_ids = batch['citation_ids']
            citation_mask = batch['citation_mask']
            labels = batch['label']

            # Forward pass
            logits = domain_model(input_ids, attention_mask, citation_ids, citation_mask)

            # Compute predictions
            _, predicted_labels = torch.max(logits, dim=1)

            # Update counts
            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    print(f"Epoch: {epoch+1} | Target Domain Accuracy: {accuracy}")

