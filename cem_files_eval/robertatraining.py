import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification, 
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

class TextDataset(Dataset):
    """Custom dataset for text classification"""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_and_prepare_data(csv_path, test_size=0.2, random_state=42):
    """Load and prepare the dataset with balancing"""
    print("📊 Loading and preparing data...")
    
    # Load the dataset
    df = pd.read_csv(csv_path)
    print(f"Original dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Balance the dataset
    counts = df["label"].value_counts()
    minority_class = counts.idxmin()
    minority_count = counts.min()
    
    balanced_df = pd.concat([
        df[df["label"] == minority_class],
        df[df["label"] != minority_class].sample(minority_count, random_state=random_state)
    ]).sample(frac=1, random_state=random_state)
    
    print(f"Balanced dataset shape: {balanced_df.shape}")
    print(f"Balanced label distribution:\n{balanced_df['label'].value_counts()}")
    
    # Split into train and test
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        balanced_df['text'].values,
        balanced_df['label'].values,
        test_size=test_size,
        random_state=random_state,
        stratify=balanced_df['label']
    )
    
    return train_texts, test_texts, train_labels, test_labels

def create_model_and_tokenizer(model_name="roberta-base", num_labels=2):
    """Create and configure the RoBERTa model and tokenizer"""
    print(f"🤖 Loading {model_name} model and tokenizer...")
    
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    
    # Load model
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="single_label_classification"
    )
    
    # Move model to GPU if available
    model = model.to(device)
    print(f"Model moved to: {device}")
    
    return model, tokenizer

def get_optimal_batch_size():
    """Determine optimal batch size based on available GPU memory"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        if gpu_memory >= 24:  # RTX 3090, A100, etc.
            return 16
        elif gpu_memory >= 12:  # RTX 3080, 4080, etc.
            return 12
        elif gpu_memory >= 8:   # RTX 3070, 4070, etc.
            return 8
        else:  # Smaller GPUs
            return 4
    else:
        return 4  # CPU fallback

def train_model(model, tokenizer, train_texts, train_labels, test_texts, test_labels, 
                batch_size=None, epochs=5, learning_rate=2e-5, max_length=512):
    """Train the RoBERTa model with GPU optimizations"""
    print("🏋️ Setting up training...")
    
    # Auto-determine optimal batch size
    if batch_size is None:
        batch_size = get_optimal_batch_size()
    print(f"Using batch size: {batch_size}")
    
    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_length)
    
    # Training arguments with GPU optimizations
    training_args = TrainingArguments(
        output_dir="./roberta_style_classifier",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=3,
        report_to="none",  # Disable wandb/tensorboard
        dataloader_pin_memory=True if torch.cuda.is_available() else False,  # GPU optimization
        remove_unused_columns=False,
        # GPU-specific optimizations
        fp16=torch.cuda.is_available(),  # Mixed precision training (faster on GPU)
        dataloader_num_workers=4 if torch.cuda.is_available() else 0,  # Parallel data loading
        gradient_accumulation_steps=1,  # Can be increased for larger effective batch size
    )
    
    # Define compute_metrics function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy}
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    print("🚀 Starting training...")
    print(f"Training on: {device}")
    if torch.cuda.is_available():
        print(f"GPU Memory Usage: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    trainer.train()
    
    return trainer

def evaluate_model(trainer, test_texts, test_labels, tokenizer, max_length=512):
    """Evaluate the trained model"""
    print("📈 Evaluating model...")
    
    # Create test dataset
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_length)
    
    # Get predictions
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, 
                              target_names=['Obama', 'Trump']))
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Obama', 'Trump'],
                yticklabels=['Obama', 'Trump'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, pred_labels

def save_model(trainer, tokenizer, model_path="./best_roberta_model"):
    """Save the trained model and tokenizer"""
    print(f"💾 Saving model to {model_path}...")
    
    # Save the model
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    
    print("✅ Model saved successfully!")

def main():
    """Main training function"""
    print("🎯 RoBERTa Style Classification Training")
    print("=" * 50)
    
    # Configuration
    CSV_PATH = "style_classification_dataset.csv"
    MODEL_NAME = "roberta-base"
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 512
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # GPU optimizations
    if torch.cuda.is_available():
        print("🎮 GPU detected - enabling optimizations!")
        # Clear GPU cache
        torch.cuda.empty_cache()
    else:
        print("💻 Running on CPU - training will be slower")
    
    try:
        # 1. Load and prepare data
        train_texts, test_texts, train_labels, test_labels = load_and_prepare_data(
            CSV_PATH, TEST_SIZE, RANDOM_STATE
        )
        
        # 2. Create model and tokenizer
        model, tokenizer = create_model_and_tokenizer(MODEL_NAME)
        
        # 3. Train the model (batch_size will be auto-determined)
        trainer = train_model(
            model, tokenizer, train_texts, train_labels, test_texts, test_labels,
            batch_size=None, epochs=EPOCHS, learning_rate=LEARNING_RATE, max_length=MAX_LENGTH
        )
        
        # 4. Evaluate the model
        accuracy, predictions = evaluate_model(
            trainer, test_texts, test_labels, tokenizer, MAX_LENGTH
        )
        
        # 5. Save the model
        save_model(trainer, tokenizer)
        
        print("\n🎉 Training completed successfully!")
        print(f"Final Test Accuracy: {accuracy:.4f}")
        
        # Final GPU memory report
        if torch.cuda.is_available():
            print(f"Final GPU Memory Usage: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
