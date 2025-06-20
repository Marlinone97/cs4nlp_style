# === 1. Dataset Class ===
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BartTokenizer, BartForConditionalGeneration,
    AutoTokenizer, AutoModel,
    RobertaTokenizer, RobertaForSequenceClassification
)
from torch.optim import AdamW
import torch.nn.functional as F
from random import shuffle
from tqdm import tqdm
import time
import sys
import warnings

sys.stdout = open("training_log.txt", "w")
sys.stderr = sys.stdout

warnings.filterwarnings("ignore")

class CycleStyleDataset(Dataset):
    def __init__(self, obama_path, trump_path):
        df_o = pd.read_csv(obama_path)
        df_t = pd.read_csv(trump_path)

        self.obama = [s.strip() for s in df_o["text"] if isinstance(s, str)]
        self.trump = [s.strip() for s in df_t["text"] if isinstance(s, str)]
        self.samples = self.obama + self.trump
        self.styles = ["obama"] * len(self.obama) + ["trump"] * len(self.trump)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        original = self.samples[idx]
        original_style = self.styles[idx]
        target_style = "trump" if original_style == "obama" else "obama"
        return original, original_style, target_style

# === 2. Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
tokenizer.add_tokens(["<to_obama>", "<to_trump>"])
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)
model.resize_token_embeddings(len(tokenizer))

sim_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
sim_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)

style_clf_tokenizer = RobertaTokenizer.from_pretrained("./roberta_style_classifier/checkpoint-411")
style_clf_model = RobertaForSequenceClassification.from_pretrained("./roberta_style_classifier/checkpoint-411").to(device)

label_map = {"obama": 0, "trump": 1}

# === 3. Helper Functions ===
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1).clamp(min=1e-9)

def sentence_similarity(sent1, sent2):
    enc = sim_tokenizer([sent1, sent2], padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_out = sim_model(**enc)
    emb = mean_pooling(model_out, enc['attention_mask'])
    return F.cosine_similarity(emb[0], emb[1], dim=0)

def soft_style_classification_loss(sentences, target_styles, temperature=2.0):
    inputs = style_clf_tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = style_clf_model(**inputs)
    logits = outputs.logits / temperature
    probs = torch.nn.functional.softmax(logits, dim=-1)
    soft_targets = torch.full_like(probs, 0.1)
    for i, style in enumerate(target_styles):
        soft_targets[i][label_map[style]] = 0.9
    return F.kl_div(probs.log(), soft_targets, reduction='batchmean')

def encode_bart_input(source_text, target_text, tokenizer, max_length=128):
    enc = tokenizer(source_text, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        target_enc = tokenizer(target_text, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
    input_ids = enc["input_ids"].squeeze(0)
    attention_mask = enc["attention_mask"].squeeze(0)
    labels = target_enc["input_ids"].squeeze(0)
    labels[labels == tokenizer.pad_token_id] = -100
    return input_ids, attention_mask, labels

# === 4. Load Dataset ===
dataset = CycleStyleDataset("all_obama_sent.csv", "all_trump_sent.csv")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# === 5. Training Loop ===
optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 10

for epoch in range(epochs):
    start_time = time.time()
    total_loss = 0
    model.train()

    for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
        originals, orig_styles, target_styles = batch
        batch_input_ids, batch_attn_masks, batch_labels = [], [], []
        all_source_texts, all_target_texts = [], []
        cycle_input_ids, cycle_attention_masks, cycle_labels = [], [], []
        all_cycle_originals, all_cycle_outputs = [], []

        for i in range(len(originals)):
            source = f"<to_{target_styles[i]}> {originals[i]}"
            target = originals[i]
            inp, attn, lbl = encode_bart_input(source, target, tokenizer)
            batch_input_ids.append(inp)
            batch_attn_masks.append(attn)
            batch_labels.append(lbl)
            all_source_texts.append(originals[i])
            all_target_texts.append(target)

            with torch.no_grad():
                gen_ids = model.generate(input_ids=inp.unsqueeze(0).to(device), attention_mask=attn.unsqueeze(0).to(device), max_length=64)
                generated = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

            cycle_prompt = f"<to_{orig_styles[i]}> {generated}"
            inp_c, attn_c, lbl_c = encode_bart_input(cycle_prompt, originals[i], tokenizer)
            cycle_input_ids.append(inp_c)
            cycle_attention_masks.append(attn_c)
            cycle_labels.append(lbl_c)
            all_cycle_originals.append(originals[i])
            all_cycle_outputs.append(generated)

        input_ids = torch.stack(batch_input_ids).to(device)
        attention_mask = torch.stack(batch_attn_masks).to(device)
        labels = torch.stack(batch_labels).to(device)

        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        lm_loss = output.loss

        generated_sentences = tokenizer.batch_decode(
            model.generate(input_ids, attention_mask=attention_mask, max_length=64),
            skip_special_tokens=True
        )

        cosine_loss = torch.stack([
            1 - sentence_similarity(s1, s2) for s1, s2 in zip(all_source_texts, generated_sentences)
        ]).mean()

        style_loss = soft_style_classification_loss(generated_sentences, target_styles)

        copy_penalty = torch.stack([
            sentence_similarity(s1, s2) for s1, s2 in zip(all_source_texts, generated_sentences)
        ]).mean()

        cycle_ids = torch.stack(cycle_input_ids).to(device)
        cycle_mask = torch.stack(cycle_attention_masks).to(device)
        cycle_lbls = torch.stack(cycle_labels).to(device)

        cycle_output = model(input_ids=cycle_ids, attention_mask=cycle_mask, labels=cycle_lbls)
        cycle_lm_loss = cycle_output.loss

        cycle_cosine_loss = torch.stack([
            1 - sentence_similarity(s1, s2) for s1, s2 in zip(all_cycle_originals, all_cycle_outputs)
        ]).mean()

        cycle_style_loss = soft_style_classification_loss(all_cycle_outputs, orig_styles)

        total_batch_loss = (
            lm_loss +
            0.25 * cosine_loss +
            0.25 * style_loss +
            0.25 * (1 - copy_penalty) +
            cycle_lm_loss +
            0.25 * cycle_cosine_loss +
            0.25 * cycle_style_loss
        )

        total_batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += total_batch_loss.item()

        print(
            f"lm={lm_loss:.2f}, cos={cosine_loss:.2f}, style={style_loss:.2f}, "
            f"copy_penalty={(1 - copy_penalty):.2f}, cyc_lm={cycle_lm_loss:.2f}, "
            f"cyc_cos={cycle_cosine_loss:.2f}, cyc_style={cycle_style_loss:.2f}"
        )

    duration = time.time() - start_time
    print(f"[Epoch {epoch+1}] avg_loss = {total_loss / len(loader):.4f} | duration: {duration:.2f}s")

# === 6. Save Model ===
model.save_pretrained("./bart-cycle-style")
tokenizer.save_pretrained("./bart-cycle-style")