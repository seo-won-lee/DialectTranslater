# -*- coding: utf-8 -*-
"""aug-deploy-model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_YYYbizQOWUcu_PKsRqPgHEUFGRME1tV
"""

from google.colab import drive
drive.mount('/content/drive')

!pip install torch torchvision torchaudio
!pip install transformers
!pip install pandas scikit-learn
!pip install tqdm
!pip install matplotlib

import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
from transformers import get_linear_schedule_with_warmup

# ✅ GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ✅ 데이터 불러오기 (로컬 경로로 수정)
data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/aug_data.csv', encoding='ISO-8859-1')

# ✅ 데이터 9:1 비율로 Train/Test 분할
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

# ✅ Test set 저장
aug_test_path = '/content/drive/MyDrive/Colab Notebooks/aug-test.csv'
test_data.to_csv(aug_test_path, index=False, encoding='ISO-8859-1')
print(f"Test set saved to {aug_test_path}")

# ✅ 데이터 크기 확인
print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

# ✅ TranslationDataset 클래스 정의
class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 데이터 유효성 검사
        required_columns = ['source', 'source_word', 'country', 'target']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        source_text = str(row['source']).strip()
        masked_word = str(row['source_word']).strip()
        country = str(row['country']).strip()
        target_text = str(row['target']).strip()

        # 문장에 마스킹 적용
        if masked_word in source_text:
            source_text = source_text.replace(masked_word, "<mask>")

        # 프롬프트 구성
        source_text = f"translate American to {country}: {source_text}"

        # 인코딩
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 레이블 처리
        labels = target_encoding['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': source_encoding['input_ids'].squeeze(0),
            'attention_mask': source_encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }

    @staticmethod
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# ✅ 데이터로더 생성 함수
def create_data_loader(data, tokenizer, batch_size, max_len=128):
    dataset = TranslationDataset(data, tokenizer, max_len)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=TranslationDataset.collate_fn,
        num_workers=0,
        drop_last=False
    )

    return data_loader

# ✅ tokenizer와 모델 로드
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# ✅ 모델을 GPU로 이동
model.to(device)
print("Model loaded and moved to device.")

learning_rates = [6e-5]
batch_sizes = [2]
num_epochs_list = [7]
param_grid = list(itertools.product(learning_rates, batch_sizes, num_epochs_list))

all_train_losses = []

for lr, batch_size, num_epochs in param_grid:
    print(f"\n🔍 Testing: lr={lr}, batch_size={batch_size}, epochs={num_epochs}")

    train_loader = create_data_loader(train_data, tokenizer, batch_size)

    model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = num_training_steps * 0.1
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()

            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Calculate average train loss
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1} - Avg Train Loss: {avg_train_loss:.4f}")

        # ✅ GPU 메모리 정리
        torch.cuda.empty_cache()

    final_train_loss = train_losses[-1]  # 마지막 에포크의 Train Loss

    print(f"🔹 Final Avg Train Loss for lr={lr}, batch_size={batch_size}, epochs={num_epochs}: {final_train_loss:.4f}")

    all_train_losses.append(((batch_size, lr, num_epochs), train_losses))

# ✅ 모델과 토크나이저 저장
output_dir = "/content/drive/MyDrive/Colab Notebooks/augDT-v1"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")

!pip install huggingface_hub

from huggingface_hub import login

# Hugging Face Token 입력
login(token="")

from huggingface_hub import HfApi

# ✅ Hugging Face API 인스턴스 생성
api = HfApi()

# ✅ Hugging Face 저장소 생성 (저장소 이름 설정)
repo_id = "thisischloe/augv1-dialect-Translater"  # Hugging Face 저장소 이름
api.create_repo(repo_id=repo_id, exist_ok=True)  # 이미 존재하면 덮어쓰기 허용

# ✅ 모델 업로드(lr = 6e-5, batch=2, ep=7)
api.upload_folder(
    folder_path="/content/drive/MyDrive/Colab Notebooks/augDT-v1",  # 업로드할 폴더 경로
    repo_id=repo_id,                   # 저장소 ID
    repo_type="model",                 # 업로드 타입 (모델)
    commit_message=" model upload"  # 커밋 메시지
)

print(f"✅ Model uploaded to Hugging Face Hub: https://huggingface.co/{repo_id}")

import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
from transformers import get_linear_schedule_with_warmup

# ✅ GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ✅ 데이터 크기 확인
print(f"Train size: {len(train_data)}, Test: {len(test_data)}")

# ✅ TranslationDataset 클래스 정의
class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 데이터 유효성 검사
        required_columns = ['source', 'source_word', 'country', 'target']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        source_text = str(row['source']).strip()
        masked_word = str(row['source_word']).strip()
        country = str(row['country']).strip()
        target_text = str(row['target']).strip()

        # 문장에 마스킹 적용
        if masked_word in source_text:
            source_text = source_text.replace(masked_word, "<mask>")

        # 프롬프트 구성
        source_text = f"translate American to {country}: {source_text}"

        # 인코딩
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 레이블 처리
        labels = target_encoding['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': source_encoding['input_ids'].squeeze(0),
            'attention_mask': source_encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }

    @staticmethod
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# ✅ 데이터로더 생성 함수
def create_data_loader(data, tokenizer, batch_size, max_len=128):
    dataset = TranslationDataset(data, tokenizer, max_len)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=TranslationDataset.collate_fn,
        num_workers=0,
        drop_last=False
    )

    return data_loader

# ✅ tokenizer와 모델 로드
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# ✅ 모델을 GPU로 이동
model.to(device)
print("Model loaded and moved to device.")

learning_rates = [5e-5]
batch_sizes = [3]
num_epochs_list = [5]
param_grid = list(itertools.product(learning_rates, batch_sizes, num_epochs_list))

all_train_losses = []

for lr, batch_size, num_epochs in param_grid:
    print(f"\n🔍 Testing: lr={lr}, batch_size={batch_size}, epochs={num_epochs}")

    train_loader = create_data_loader(train_data, tokenizer, batch_size)

    model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()

            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Calculate average train loss
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1} - Avg Train Loss: {avg_train_loss:.4f}")

        # ✅ GPU 메모리 정리
        torch.cuda.empty_cache()

    final_train_loss = train_losses[-1]  # 마지막 에포크의 Train Loss

    print(f"🔹 Final Avg Train Loss for lr={lr}, batch_size={batch_size}, epochs={num_epochs}: {final_train_loss:.4f}")

    all_train_losses.append(((batch_size, lr, num_epochs), train_losses))

# ✅ 모델과 토크나이저 저장
output_dir = "/content/drive/MyDrive/Colab Notebooks/augDT-v2"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")

from huggingface_hub import HfApi

# ✅ Hugging Face API 인스턴스 생성
api = HfApi()

# ✅ Hugging Face 저장소 생성 (lr = 5e-5, batch=3, ep=5)
repo_id = "thisischloe/augv2-dialect-Translater"  # Hugging Face 저장소 이름
api.create_repo(repo_id=repo_id, exist_ok=True)  # 이미 존재하면 덮어쓰기 허용

# ✅ 모델 업로드
api.upload_folder(
    folder_path="/content/drive/MyDrive/Colab Notebooks/augDT-v2",  # 업로드할 폴더 경로
    repo_id=repo_id,                   # 저장소 ID
    repo_type="model",                 # 업로드 타입 (모델)
    commit_message=" model upload"  # 커밋 메시지
)

print(f"✅ Model uploaded to Hugging Face Hub: https://huggingface.co/{repo_id}")

import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
from transformers import get_linear_schedule_with_warmup

# ✅ GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ✅ 데이터 크기 확인
print(f"Train size: {len(train_data)}, Test: {len(test_data)}")

# ✅ TranslationDataset 클래스 정의
class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 데이터 유효성 검사
        required_columns = ['source', 'source_word', 'country', 'target']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        source_text = str(row['source']).strip()
        masked_word = str(row['source_word']).strip()
        country = str(row['country']).strip()
        target_text = str(row['target']).strip()

        # 문장에 마스킹 적용
        if masked_word in source_text:
            source_text = source_text.replace(masked_word, "<mask>")

        # 프롬프트 구성
        source_text = f"translate American to {country}: {source_text}"

        # 인코딩
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 레이블 처리
        labels = target_encoding['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': source_encoding['input_ids'].squeeze(0),
            'attention_mask': source_encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }

    @staticmethod
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# ✅ 데이터로더 생성 함수
def create_data_loader(data, tokenizer, batch_size, max_len=128):
    dataset = TranslationDataset(data, tokenizer, max_len)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=TranslationDataset.collate_fn,
        num_workers=0,
        drop_last=False
    )

    return data_loader

# ✅ tokenizer와 모델 로드
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# ✅ 모델을 GPU로 이동
model.to(device)
print("Model loaded and moved to device.")

learning_rates = [5e-5]
batch_sizes = [2]
num_epochs_list = [5]
param_grid = list(itertools.product(learning_rates, batch_sizes, num_epochs_list))

all_train_losses = []

for lr, batch_size, num_epochs in param_grid:
    print(f"\n🔍 Testing: lr={lr}, batch_size={batch_size}, epochs={num_epochs}")

    train_loader = create_data_loader(train_data, tokenizer, batch_size)

    model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()

            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Calculate average train loss
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1} - Avg Train Loss: {avg_train_loss:.4f}")

        # ✅ GPU 메모리 정리
        torch.cuda.empty_cache()

    final_train_loss = train_losses[-1]  # 마지막 에포크의 Train Loss

    print(f"🔹 Final Avg Train Loss for lr={lr}, batch_size={batch_size}, epochs={num_epochs}: {final_train_loss:.4f}")

    all_train_losses.append(((batch_size, lr, num_epochs), train_losses))

# ✅ 모델과 토크나이저 저장
output_dir = "/content/drive/MyDrive/Colab Notebooks/augDT-v3"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")

from huggingface_hub import HfApi

# ✅ Hugging Face API 인스턴스 생성
api = HfApi()

# ✅ Hugging Face 저장소 생성 (저장소 이름 설정)
repo_id = "thisischloe/augv3-dialect-Translater"  # Hugging Face 저장소 이름
api.create_repo(repo_id=repo_id, exist_ok=True)  # 이미 존재하면 덮어쓰기 허용

# ✅ 모델 업로드(lr = 5e-5, batch=2, ep=5)
api.upload_folder(
    folder_path="/content/drive/MyDrive/Colab Notebooks/augDT-v3",  # 업로드할 폴더 경로
    repo_id=repo_id,                   # 저장소 ID
    repo_type="model",                 # 업로드 타입 (모델)
    commit_message=" model upload"  # 커밋 메시지
)

print(f"✅ Model uploaded to Hugging Face Hub: https://huggingface.co/{repo_id}")