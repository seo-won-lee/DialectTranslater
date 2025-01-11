# 필요한 라이브러리 임포트
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# ✅ GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ✅ 데이터 불러오기 (로컬 경로로 수정)
data = pd.read_csv('./final_data.csv', encoding='ISO-8859-1')

# ✅ TranslationDataset 클래스 정의
class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 데이터 유효성 검사
        required_columns = ['source', 'source_word', 'country', 'meaning', 'target']
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
        meaning = str(row['meaning']).strip()
        target_text = str(row['target']).strip()

        # mask 처리
        if masked_word and masked_word in source_text:
            masked_source = source_text.replace(masked_word, "<mask>")
        else:
            masked_source = source_text

        # 프롬프트 구성
        source_text = f"Translate this sentence to {country} English, considering the meaning '{meaning}': {masked_source}"

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

# ✅ 데이터로더 생성
# 하이퍼 파라미터
train_loader = create_data_loader(data, tokenizer, batch_size=2, max_len=16)

# ✅ 데이터 확인
for batch in train_loader:
    print("Batch input IDs shape:", batch['input_ids'].shape)
    print("Batch labels shape:", batch['labels'].shape)
    break


# 하이퍼 파라미터
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Gradient Accumulation 설정
gradient_accumulation_steps = 2
train_losses = []

# 하이퍼 파라미터
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for step, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        train_loss += loss.item()

        # Gradient 누적
        loss = loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # 평균 Train Loss 저장
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    print(f"Epoch {epoch + 1}, Avg Train Loss: {avg_train_loss:.4f}")

# ✅ 메모리 정리
torch.cuda.empty_cache()

# ✅ 모델과 토크나이저 저장
output_dir = "./dialectTranslater"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")
