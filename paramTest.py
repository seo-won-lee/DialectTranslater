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

# ✅ 데이터 나누기 (Train: 80%, Validation: 20%)
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
print(f"Train Data 크기: {len(train_data)}")
print(f"Validation Data 크기: {len(val_data)}")

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
def create_data_loaders(train_data, val_data, tokenizer, batch_size, max_len=128):
    train_dataset = TranslationDataset(train_data, tokenizer, max_len)
    val_dataset = TranslationDataset(val_data, tokenizer, max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=TranslationDataset.collate_fn,
        num_workers=0,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=TranslationDataset.collate_fn,
        num_workers=0,
        drop_last=False
    )

    return train_loader, val_loader

# ✅ tokenizer와 모델 로드
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# ✅ 모델을 GPU로 이동
model.to(device)
print("Model loaded and moved to device.")

# ✅ 데이터로더 생성
train_loader, val_loader = create_data_loaders(train_data, val_data, tokenizer, batch_size=16)

# ✅ 데이터 확인
for batch in train_loader:
    print("Batch input IDs shape:", batch['input_ids'].shape)
    print("Batch labels shape:", batch['labels'].shape)
    break


# 데이터프레임의 인덱스를 리셋
train_data = train_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)

# 하이퍼 파라미터 후보
learning_rates = [1e-5, 1e-4, 1e-3]
num_epochs_list = [5, 10, 15]
batch_sizes = [2, 4, 8]  # 배치 크기 줄이기

best_val_loss = float("inf")  # 무한대로 초기화
best_params = None

# 손실 값을 저장할 리스트
all_train_losses = []
all_val_losses = []

# Gradient Accumulation 설정
gradient_accumulation_steps = 2

# 하이퍼파라미터 튜닝 루프
for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        for num_epochs in num_epochs_list:
            print(f"\n🚀 Training with Batch Size: {batch_size}, Learning Rate: {learning_rate}, Epochs: {num_epochs}")

            # 데이터로더 생성
            train_loader, val_loader = create_data_loaders(
                train_data=train_data,
                val_data=val_data,
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_len=64  # 토큰 최대 길이 줄이기
            )

            # 옵티마이저 정의
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # 손실 값 초기화
            train_losses = []
            val_losses = []

            # 훈련 루프
            for epoch in range(num_epochs):
                model.train()
                train_loss = 0

                for step, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
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

                # 검증 루프
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        labels = batch["labels"].to(device)

                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                        val_loss += loss.item()

                # 평균 Validation Loss 저장
                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)

                # 손실 출력
                print(f"Epoch {epoch + 1}, Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")

                # 최적의 검증 손실 값 저장
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_params = {"batch_size": batch_size, "learning_rate": learning_rate, "num_epochs": num_epochs}

            # 모든 손실 값 저장
            all_train_losses.append((batch_size, learning_rate, num_epochs, train_losses))
            all_val_losses.append((batch_size, learning_rate, num_epochs, val_losses))

            # ✅ 메모리 정리
            torch.cuda.empty_cache()

# 최적의 하이퍼 파라미터 출력
print("\n✅ 최적의 하이퍼 파라미터 조합:")
print(best_params)
print(f"최소 Validation Loss: {best_val_loss:.4f}")

# 그래프 그리기
for params, train_loss in all_train_losses:
    batch_size, learning_rate, num_epochs, losses = params
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, label=f"Train Loss (Batch: {batch_size}, LR: {learning_rate}, Epochs: {num_epochs})")
    plt.title(f"Train Loss for Batch Size {batch_size}, Learning Rate {learning_rate}, Epochs {num_epochs}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

for params, val_loss in all_val_losses:
    batch_size, learning_rate, num_epochs, losses = params
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, label=f"Validation Loss (Batch: {batch_size}, LR: {learning_rate}, Epochs {num_epochs})")
    plt.title(f"Validation Loss for Batch Size {batch_size}, Learning Rate {learning_rate}, Epochs {num_epochs}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()





