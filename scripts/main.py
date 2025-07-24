import os
import sys
import json
from typing import Type, TypeVar
from dataclasses import dataclass, field, fields

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
from tqdm import tqdm
import random
from matplotlib import pyplot as plt

# main.py 맨 위에 추가
import sys
sys.path.insert(0, r"C:\Users\wonseok\cocoapi\PythonAPI")

# 이제 정상 import
from pycocotools.coco import COCO
import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.Unet import UNet

# ✅ COCO 데이터셋 클래스 추가
class COCOClassificationDataset(Dataset):
    """
    COCO 데이터셋을 분류 작업용으로 변환하는 클래스
    각 이미지에서 가장 면적이 큰 객체의 클래스를 해당 이미지의 라벨로 사용
    """
    
    def __init__(self, root_dir, annotation_file, transform=None, max_classes=80, min_area=1000):
        """
        Args:
            root_dir: 이미지 폴더 경로 (예: 'coco/images/train2017')
            annotation_file: 어노테이션 파일 (예: 'coco/annotations/instances_train2017.json')
            transform: 이미지 전처리 변환
            max_classes: 사용할 최대 클래스 수
            min_area: 최소 객체 면적 (작은 객체 제외)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.max_classes = max_classes
        self.min_area = min_area
        
        print(f"📂 COCO 데이터셋 로딩 중...")
        print(f"   📍 이미지 경로: {root_dir}")
        print(f"   📄 어노테이션: {annotation_file}")
        
        # COCO API 초기화
        self.coco = COCO(annotation_file)
        
        # 카테고리 정보 가져오기
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.category_mapping = self._create_category_mapping()
        
        # 이미지-라벨 데이터 준비
        self.image_data = []
        self._prepare_dataset()
        
        print(f"✅ 데이터셋 준비 완료:")
        print(f"   📊 총 이미지 수: {len(self.image_data)}")
        print(f"   🏷️  클래스 수: {len(self.category_mapping)}")
    
    def _create_category_mapping(self):
        """COCO 카테고리 ID를 연속된 클래스 ID로 매핑"""
        # 가장 빈번한 클래스들 선택 (max_classes만큼)
        category_counts = {}
        ann_ids = self.coco.getAnnIds()
        
        print(f"📊 카테고리 빈도 분석 중... (전체 어노테이션: {len(ann_ids)}개)")
        
        for ann_id in ann_ids[:10000]:  # 샘플링으로 빠르게 계산
            ann = self.coco.loadAnns([ann_id])[0]
            cat_id = ann['category_id']
            category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
        
        # 빈도순으로 정렬하여 상위 클래스들 선택
        sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        top_categories = [cat_id for cat_id, count in sorted_cats[:self.max_classes]]
        
        # 연속된 ID로 매핑
        mapping = {cat_id: idx for idx, cat_id in enumerate(top_categories)}
        
        print(f"📋 선택된 상위 {len(mapping)}개 카테고리:")
        for cat_id, class_id in mapping.items():
            cat_info = self.coco.loadCats([cat_id])[0]
            print(f"   {class_id}: {cat_info['name']} (ID: {cat_id})")
        
        return mapping
    
    def _prepare_dataset(self):
        """이미지-라벨 쌍 데이터 준비"""
        print(f"🔄 데이터셋 준비 중...")
        
        # 선택된 카테고리의 어노테이션만 가져오기
        selected_cat_ids = list(self.category_mapping.keys())
        ann_ids = self.coco.getAnnIds(catIds=selected_cat_ids)
        
        # 이미지별로 어노테이션 그룹화
        image_annotations = {}
        
        for ann_id in tqdm(ann_ids, desc="어노테이션 처리"):
            ann = self.coco.loadAnns([ann_id])[0]
            img_id = ann['image_id']
            cat_id = ann['category_id']
            area = ann['area']
            
            # 너무 작은 객체는 제외
            if area < self.min_area:
                continue
                
            if cat_id in self.category_mapping:
                if img_id not in image_annotations:
                    image_annotations[img_id] = []
                
                image_annotations[img_id].append({
                    'category_id': self.category_mapping[cat_id],
                    'area': area,
                    'bbox': ann['bbox']
                })
        
        # 각 이미지의 주요 객체 결정 (가장 큰 면적)
        for img_id, annotations in tqdm(image_annotations.items(), desc="이미지 라벨 생성"):
            if not annotations:
                continue
                
            # 이미지 정보 가져오기
            img_info = self.coco.loadImgs([img_id])[0]
            img_path = os.path.join(self.root_dir, img_info['file_name'])
            
            # 이미지 파일이 존재하는지 확인
            if not os.path.exists(img_path):
                continue
            
            # 가장 큰 객체의 클래스를 라벨로 선택
            largest_annotation = max(annotations, key=lambda x: x['area'])
            
            self.image_data.append({
                'image_path': img_path,
                'label': largest_annotation['category_id'],
                'image_id': img_id,
                'area': largest_annotation['area']
            })
    
    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, idx):
        item = self.image_data[idx]
        
        # 이미지 로드
        try:
            image = Image.open(item['image_path']).convert('RGB')
        except Exception as e:
            print(f"⚠️ 이미지 로드 실패: {item['image_path']}")
            # 더미 이미지 생성
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # 전처리 적용
        if self.transform:
            image = self.transform(image)
        
        return image, item['label']
    
    def get_class_names(self):
        """클래스 이름 목록 반환"""
        class_names = [''] * len(self.category_mapping)
        for cat_id, class_id in self.category_mapping.items():
            cat_info = self.coco.loadCats([cat_id])[0]
            class_names[class_id] = cat_info['name']
        return class_names

T = TypeVar('T')

def parse_config(config_path: str, cls: Type[T]) -> T:
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    cls_fields = {field.name for field in fields(cls)}
    filtered_data = {k: v for k, v in config_data.items() if k in cls_fields}
    
    return cls(**filtered_data)

@dataclass
class Config:
    model_name: str = field(
        default="UNet_COCO",
        metadata={"help": "The name of the model to use."}
    )

    hidden_size: int = field(
        default=768,
        metadata={"help": "The size of the hidden layer."}
    )

    n_classes: int = field(
        default=10,  # COCO에서 선택할 클래스 수 (조정 가능)
        metadata={"help": "The number of classes to classify."}
    )

    batch_size: int = field(
        default=16,  # COCO 이미지가 크므로 배치 사이즈 줄임
        metadata={"help": "The batch size to use."}
    )

    opt_name: str = field(
        default="adam",
        metadata={"help": "The name of the optimizer to use."}
    )

    lr: float = field(
        default=1e-4,  # 큰 데이터셋에 맞게 학습률 조정
        metadata={"help": "The learning rate to use."}
    )

    num_epochs: int = field(
        default=20,  # 큰 데이터셋이므로 에포크 줄임
        metadata={"help": "The number of epochs to train for."}
    )

    early_stopping: int = field(
        default=5,
        metadata={"help": "The number of epochs to wait before early stopping."}
    )

    p: float = field(
        default=0.1,
        metadata={"help": "The dropout rate to use."}
    )
    
    # ✅ COCO 관련 설정 추가
    coco_data_dir: str = field(
        default="./coco",
        metadata={"help": "COCO dataset directory path"}
    )
    
    image_size: int = field(
        default=224,
        metadata={"help": "Input image size"}
    )

def train(model, train_loader, criterion, optimizer, device):
    epoch_loss = 0
    correct = 0
    total = 0
    
    model.train()    
    for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc="Training")):  
        x = x.to(device)
        y = y.to(device)
            
        optimizer.zero_grad()                
        y_pred, _ = model(x)  
        
        loss = criterion(y_pred, y)  
        loss.backward()        
        optimizer.step()        
        
        epoch_loss += loss.item()
        
        _, predicted = torch.max(y_pred, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
        
    epoch_loss /= len(train_loader)  
    epoch_acc = correct / total
       
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    epoch_loss = 0
    correct = 0
    total = 0
    
    model.eval()    
    with torch.no_grad():        
        for batch_idx, (x, y) in enumerate(tqdm(val_loader, desc="Validation")):  
            x = x.to(device)
            y = y.to(device)
            y_pred, _ = model(x) 
            loss = criterion(y_pred, y)  

            epoch_loss += loss.item()
            
            _, predicted = torch.max(y_pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        
    epoch_loss /= len(val_loader)  
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

loss_train = []
loss_val = []

def main():
    config = Config()

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        config = parse_config(sys.argv[1], Config)

    os.makedirs(f"{config.model_name}_logs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 사용 디바이스: {device}")

    # ✅ 모델 생성
    model = UNet(config, output_dim=config.n_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    print(f"📦 모델: {config.model_name}, 클래스 수: {config.n_classes}")

    # ✅ COCO용 전처리 (ImageNet 정규화 사용)
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),  # 데이터 증강
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet 정규화
    ])

    # 검증용 전처리 (데이터 증강 제외)
    val_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    # ✅ COCO 데이터셋 경로 설정
    train_img_dir = os.path.join(config.coco_data_dir, 'images', 'train2017')
    val_img_dir = os.path.join(config.coco_data_dir, 'images', 'val2017')
    train_ann_file = os.path.join(config.coco_data_dir, 'annotations', 'instances_train2017.json')
    val_ann_file = os.path.join(config.coco_data_dir, 'annotations', 'instances_val2017.json')

    # 파일 존재 확인
    required_paths = [train_img_dir, val_img_dir, train_ann_file, val_ann_file]
    missing_paths = [path for path in required_paths if not os.path.exists(path)]
    
    if missing_paths:
        print("❌ COCO 데이터셋 파일이 없습니다:")
        for path in missing_paths:
            print(f"   - {path}")
        print("\n💡 COCO 데이터셋을 다운로드하세요:")
        print("   1. https://cocodataset.org/#download")
        print("   2. 또는 스크립트로 자동 다운로드")
        return

    # ✅ COCO 데이터셋 생성
    print("📊 COCO 훈련 데이터셋 준비 중...")
    train_dataset = COCOClassificationDataset(
        root_dir=train_img_dir,
        annotation_file=train_ann_file,
        transform=transform,
        max_classes=config.n_classes
    )
    
    print("\n📊 COCO 검증 데이터셋 준비 중...")
    val_dataset = COCOClassificationDataset(
        root_dir=val_img_dir,
        annotation_file=val_ann_file,
        transform=val_transform,
        max_classes=config.n_classes
    )

    # 데이터로더 생성
    trainloader = DataLoader(train_dataset, batch_size=config.batch_size,  
                            shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(val_dataset, batch_size=config.batch_size,  
                           shuffle=False, num_workers=4, pin_memory=True)

    # 클래스 이름 가져오기
    class_names = train_dataset.get_class_names()
    print(f"\n🏷️  클래스 목록: {class_names}")
    
    best_valid_loss = float('inf')  
    best_valid_acc = 0.0
    
    print(f"\n🚀 훈련 시작! (총 {config.num_epochs}개 에포크)")
    print(f"   📊 훈련 데이터: {len(train_dataset)}개")
    print(f"   📊 검증 데이터: {len(val_dataset)}개")
    
    for epoch in range(config.num_epochs):
        print(f"\n📅 Epoch {epoch+1}/{config.num_epochs}")
        print("-" * 60)
        
        train_loss, train_acc = train(model, trainloader, criterion, optimizer, device)
        valid_loss, valid_acc = validate(model, testloader, criterion, device)
        
        if valid_acc > best_valid_acc:  
            best_valid_acc = valid_acc
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'{config.model_name}_logs/best_model.pth')
            print(f"💾 최고 성능 모델 저장됨 (정확도: {best_valid_acc*100:.2f}%)")

        loss_train.append(train_loss)
        loss_val.append(valid_loss)

        print(f'🔸 Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:6.2f}%')
        print(f'🔹 Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc*100:6.2f}%')
        
        # Early Stopping
        if config.early_stopping and epoch > config.early_stopping:
            if valid_acc < best_valid_acc:
                print("🛑 Early stopping 적용됨")
                break
    
    # 결과 시각화
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(loss_train, 'b-', label='Train Loss')
    plt.plot(loss_val, 'r-', label='Val Loss')    
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    train_acc_list = [1 - (loss_train[i] / max(loss_train)) for i in range(len(loss_train))]
    val_acc_list = [1 - (loss_val[i] / max(loss_val)) for i in range(len(loss_val))]
    plt.plot(train_acc_list, 'b-', label='Train Acc (approx)')
    plt.plot(val_acc_list, 'r-', label='Val Acc (approx)')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{config.model_name}_logs/training_curves.png')
    plt.show()
    
    print(f"\n🎉 훈련 완료!")
    print(f"   🏆 최고 검증 정확도: {best_valid_acc*100:.2f}%")
    print(f"   💾 모델 저장 위치: {config.model_name}_logs/best_model.pth")

if __name__ == "__main__":
    main()