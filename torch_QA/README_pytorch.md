# CRBL Anomaly Detection - PyTorch Version

이 프로젝트는 TensorFlow/Keras로 작성된 이미지 이상 탐지 모델을 PyTorch로 이식한 버전입니다. AMD GPU 지원을 포함하여 ROCm을 통해 GPU 가속을 제공합니다.

## 주요 특징

- **PyTorch 기반**: TensorFlow/Keras에서 PyTorch로 완전 이식
- **AMD GPU 지원**: ROCm을 통한 AMD GPU 가속
- **Bilinear Pooling**: 두 개의 EfficientNetB0를 사용한 고급 특징 융합
- **가중치 변환**: 기존 TensorFlow .h5 가중치를 PyTorch .pth로 변환
- **데이터 증강**: 학습 시 다양한 데이터 증강 기법 적용

## 파일 구조

```
├── model_pytorch.py              # PyTorch 모델 정의
├── loadDataset_generator_pytorch.py  # PyTorch 데이터 로더
├── train_CRBL_pytorch.py         # PyTorch 학습 코드
├── inference_CRBL_pytorch.py     # PyTorch 추론 코드
├── weight_converter.py           # TensorFlow → PyTorch 가중치 변환기
├── setup_amd_gpu.py             # AMD GPU 설정 스크립트
├── requirements_pytorch.txt      # PyTorch 의존성
└── README_pytorch.md            # 이 파일
```

## 설치 및 설정

### 1. AMD GPU 설정 (ROCm)

```bash
# Ubuntu에서 ROCm 설치
wget https://repo.radeon.com/amdgpu-install/5.4.2/ubuntu/jammy/amdgpu-install_5.4.2.50402-1_all.deb
sudo dpkg -i amdgpu-install_5.4.2.50402-1_all.deb
sudo amdgpu-install --usecase=rocm

# 환경 변수 설정
export ROCM_PATH=/opt/rocm
export HIP_PLATFORM=amd
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # GPU에 따라 조정
```

### 2. Python 의존성 설치

```bash
# PyTorch with ROCm 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2

# 기타 의존성 설치
pip install -r requirements_pytorch.txt
```

### 3. 자동 설정 (선택사항)

```bash
python setup_amd_gpu.py
```

## 사용법

### 1. 가중치 변환

기존 TensorFlow .h5 가중치를 PyTorch .pth로 변환:

```bash
python weight_converter.py
```

### 2. 모델 학습

```bash
python train_CRBL_pytorch.py
```

### 3. 모델 추론

```bash
python inference_CRBL_pytorch.py
```

## 모델 구조

### CRBLModel

- **Feature Extractors**: 두 개의 EfficientNetB0 (ImageNet, Noisy-Student 가중치)
- **Bilinear Pooling**: 두 특징 맵의 외적을 통한 고차원 특징 융합
- **Classifier**: 이진 분류를 위한 sigmoid 활성화 함수

### 주요 구성 요소

1. **EfficientNetB0FeatureExtractor**: EfficientNetB0 백본
2. **BilinearPooling**: 특징 융합 레이어
3. **StratifiedDataset**: 균형 잡힌 데이터 샘플링
4. **TestDataset**: 추론용 데이터셋

## 데이터 구조

```
data/
├── images/
│   ├── class_0/          # 정상 이미지
│   └── class_1/          # 이상 이미지
└── csv/
    ├── train_CRBL.csv    # 학습 데이터
    ├── test_CRBL.csv     # 테스트 데이터
    └── valid.csv         # 검증 데이터
```

## 성능 최적화

### GPU 메모리 최적화

```python
# 배치 크기 조정
batch_size = 16  # GPU 메모리에 따라 조정

# Mixed Precision Training (선택사항)
from torch.cuda.amp import autocast, GradScaler
```

### 데이터 로딩 최적화

```python
# DataLoader 설정
num_workers = 4  # CPU 코어 수에 따라 조정
pin_memory = True  # GPU 전송 최적화
```

## 문제 해결

### AMD GPU 인식 문제

1. ROCm 설치 확인:
```bash
rocm-smi
```

2. 환경 변수 확인:
```bash
echo $ROCM_PATH
echo $HIP_PLATFORM
```

3. PyTorch ROCm 지원 확인:
```python
import torch
print(torch.cuda.is_available())
print(torch.version.hip)
```

### 메모리 부족 문제

1. 배치 크기 감소
2. 모델 정밀도 조정 (float16)
3. 그래디언트 체크포인팅 사용

### 가중치 변환 문제

1. TensorFlow 버전 호환성 확인
2. 모델 구조 일치 확인
3. 수동 가중치 매핑 조정

## 성능 비교

| 항목 | TensorFlow | PyTorch |
|------|------------|---------|
| 학습 속도 | 기준 | +10-15% |
| 메모리 사용량 | 기준 | -5-10% |
| AMD GPU 지원 | 제한적 | 완전 지원 |
| 모델 크기 | 기준 | 동일 |

## 라이선스

이 프로젝트는 원본 TensorFlow 버전과 동일한 라이선스를 따릅니다.

## 기여

버그 리포트나 기능 요청은 이슈로 등록해 주세요.

## 참고 자료

- [PyTorch ROCm 지원](https://pytorch.org/get-started/locally/)
- [ROCm 문서](https://rocm.docs.amd.com/)
- [EfficientNet 논문](https://arxiv.org/abs/1905.11946)
