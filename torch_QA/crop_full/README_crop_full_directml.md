# CRBL Anomaly Detection - PyTorch with DirectML (Crop & Full Models)

이 프로젝트는 TensorFlow/Keras로 작성된 이미지 이상 탐지 모델을 PyTorch로 이식하고, Windows에서 AMD GPU를 사용하기 위해 DirectML을 활용한 버전입니다. **Crop 모델과 Full 모델을 모두 지원**합니다.

## 주요 특징

- **PyTorch 기반**: TensorFlow/Keras에서 PyTorch로 완전 이식
- **DirectML 지원**: Windows에서 AMD GPU 가속 (ROCm 대신)
- **Dual Model 지원**: Crop 모델과 Full 모델을 모두 지원
- **Bilinear Pooling**: 두 개의 EfficientNetB0를 사용한 고급 특징 융합
- **가중치 변환**: 기존 TensorFlow .h5 가중치를 PyTorch .pth로 변환
- **데이터 증강**: 학습 시 다양한 데이터 증강 기법 적용

## 모델 타입 설명

### Crop Model (is_full=False)
- **용도**: 크롭된 이미지에 특화된 이상 탐지
- **특징**: 150번째 레이어 이후가 **학습 가능** (trainable=True)
- **임계값**: 0.01 (CROPPED_THRESHOLD)
- **파일명**: `CRBL_250328_pytorch_directml_crop.pth`

### Full Model (is_full=True)
- **용도**: 전체 이미지에 특화된 이상 탐지
- **특징**: 150번째 레이어 이후가 **고정** (trainable=False)
- **임계값**: 0.5 (FULL_THRESHOLD)
- **파일명**: `CRBL_250328_pytorch_directml_full.pth`

## 설치 및 설정

### 1. DirectML 설치

```bash
# PyTorch with DirectML 설치
pip install torch torchvision torchaudio torch-directml

# 기타 의존성 설치
pip install -r requirements_directml.txt
```

### 2. 자동 설정

```bash
python setup_amd_gpu_directml.py
```

## 사용법

### 1. 모델 학습

#### Crop 모델 학습
```python
# train_CRBL_pytorch_directml.py 또는 train_CRBL_pytorch_directml.ipynb에서
MODEL_TYPE = "crop"  # 설정
```

#### Full 모델 학습
```python
# train_CRBL_pytorch_directml.py 또는 train_CRBL_pytorch_directml.ipynb에서
MODEL_TYPE = "full"  # 설정
```

### 2. 모델 추론

#### Crop 모델 추론
```python
# inference_CRBL_pytorch_directml.py 또는 inference_CRBL_pytorch_directml.ipynb에서
MODEL_TYPE = "crop"  # 설정
```

#### Full 모델 추론
```python
# inference_CRBL_pytorch_directml.py 또는 inference_CRBL_pytorch_directml.ipynb에서
MODEL_TYPE = "full"  # 설정
```

## 파일 구조

```
├── model_pytorch_dual_directml.py          # Dual 모델 정의 (Crop & Full)
├── train_CRBL_pytorch_directml.py          # DirectML 학습 스크립트
├── train_CRBL_pytorch_directml.ipynb       # DirectML 학습 노트북
├── inference_CRBL_pytorch_directml.py      # DirectML 추론 스크립트
├── inference_CRBL_pytorch_directml.ipynb   # DirectML 추론 노트북
├── setup_amd_gpu_directml.py              # DirectML 설정 스크립트
├── requirements_directml.txt               # DirectML 의존성
└── README_crop_full_directml.md            # 이 파일
```

## 모델 구조

### CRBLDualModel

- **Feature Extractors**: 두 개의 EfficientNetB0 (ImageNet, Noisy-Student 가중치)
- **Bilinear Pooling**: 두 특징 맵의 외적을 통한 고차원 특징 융합
- **Classifier**: 이진 분류를 위한 sigmoid 활성화 함수
- **DirectML 지원**: Windows AMD GPU 자동 감지 및 사용
- **is_full 매개변수**: Crop/Full 모델 구분

### 주요 구성 요소

1. **EfficientNetB0FeatureExtractor**: EfficientNetB0 백본 (is_full 매개변수 지원)
2. **BilinearPooling**: 특징 융합 레이어
3. **get_device()**: DirectML > CUDA > CPU 순서로 최적 디바이스 선택
4. **create_crop_model()**: Crop 모델 생성 함수
5. **create_full_model()**: Full 모델 생성 함수

## 모델별 차이점

| 항목 | Crop Model | Full Model |
|------|------------|------------|
| is_full | False | True |
| 150번째 레이어 이후 | 학습 가능 | 고정 |
| 임계값 | 0.01 | 0.5 |
| 용도 | 크롭된 이미지 | 전체 이미지 |
| 학습 파라미터 | 많음 | 적음 |

## 성능 비교

| 항목 | ROCm | DirectML |
|------|------|----------|
| 설치 난이도 | 어려움 | 쉬움 |
| Windows 지원 | 제한적 | 완전 지원 |
| 성능 | 높음 | 중간 |
| 안정성 | 높음 | 높음 |
| 메모리 사용량 | 낮음 | 중간 |

## 사용 예시

### 1. Crop 모델 학습
```bash
# 스크립트 실행
python train_CRBL_pytorch_directml.py
# MODEL_TYPE = "crop"으로 설정

# 또는 노트북 실행
jupyter notebook train_CRBL_pytorch_directml.ipynb
# MODEL_TYPE = "crop"으로 설정
```

### 2. Full 모델 학습
```bash
# 스크립트 실행
python train_CRBL_pytorch_directml.py
# MODEL_TYPE = "full"으로 설정

# 또는 노트북 실행
jupyter notebook train_CRBL_pytorch_directml.ipynb
# MODEL_TYPE = "full"으로 설정
```

### 3. 추론 실행
```bash
# Crop 모델 추론
python inference_CRBL_pytorch_directml.py
# MODEL_TYPE = "crop"으로 설정

# Full 모델 추론
python inference_CRBL_pytorch_directml.py
# MODEL_TYPE = "full"으로 설정
```

## 문제 해결

### DirectML 관련 문제

1. **DirectML 인식 실패**:
```python
import torch_directml
print(torch_directml.is_available())
```

2. **드라이버 문제**:
   - AMD GPU 드라이버 최신 버전 설치
   - Windows 업데이트 확인

3. **메모리 부족**:
   - 배치 크기 감소
   - 모델 정밀도 조정

### 모델 타입 관련 문제

1. **잘못된 모델 타입**:
   - MODEL_TYPE은 "crop" 또는 "full"만 가능
   - 대소문자 구분

2. **가중치 파일 경로**:
   - Crop 모델: `CRBL_250328_pytorch_directml_crop.pth`
   - Full 모델: `CRBL_250328_pytorch_directml_full.pth`

## 권장사항

### Windows 사용자
1. **DirectML 사용** (설치 간단, 안정적)
2. AMD GPU 드라이버 최신 버전 유지
3. Windows 10/11 사용

### 모델 선택
1. **Crop 모델**: 크롭된 이미지에 특화된 경우
2. **Full 모델**: 전체 이미지에 특화된 경우
3. **두 모델 모두 학습**: 다양한 시나리오 대응

## 성능 최적화

### DirectML 최적화
```python
# 배치 크기 조정
batch_size = 16  # GPU 메모리에 따라 조정

# Mixed Precision Training (선택사항)
from torch.cuda.amp import autocast, GradScaler
```

## 라이선스

이 프로젝트는 원본 TensorFlow 버전과 동일한 라이선스를 따릅니다.

## 기여

버그 리포트나 기능 요청은 이슈로 등록해 주세요.

## 참고 자료

- [PyTorch DirectML](https://pytorch.org/get-started/locally/)
- [DirectML 문서](https://docs.microsoft.com/en-us/windows/ai/directml/)
- [EfficientNet 논문](https://arxiv.org/abs/1905.11946)
