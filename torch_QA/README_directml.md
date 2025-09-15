# CRBL Anomaly Detection - PyTorch with DirectML (Windows AMD GPU)

이 프로젝트는 TensorFlow/Keras로 작성된 이미지 이상 탐지 모델을 PyTorch로 이식하고, Windows에서 AMD GPU를 사용하기 위해 DirectML을 활용한 버전입니다.

## 주요 특징

- **PyTorch 기반**: TensorFlow/Keras에서 PyTorch로 완전 이식
- **DirectML 지원**: Windows에서 AMD GPU 가속 (ROCm 대신)
- **Bilinear Pooling**: 두 개의 EfficientNetB0를 사용한 고급 특징 융합
- **가중치 변환**: 기존 TensorFlow .h5 가중치를 PyTorch .pth로 변환
- **데이터 증강**: 학습 시 다양한 데이터 증강 기법 적용

## DirectML vs ROCm

### DirectML (권장 - Windows)
- **장점**: 설치가 간단, Windows 네이티브 지원
- **단점**: Windows 전용, 일부 연산에서 성능 제한
- **설치**: `pip install torch-directml`

### ROCm (Linux/Windows)
- **장점**: 완전한 GPU 가속, Linux에서 안정적
- **단점**: 복잡한 설치 과정, Windows에서 제한적
- **설치**: 복잡한 드라이버 및 SDK 설치 필요

## 설치 및 설정

### 1. DirectML 설치 (권장)

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

### 3. ROCm 설치 (대안)

Windows에서 ROCm을 사용하려면:

```bash
# ROCm Windows 설치 가이드
python setup_amd_gpu_windows.py
```

## 사용법

### 1. DirectML 버전 (권장)

```bash
# 학습
python train_CRBL_pytorch_directml.py

# 추론
python inference_CRBL_pytorch_directml.py
```

### 2. ROCm 버전

```bash
# 학습
python train_CRBL_pytorch.py

# 추론
python inference_CRBL_pytorch.py
```

## 파일 구조

```
├── model_pytorch_directml.py          # DirectML 지원 모델
├── train_CRBL_pytorch_directml.py     # DirectML 학습 코드
├── inference_CRBL_pytorch_directml.py # DirectML 추론 코드
├── setup_amd_gpu_directml.py         # DirectML 설정 스크립트
├── requirements_directml.txt          # DirectML 의존성
├── README_directml.md                 # 이 파일
└── [기존 ROCm 버전 파일들...]
```

## 모델 구조

### CRBLModel (DirectML 버전)

- **Feature Extractors**: 두 개의 EfficientNetB0 (ImageNet, Noisy-Student 가중치)
- **Bilinear Pooling**: 두 특징 맵의 외적을 통한 고차원 특징 융합
- **Classifier**: 이진 분류를 위한 sigmoid 활성화 함수
- **DirectML 지원**: Windows AMD GPU 자동 감지 및 사용

### 주요 구성 요소

1. **EfficientNetB0FeatureExtractor**: EfficientNetB0 백본
2. **BilinearPooling**: 특징 융합 레이어
3. **get_device()**: DirectML > CUDA > CPU 순서로 최적 디바이스 선택
4. **StratifiedDataset**: 균형 잡힌 데이터 샘플링
5. **TestDataset**: 추론용 데이터셋

## 성능 비교

| 항목 | ROCm | DirectML |
|------|------|----------|
| 설치 난이도 | 어려움 | 쉬움 |
| Windows 지원 | 제한적 | 완전 지원 |
| 성능 | 높음 | 중간 |
| 안정성 | 높음 | 높음 |
| 메모리 사용량 | 낮음 | 중간 |

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

### ROCm 관련 문제

1. **ROCm 설치 실패**:
   - Windows 11 권장
   - 관리자 권한으로 설치
   - 시스템 재시작

2. **GPU 인식 실패**:
   - ROCm 지원 GPU 확인
   - 환경 변수 설정

## 권장사항

### Windows 사용자
1. **DirectML 사용** (설치 간단, 안정적)
2. AMD GPU 드라이버 최신 버전 유지
3. Windows 10/11 사용

### Linux 사용자
1. **ROCm 사용** (완전한 GPU 가속)
2. Ubuntu 20.04+ 권장
3. ROCm 5.4.2+ 사용

## 성능 최적화

### DirectML 최적화
```python
# 배치 크기 조정
batch_size = 16  # GPU 메모리에 따라 조정

# Mixed Precision Training (선택사항)
from torch.cuda.amp import autocast, GradScaler
```

### ROCm 최적화
```python
# 환경 변수 설정
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HIP_PLATFORM=amd
```

## 라이선스

이 프로젝트는 원본 TensorFlow 버전과 동일한 라이선스를 따릅니다.

## 기여

버그 리포트나 기능 요청은 이슈로 등록해 주세요.

## 참고 자료

- [PyTorch DirectML](https://pytorch.org/get-started/locally/)
- [DirectML 문서](https://docs.microsoft.com/en-us/windows/ai/directml/)
- [ROCm 문서](https://rocm.docs.amd.com/)
- [EfficientNet 논문](https://arxiv.org/abs/1905.11946)
