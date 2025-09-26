# CRBL Anomaly Detection - PyTorch Implementation

이 프로젝트는 TensorFlow로 구현된 CRBL (Crop-based anomaly detection) 모델을 PyTorch로 변환한 것입니다. 이미지에서 anomaly를 감지하며, impurity가 있는 경우 1, 없는 경우 0으로 분류합니다.

## 주요 특징

- **TensorFlow에서 PyTorch로 완전 변환**: 기존 TensorFlow 모델을 PyTorch로 변환
- **가중치 변환**: TensorFlow .weights.h5 파일을 PyTorch .pth 형식으로 변환
- **재현성 보장**: 시드 설정을 통한 완전한 재현성
- **배치 추론**: 효율적인 배치 처리 지원
- **이중 모델 구조**: Crop 모델과 Full 모델을 결합한 앙상블 방식

## 모델 아키텍처

- **백본**: 두 개의 EfficientNet-B0 (ImageNet, Noisy-Student)
- **Bilinear Pooling**: 두 백본의 특징을 결합
- **분류기**: 이진 분류를 위한 sigmoid 활성화 함수

## 설치

### 요구사항

- Python 3.12.9
- PyTorch 1.12.0+ (ROCm 지원 버전)
- TensorFlow 2.9.0 (가중치 변환용)

### 패키지 설치

#### AMD GPU (ROCm) 환경
```bash
# ROCm 지원 PyTorch 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

# 나머지 패키지 설치
pip install -r requirements.txt
```

#### NVIDIA GPU (CUDA) 환경
```bash
# CUDA 지원 PyTorch 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 나머지 패키지 설치
pip install -r requirements.txt
```

#### CPU 전용 환경
```bash
# CPU 전용 PyTorch 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 나머지 패키지 설치
pip install -r requirements.txt
```

## 사용법

### 1. 전체 파이프라인 실행

```bash
python main.py --mode all
```

### 2. 개별 단계 실행

#### 가중치 변환만
```bash
python main.py --mode convert
```

#### 훈련만
```bash
python main.py --mode train --epochs 30 --batch_size 32 --learning_rate 0.0001
```

#### 추론만
```bash
python main.py --mode inference
```

### 3. 고급 사용법

```bash
# 사용자 정의 경로로 추론
python main.py --mode inference \
    --crop_model_path ./weights_pytorch/crop_model_pytorch.pth \
    --full_model_path ./weights_pytorch/full_model_pytorch.pth \
    --test_csv_path ./data/csv/valid.csv \
    --image_dir ./data/images

# 다른 시드로 재현성 테스트
python main.py --mode all --seed 123
```

## 파일 구조

```
├── main.py                    # 메인 실행 스크립트
├── pytorch_model.py          # PyTorch 모델 정의
├── pytorch_dataset.py        # 데이터 로더 및 전처리
├── train_pytorch.py          # 훈련 스크립트
├── inference_pytorch.py      # 추론 스크립트
├── weight_converter.py       # TensorFlow → PyTorch 가중치 변환
├── utils.py                  # 유틸리티 함수
├── requirements.txt          # 필요한 패키지 목록
└── README.md                 # 이 파일
```

## 데이터 구조

프로젝트는 다음 데이터 구조를 가정합니다:

```
data/
├── images/
│   ├── class_0/              # 정상 이미지
│   ├── class_1/              # 이상 이미지
│   └── [원본 이미지들]
└── csv/
    ├── train_CRBL.csv        # 훈련 데이터
    ├── test_CRBL.csv         # 테스트 데이터
    └── valid.csv             # 검증 데이터
```

## 출력 파일

실행 후 다음 디렉토리에 결과가 저장됩니다:

```
crbl_pytorch_outputs/
├── weights/                  # 훈련된 모델 가중치
├── results/                  # 추론 결과
├── logs/                     # 훈련 로그
├── plots/                    # 시각화 결과
└── configs/                  # 설정 파일
```

## 재현성

모든 실행에서 동일한 결과를 얻기 위해 다음이 보장됩니다:

- **시드 설정**: Python, NumPy, PyTorch 시드 고정
- **결정적 알고리즘**: PyTorch 결정적 알고리즘 사용
- **CUDA 설정**: CUDA 결정적 동작 활성화

## 성능 최적화

- **배치 처리**: 효율적인 배치 추론
- **GPU 가속**: CUDA/ROCm/MPS 지원 (AMD GPU 포함)
- **메모리 최적화**: 적절한 배치 크기 설정
- **멀티프로세싱**: 데이터 로딩 병렬화

## AMD GPU 지원

이 구현은 AMD GPU에서 ROCm을 통해 GPU 가속을 지원합니다:

### ROCm 설치 요구사항
- AMD GPU (RDNA2 이상 권장)
- ROCm 5.6+ 설치
- ROCm 지원 PyTorch 설치

### AMD GPU 확인
```bash
# ROCm 설치 확인
rocm-smi

# PyTorch에서 GPU 인식 확인
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"

# 환경 정보 확인
python utils.py
```

### AMD GPU 성능 최적화
```bash
# 환경 변수 설정 (선택사항)
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export ROCM_PATH=/opt/rocm

# GPU 메모리 모니터링
python -c "from utils import print_gpu_memory_info; print_gpu_memory_info()"
```

## 모델 성능

### 훈련 설정
- **배치 크기**: 32
- **학습률**: 0.0001
- **에포크**: 30
- **조기 종료**: 10 에포크 patience
- **학습률 스케줄링**: ReduceLROnPlateau

### 클래스 가중치
- **Class 0**: 0.88
- **Class 1**: 0.12

## 문제 해결

### 일반적인 문제

1. **GPU 메모리 부족**
   ```bash
   # 배치 크기 줄이기
   python main.py --mode train --batch_size 16
   ```

2. **AMD GPU 인식 안됨**
   ```bash
   # ROCm 설치 확인
   rocm-smi
   
   # ROCm 지원 PyTorch 재설치
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
   ```

3. **가중치 변환 실패**
   ```bash
   # TensorFlow 버전 확인
   pip install tensorflow==2.9.0
   ```

4. **데이터 경로 오류**
   - `data/` 디렉토리 구조 확인
   - CSV 파일 경로 확인

### 로그 확인

```bash
# 최신 로그 파일 확인
tail -f ./logs/crbl_pytorch_*.log
```

## 기여

버그 리포트나 기능 요청은 이슈로 등록해 주세요.

## 라이선스

이 프로젝트는 원본 TensorFlow 구현을 기반으로 합니다.

## 참고사항

- 원본 TensorFlow 모델과의 호환성을 위해 동일한 아키텍처를 유지했습니다
- 가중치 변환 시 일부 레이어 이름이 변경될 수 있습니다
- 성능 차이가 있을 수 있으니 검증을 권장합니다
