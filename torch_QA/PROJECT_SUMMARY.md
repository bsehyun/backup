# CRBL Anomaly Detection - TensorFlow to PyTorch Migration

## 프로젝트 개요

이 프로젝트는 TensorFlow/Keras로 작성된 이미지 이상 탐지 모델을 PyTorch로 완전 이식한 버전입니다. AMD GPU 지원을 포함하여 ROCm을 통해 GPU 가속을 제공합니다.

## 변환된 파일 목록

### 1. 핵심 모델 파일
- **`model_pytorch.py`**: PyTorch 모델 정의 (CRBLModel, EfficientNetB0FeatureExtractor, BilinearPooling)
- **`loadDataset_generator_pytorch.py`**: PyTorch DataLoader 및 데이터 전처리
- **`train_CRBL_pytorch.py`**: PyTorch 학습 스크립트
- **`inference_CRBL_pytorch.py`**: PyTorch 추론 스크립트

### 2. Jupyter 노트북
- **`train_CRBL_pytorch.ipynb`**: 학습용 노트북
- **`inference_CRBL_pytorch.ipynb`**: 추론용 노트북

### 3. 유틸리티 파일
- **`weight_converter.py`**: TensorFlow .h5 → PyTorch .pth 가중치 변환기
- **`setup_amd_gpu.py`**: AMD GPU 설정 자동화 스크립트
- **`requirements_pytorch.txt`**: PyTorch 의존성 목록
- **`README_pytorch.md`**: 상세한 사용법 및 설정 가이드

## 주요 변환 내용

### 1. 모델 구조 변환
- **TensorFlow/Keras** → **PyTorch nn.Module**
- EfficientNetB0 두 개를 사용한 bilinear pooling 구조 유지
- 가중치 동결/해제 로직 구현
- 커스텀 bilinear pooling 레이어 구현

### 2. 데이터 처리 변환
- **Keras ImageDataGenerator** → **PyTorch DataLoader**
- **tf.keras.utils.Sequence** → **torch.utils.data.Dataset**
- 데이터 증강을 위한 torchvision.transforms 사용
- 배치 처리 및 멀티프로세싱 지원

### 3. 학습 루프 변환
- **model.fit()** → **커스텀 학습 루프**
- Early stopping, Learning rate scheduling 구현
- 클래스 가중치를 통한 불균형 데이터 처리
- 메트릭 계산 (정확도, 정밀도, 재현율, F1)

### 4. 추론 시스템 변환
- **model.predict()** → **커스텀 추론 클래스**
- 단일 이미지 및 배치 추론 지원
- 이미지 전처리 파이프라인 구현
- 결과 시각화 및 분석

## AMD GPU 지원

### ROCm 설정
- PyTorch ROCm 5.4.2 지원
- 환경 변수 자동 설정
- GPU 감지 및 설정 검증

### 설치 명령어
```bash
# PyTorch with ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2

# 자동 설정
python setup_amd_gpu.py
```

## 가중치 변환

### 변환 과정
1. TensorFlow .h5 파일에서 가중치 추출
2. 레이어 이름 매핑 (TensorFlow → PyTorch)
3. 가중치 형태 변환 (HWC → CHW)
4. PyTorch 모델에 로드 및 검증

### 사용법
```bash
python weight_converter.py
```

## 성능 최적화

### 메모리 최적화
- 배치 크기 조정 가능
- Mixed precision training 지원 (선택사항)
- 그래디언트 체크포인팅 (선택사항)

### 데이터 로딩 최적화
- 멀티프로세싱 지원
- 메모리 핀닝
- 비동기 데이터 로딩

## 사용법

### 1. 환경 설정
```bash
pip install -r requirements_pytorch.txt
python setup_amd_gpu.py
```

### 2. 가중치 변환
```bash
python weight_converter.py
```

### 3. 모델 학습
```bash
python train_CRBL_pytorch.py
# 또는
jupyter notebook train_CRBL_pytorch.ipynb
```

### 4. 모델 추론
```bash
python inference_CRBL_pytorch.py
# 또는
jupyter notebook inference_CRBL_pytorch.ipynb
```

## 호환성

### Python 버전
- Python 3.10.2 (권장)
- Python 3.8+ 지원

### PyTorch 버전
- PyTorch 2.0.0+ (ROCm 지원)
- torchvision 0.15.0+
- torchaudio 2.0.0+

### GPU 지원
- AMD GPU (ROCm 5.4.2+)
- NVIDIA GPU (CUDA 지원)
- CPU 전용 실행 가능

## 파일 구조

```
anomaly_detector/
├── model_pytorch.py                    # PyTorch 모델 정의
├── loadDataset_generator_pytorch.py    # 데이터 로더
├── train_CRBL_pytorch.py              # 학습 스크립트
├── train_CRBL_pytorch.ipynb           # 학습 노트북
├── inference_CRBL_pytorch.py          # 추론 스크립트
├── inference_CRBL_pytorch.ipynb       # 추론 노트북
├── weight_converter.py                # 가중치 변환기
├── setup_amd_gpu.py                   # AMD GPU 설정
├── requirements_pytorch.txt            # 의존성 목록
├── README_pytorch.md                   # 상세 가이드
└── PROJECT_SUMMARY.md                  # 이 파일
```

## 주요 개선사항

1. **AMD GPU 지원**: ROCm을 통한 완전한 AMD GPU 가속
2. **모듈화**: 각 기능별로 분리된 모듈 구조
3. **유연성**: 다양한 설정 옵션 제공
4. **호환성**: 기존 TensorFlow 가중치 재사용 가능
5. **성능**: PyTorch의 최적화된 연산 활용

## 문제 해결

### 일반적인 문제
1. **GPU 인식 문제**: `setup_amd_gpu.py` 실행
2. **메모리 부족**: 배치 크기 감소
3. **가중치 변환 실패**: 모델 구조 확인

### 지원
- 상세한 오류 메시지 제공
- 단계별 문제 해결 가이드
- 성능 모니터링 도구

## 라이선스

원본 TensorFlow 프로젝트와 동일한 라이선스를 따릅니다.

---

**변환 완료일**: 2024년
**PyTorch 버전**: 2.0.0+
**ROCm 버전**: 5.4.2+
**Python 버전**: 3.10.2
