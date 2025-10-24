document.addEventListener('DOMContentLoaded', function () {
  // ===== 상태 관리 =====
  let wwtAppState = {
    hmiFilesUploaded: [],
    uploadedDataIds: [], // 업로드된 데이터 ID 목록
    inferenceTime: null,
    analysisValues: null,
    isReady: false,
  };

  // ===== DOM 요소 캐시 =====
  const elements = {
    // 파일 업로드
    hmiFileInput: document.getElementById('hmiFileInput'),
    hmiUploadConfirm: document.getElementById('hmiUploadConfirm'),
    hmiFileList: document.getElementById('hmiFileList'),
    hmiPreviewBox: document.getElementById('hmiPreviewBox'),
    selectedFileCount: document.getElementById('selectedFileCount'),

    // 입력 폼
    inferenceTime: document.getElementById('inferenceTime'),
    inferenceTimeExample: document.getElementById('inferenceTimeExample'),
    analysisValues: document.getElementById('analysisValues'),

    // 결과
    predictBtn: document.getElementById('predictBtn'),
    predictionResult: document.getElementById('predictionResult'),
    resultPlaceholder: document.getElementById('resultPlaceholder'),
  };

  // ===== 파일 다중 선택 처리 =====
  elements.hmiFileInput.addEventListener('change', handleMultipleFileSelect);

  function handleMultipleFileSelect(e) {
    const files = Array.from(e.target.files);
    
    // 최대 4개 파일만 선택 가능
    if (files.length > 4) {
      alert('최대 4개 파일까지만 선택 가능합니다.');
      elements.hmiFileInput.value = '';
      return;
    }

    wwtAppState.hmiFilesUploaded = files;
    
    // UI 업데이트
    updateFileCount();
    validateAndDisplayFiles(files);
    updateUploadButton();
  }

  function updateFileCount() {
    const count = wwtAppState.hmiFilesUploaded.length;
    elements.selectedFileCount.textContent = count;
  }

  function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  }

  function validateAndDisplayFiles(files) {
    const validationHTML = files
      .map((file, index) => {
        // 기본 검증: CSV 파일 확인
        const isCSV = file.name.endsWith('.csv');
        const isValidSize = file.size > 0 && file.size < 100 * 1024 * 1024; // 100MB 제한

        const isValid = isCSV && isValidSize;
        const statusIcon = isValid ? '✓' : '✗';
        const statusClass = isValid ? 'success' : 'error';
        const statusText = isValid ? '유효한 파일' : '유효하지 않은 파일';

        return `
          <div class="validation-item">
            <div class="validation-icon ${statusClass}">${statusIcon}</div>
            <div class="validation-content">
              <div class="validation-filename">${index + 1}. ${file.name}</div>
              <div class="validation-size">크기: ${formatFileSize(file.size)}</div>
              <div class="validation-status ${statusClass === 'error' ? 'error' : ''}">
                ${statusText}${!isCSV ? ' (CSV 파일만 지원)' : ''}${!isValidSize ? ' (파일 크기 초과)' : ''}
              </div>
            </div>
          </div>
        `;
      })
      .join('');

    elements.hmiPreviewBox.innerHTML = validationHTML || '파일을 선택해주세요.';
  }

  function updateUploadButton() {
    const allValid = wwtAppState.hmiFilesUploaded.every(
      file => file.name.endsWith('.csv') && file.size > 0
    );
    const hasFiles = wwtAppState.hmiFilesUploaded.length > 0;
    
    elements.hmiUploadConfirm.disabled = !(allValid && hasFiles);
  }

  // ===== HMI 파일 업로드 =====
  elements.hmiUploadConfirm.addEventListener('click', async () => {
    const files = wwtAppState.hmiFilesUploaded;
    
    if (files.length === 0) {
      alert('파일을 선택해주세요.');
      return;
    }

    // 각 파일을 개별적으로 업로드
    const uploadPromises = files.map(async (file) => {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('description', `WWT HMI 데이터 - ${file.name}`);
      formData.append('uploaded_by', 'user'); // 실제 사용자 ID로 교체 필요

      const res = await fetch('/WWT/upload', {
        method: 'POST',
        body: formData
      });

      if (res.ok) {
        const data = await res.json();
        return data.data_id;
      } else {
        const errorData = await res.json();
        throw new Error(errorData.error || 'Upload failed');
      }
    });

    try {
      const dataIds = await Promise.all(uploadPromises);
      wwtAppState.uploadedDataIds = dataIds;
      
      console.log('HMI 파일 업로드 성공:', dataIds);

      // UI 업데이트
      elements.inferenceTime.disabled = false;
      elements.analysisValues.disabled = false;

      // 파일 목록 좌측 폼에 표시
      const fileCount = files.length;
      elements.hmiFileList.innerHTML = `
        <small class="text-success">
          <i class="fas fa-check-circle"></i>
          업로드된 파일: ${fileCount}/4
        </small>
      `;

      // 모달 닫기
      const modal = bootstrap.Modal.getInstance(document.getElementById('hmiUploadModal'));
      if (modal) modal.hide();

      // 파일 입력 초기화
      elements.hmiFileInput.value = '';
      elements.selectedFileCount.textContent = '0';

      alert('HMI 데이터가 성공적으로 업로드되었습니다.');
    } catch (err) {
      console.error('업로드 오류:', err);
      alert('업로드 실패: ' + err.message);
    }
  });

  // ===== 추론일시 입력 처리 =====
  elements.inferenceTime.addEventListener('change', (e) => {
    const value = e.target.value;
    if (value) {
      // datetime-local 값을 YYYY-MM-DD HH:MM:SS 형식으로 변환
      const [date, time] = value.split('T');
      const formattedTime = `${date} ${time}`;
      
      // 예시 텍스트 업데이트
      elements.inferenceTimeExample.textContent = formattedTime;
      
      wwtAppState.inferenceTime = formattedTime;
      updatePredictButtonState();
    }
  });

  // ===== DO 분석값 입력 처리 =====
  elements.analysisValues.addEventListener('input', (e) => {
    const value = parseFloat(e.target.value);
    if (!isNaN(value) && value >= 0) {
      wwtAppState.analysisValues = value;
      updatePredictButtonState();
    }
  });

  function updatePredictButtonState() {
    const isReady =
      wwtAppState.uploadedDataIds.length > 0 &&
      wwtAppState.inferenceTime &&
      wwtAppState.analysisValues !== null;

    elements.predictBtn.disabled = !isReady;
  }

  // ===== 추론 버튼 =====
  elements.predictBtn.addEventListener('click', async () => {
    const payload = {
      data_ids: wwtAppState.uploadedDataIds,
      target_time: wwtAppState.inferenceTime,
    };

    try {
      const res = await fetch('/WWT/predict_do', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        const data = await res.json();
        displayPredictionResult(data);
      } else {
        const errorData = await res.json();
        alert('추론 실패: ' + (errorData.error || 'Server error'));
      }
    } catch (err) {
      console.error('추론 오류:', err);
      alert('서버 연결 실패: ' + err.message);
    }
  });

  function displayPredictionResult(data) {
    elements.resultPlaceholder.style.display = 'none';
    elements.predictionResult.classList.remove('d-none');

    // 새로운 API 응답 형식에 맞게 수정
    const predictions = data.predictions || {};
    
    elements.predictionResult.innerHTML = `
      <div class="prediction-results">
        <h5 class="fw-bold mb-4">🎯 추론 결과</h5>
        
        <div class="prediction-card">
          <div class="result-metric">
            <span class="result-metric-label">추론일시</span>
            <span class="result-metric-value">${data.target_time || 'N/A'}</span>
          </div>
          <div class="result-metric">
            <span class="result-metric-label">사용된 데이터</span>
            <span class="result-metric-value">${data.data_ids_used?.length || 0}개 파일</span>
          </div>
        </div>

        <h6 class="fw-bold mt-4 mb-3">목표값별 송풍량 예측</h6>
        <div class="prediction-card">
          <div class="result-metric">
            <span class="result-metric-label">실제 DO</span>
            <span class="result-metric-value">${predictions.actual_do?.toFixed(2) ?? '계산 중...'} mg/L</span>
          </div>
        </div>
        
        <div class="prediction-card">
          <div class="result-metric">
            <span class="result-metric-label">목표값 96%</span>
            <span class="result-metric-value">${predictions.do_96?.toFixed(2) ?? '계산 중...'} mg/L</span>
          </div>
        </div>
        
        <div class="prediction-card">
          <div class="result-metric">
            <span class="result-metric-label">목표값 97%</span>
            <span class="result-metric-value">${predictions.do_97?.toFixed(2) ?? '계산 중...'} mg/L</span>
          </div>
        </div>
        
        <div class="prediction-card">
          <div class="result-metric">
            <span class="result-metric-label">목표값 98%</span>
            <span class="result-metric-value">${predictions.do_98?.toFixed(2) ?? '계산 중...'} mg/L</span>
          </div>
        </div>

        ${data.result_file ? `
          <div class="mt-4">
            <h6 class="fw-bold mb-3">📊 결과 파일</h6>
            <div class="alert alert-info">
              <i class="fas fa-file-csv"></i> 결과가 저장되었습니다: ${data.result_file}
            </div>
          </div>
        ` : ''}
      </div>
    `;
  }

  // 초기 상태 설정
  elements.inferenceTime.disabled = true;
  elements.analysisValues.disabled = true;
  elements.predictBtn.disabled = true;
});
