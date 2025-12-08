# 카드 명세서 OCR 분석 서비스

## 📐 Architecture Diagram

```
[User] 
  ↓ (이미지 업로드)
[HTML/JS Frontend]
  ↓ (HTTP POST /api/analyze)
[FastAPI Server]
  ↓ (이미지 바이트)
[Image Preprocessing]
  ├─ Grayscale Conversion
  ├─ Scaling (if needed)
  ├─ Sharpening
  ├─ CLAHE (Contrast Enhancement)
  ├─ Adaptive Thresholding
  └─ Morphological Operations
  ↓ (전처리된 이미지)
[VLM OCR Model (OpenAI GPT-4o-mini)]
  ↓ (Raw OCR Text with Bounding Boxes)
[Post-processing]
  ├─ Transaction Parsing (RegEx 기반)
  │   ├─ 날짜 패턴 매칭 (MM.DD)
  │   ├─ 시간 패턴 매칭 (HH:MM)
  │   ├─ 금액 패턴 매칭 (숫자 + "원")
  │   └─ 상호명 추출
  ├─ Category Classification
  │   ├─ Rule-based Classification
  │   └─ LLM-based Classification (기타 카테고리)
  └─ Data Aggregation
  ↓ (구조화된 JSON)
[JSON Output]
  ├─ transactions: [{date, merchant, amount, category}]
  ├─ summary: {total_spent, total_income, by_category_expense}
  └─ llm_summary: "자연어 요약"
```

## 🚀 서비스 실행 방법

### 1. 환경 설정

`.env` 파일에 OpenAI API 키를 설정하세요:
```
OPENAI_API_KEY=your_api_key_here
```

### 2. 가상환경 설정 (선택사항)

프로젝트를 처음 실행하는 경우 가상환경을 생성하고 패키지를 설치하세요:

```bash
# 가상환경 생성 (Windows)
python -m venv ocr_project

# 가상환경 활성화 (Windows)
ocr_project\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```


### 3. 서버 실행

```bash
# 가상환경 활성화 (Windows)
ocr_project\Scripts\activate

# FastAPI 서버 실행
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. 웹 브라우저에서 접속

서버가 실행되면 브라우저에서 다음 주소로 접속하세요:
```
http://localhost:8000
```

## 🛠️ Key Features & Tech Stack

### 기술 스택

- **백엔드**: FastAPI
- **OCR 엔진**: OpenAI Vision API (GPT-4o-mini)
- **이미지 전처리**: OpenCV (cv2)
- **프론트엔드**: 순수 HTML/CSS/JavaScript
- **LLM**: OpenAI GPT-4o-mini (카테고리 분류 및 자연어 요약)
- **데이터 처리**: Pandas, NumPy
- **패턴 매칭**: Python RegEx

### 주요 기능

1. **다중 이미지 업로드**
   - 드래그 앤 드롭 또는 클릭하여 여러 이미지 동시 업로드
   - PNG, JPG, JPEG 형식 지원

2. **고급 이미지 전처리 파이프라인**
   - Enhanced 전처리 (기본 사용)
     - 이미지 스케일링 (작은 이미지 확대)
     - 그레이스케일 변환
     - 샤프닝 필터 적용
     - CLAHE 대비 향상
     - 적응형 이진화
     - 모폴로지 연산 (노이즈 제거)

3. **VLM OCR 분석**
   - OpenAI GPT-4o-mini를 사용한 고정밀 OCR
   - 한국어/영어 텍스트 인식
   - Bounding box 정보 제공

4. **정보 추출 (Information Extraction)**
   - OCR 결과에서 구조화된 거래 데이터 추출
   - 날짜, 시간, 상호명, 금액 자동 파싱
   - 정규표현식 기반 패턴 매칭

5. **하이브리드 카테고리 분류**
   - 규칙 기반 자동 분류 (1차)
   - LLM 기반 분류 (기타 카테고리 보완)
   - 카테고리: 식비, 카페/간식, 쇼핑, 교통, 생활/공과금, 이체, 수입, 기타

6. **데이터 요약 및 시각화**
   - 총 지출/수입 통계
   - 카테고리별 지출 막대 그래프
   - LLM 기반 자연어 요약 (5-7문장, 주요 수치 강조)

7. **거래 내역 표시**
   - 날짜, 상호명, 금액, 카테고리 정보 표시
   - 카테고리별 색상 구분

## 📝 API 엔드포인트

### POST `/api/analyze`
단일 카드 명세서 이미지를 분석합니다.

**요청:**
- Content-Type: `multipart/form-data`
- 파일: `file` (이미지 파일)

**응답:**
```json
{
  "transactions": [
    {
      "date": "2025.12.04 00:02",
      "merchant": "(주)카카오스타일",
      "amount": "-31,410원",
      "category_rule": "쇼핑"
    }
  ],
  "summary": {
    "total_spent": 31410,
    "total_income": 0,
    "by_category_expense": {
      "쇼핑": {
        "amount": 31410,
        "ratio": 1.0
      }
    }
  },
  "llm_summary": "요약 텍스트..."
}
```

### POST `/api/analyze-multiple`
여러 카드 명세서 이미지를 동시에 분석하고 결과를 집계합니다.

**요청:**
- Content-Type: `multipart/form-data`
- 파일: `files` (이미지 파일 배열)

**응답:** `/api/analyze`와 동일한 형식 (모든 이미지의 거래 내역 집계)

## 🔍 문제 해결 과정 (Trouble Shooting)

### 1. 이미지 전처리 파이프라인 개발

#### 문제 상황
초기 OCR 성능이 낮았으며, 특히 다음 문제들이 발생했습니다:
- 노이즈가 많은 이미지에서 텍스트 인식률 저하
- 작은 해상도 이미지에서 문자 오인식
- 조명 변화에 따른 이진화 실패
- 구겨진 부분에서 숫자 오인식 (예: '8'을 'B'로 인식)

#### 해결 과정

**단계별 전처리 파이프라인 구축:**

1. **Grayscale Conversion (그레이스케일 변환)**
   - 컬러 이미지를 그레이스케일로 변환하여 처리 속도 향상 및 노이즈 감소

2. **Image Scaling (이미지 스케일링)**
   - 작은 이미지(1000px 미만)를 1500px로 확대하여 해상도 향상
   - `cv2.resize()` with `INTER_LINEAR` interpolation 사용

3. **Sharpening (샤프닝)**
   - 3x3 커널을 사용한 샤프닝 필터 적용
   - 텍스트 경계를 선명하게 하여 인식률 향상
   ```python
   kernel = np.array([[-1, -1, -1],
                      [-1,  9, -1],
                      [-1, -1, -1]])
   sharpened = cv2.filter2D(gray, -1, kernel)
   ```

4. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
   - 대비 향상을 통한 텍스트 가독성 개선
   - `clipLimit=2.0`, `tileGridSize=(8, 8)` 파라미터 사용

5. **Adaptive Thresholding (적응형 이진화)**
   - 조명 변화에 강한 이진화 적용
   - `ADAPTIVE_THRESH_GAUSSIAN_C` 방식 사용
   - 블록 크기: 11, 상수: 2

6. **Morphological Operations (모폴로지 연산)**
   - `MORPH_CLOSE` 연산으로 노이즈 제거 및 텍스트 연결
   - 2x2 커널 사용
   - **핵심 해결**: 구겨진 부분의 숫자 오인식 문제 해결
     - 예: '8'을 'B'로 오인식하던 문제를 모폴로지 연산으로 글자 획을 뚜렷하게 만들어 해결

#### 성능 개선 결과

Enhanced 전처리 적용 시 (전처리 없음 대비):
- **텍스트 정확도**: 17.3% 향상 (0.133 → 0.156)
- **문자 정확도**: 8.3% 향상 (0.785 → 0.850)
- **Precision**: 5.5% 향상 (0.800 → 0.844)
- **Recall**: 12.4% 향상 (0.828 → 0.931)
- **F1 Score**: 8.7% 향상 (0.814 → 0.885)
- **매칭된 텍스트**: 3개 증가 (24/29 → 27/29)

> **참고**: Adaptive 전처리는 오히려 성능이 저하되었으므로 사용하지 않습니다.

### 2. 정보 추출 (Information Extraction) 로직 개발

#### 문제 상황
OCR 결과는 단순 텍스트 리스트였으며, 구조화된 거래 데이터로 변환하는 과정에서 다음 문제들이 발생했습니다:
- 여러 거래가 하나의 이미지에 포함된 경우 구분 실패
- 날짜 형식 다양성 (MM.DD, MM/DD, MM-DD)
- 금액 형식 다양성 (10,000원, 10.000원, ~10,000원 등)
- 잔액 정보와 거래 금액 구분 실패

#### 해결 과정

**후처리(Post-processing) 로직 구현:**

1. **Transaction Parsing (거래 파싱)**
   - 정규표현식 기반 패턴 매칭
     - 날짜: `r'\d{2}[./-]\d{2}'` (MM.DD, MM/DD, MM-DD 모두 지원)
     - 시간: `r'\d{2}:\d{2}'` (HH:MM)
     - 금액: `"원"` 포함 + 숫자 포함 텍스트
   - 상태 기반 파싱: 날짜를 만나면 새로운 거래 시작
   - 잔액 필터링: "잔액" 키워드 포함 텍스트 제거

2. **금액 정규화**
   - 특수문자 처리: `~`, `+`, `-` 제거
   - 구분자 통일: `.` → `,` 변환 (예: 10.000 → 10,000)
   - 후처리 로직으로 오인식된 특수문자 보정

3. **다중 거래 처리**
   - 날짜 패턴을 만나면 이전 거래 저장 및 새 거래 시작
   - 거래 완성 조건: 날짜 + 금액 필수, 상호명/시간 선택

**예시:**
```python
# OCR Raw Text
["12.04", "00:02", "(주)카카오스타일", "~31,410원", "12.02", "13:04", "오가다 제주공항점", "-6,500원"]

# → 구조화된 데이터
[
  {"date": "2025.12.04 00:02", "merchant": "(주)카카오스타일", "amount": "-31,410원"},
  {"date": "2025.12.02 13:04", "merchant": "오가다 제주공항점", "amount": "-6,500원"}
]
```

### 3. 카테고리 분류 개선

#### 문제 상황
- 규칙 기반 분류만으로는 "기타" 카테고리가 과도하게 생성됨
- 맥락을 이해하지 못해 잘못된 분류 발생

#### 해결 과정

**하이브리드 분류 시스템 구축:**

1. **1차: Rule-based Classification**
   - 상호명 키워드 기반 분류
   - 카테고리: 식비, 카페/간식, 쇼핑, 교통, 생활/공과금, 이체, 수입

2. **2차: LLM-based Classification**
   - 규칙 기반에서 "기타"로 분류된 경우에만 LLM 호출
   - GPT-4o-mini를 사용하여 맥락 기반 분류
   - 프롬프트에 거래 정보(상호명, 금액)와 카테고리 목록 제공

**성능 개선:**
- "기타" 카테고리 비율 감소
- 맥락을 고려한 정확한 분류 (예: "카뱅오픈" → "이체")

### 4. 다중 이미지 처리 및 집계

#### 문제 상황
- 여러 이미지를 업로드해도 요약 정보가 마지막 이미지 결과만 반영됨
- 거래 내역이 누적되지 않음

#### 해결 과정

1. **거래 내역 누적**
   - `/api/analyze-multiple` 엔드포인트에서 모든 이미지의 거래 내역 수집

2. **월별 집계 로직 수정**
   - `monthly_summary` 함수에서 `month=None`일 때 모든 거래 집계
   - 카테고리별 금액 및 비율 계산

3. **프론트엔드 수정**
   - 다중 파일 선택 지원 (`multiple` 속성 추가)
   - 파일 미리보기 기능 추가

## 🔧 일반적인 문제 해결

### 서버가 시작되지 않는 경우
- Python 가상환경이 활성화되어 있는지 확인
- 필요한 패키지가 설치되어 있는지 확인: `pip install fastapi uvicorn openai python-dotenv pillow opencv-python pandas numpy`

### OCR이 작동하지 않는 경우
- `.env` 파일에 `OPENAI_API_KEY`가 올바르게 설정되어 있는지 확인
- API 키에 충분한 크레딧이 있는지 확인

### CORS 오류가 발생하는 경우
- `main.py`의 CORS 설정이 올바른지 확인
- 브라우저 콘솔에서 정확한 오류 메시지 확인
- HTML 파일을 FastAPI 서버를 통해 제공 (직접 파일 열기 대신 `http://localhost:8000` 접속)

### 전처리 성능 측정

성능 평가를 실행하려면:
```bash
# 가상환경 활성화 후
python evaluate_performance.py
```

**주의사항:**
- 개인정보 보호를 위해 `data/raw/` 폴더의 이미지 파일은 GitHub에 업로드되지 않습니다.
- 로컬에서 평가를 수행하려면 `data/raw/` 폴더에 테스트 이미지를 추가하고, `data/ground_truth/` 폴더에 해당하는 Ground Truth JSON 파일을 준비하세요.
- 이미지 파일이 없는 경우 스크립트는 안내 메시지를 출력하고 종료됩니다.
- 기존 평가 결과는 `PERFORMANCE_EVALUATION.md` 파일을 참고하세요.

결과는 `performance_results.json` 파일에 저장되며, `PERFORMANCE_EVALUATION.md`에 상세 결과가 기록됩니다.


## 📚 참고 문서

- [성능 평가 보고서](./PERFORMANCE_EVALUATION.md)
- [OCR 서비스 코드](./app/ocr_service.py)
- [이미지 전처리 코드](./app/image_preprocessor.py)
- [거래 파싱 코드](./app/parser.py)
- [카테고리 분류 코드](./app/categorizer.py)
