# app/image_preprocessor.py

"""
이미지 전처리 파이프라인
OpenCV를 활용한 이미지 보정으로 OCR 성능 향상
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from io import BytesIO
from PIL import Image


def preprocess_image(
    image_bytes: bytes,
    method: str = "adaptive"
) -> bytes:
    """
    이미지 전처리 파이프라인
    
    Args:
        image_bytes: 원본 이미지 바이트
        method: 전처리 방법 ("none", "basic", "adaptive", "enhanced")
            - "none": 전처리 없음
            - "basic": 그레이스케일 + Otsu 이진화
            - "adaptive": 적응형 이진화 + 노이즈 제거
            - "enhanced": 향상된 전처리 (스케일링 + 샤프닝 + 이진화)
    
    Returns:
        전처리된 이미지 바이트
    """
    if method == "none":
        return image_bytes
    
    # 이미지 로드
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is None:
        # PIL로 재시도
        img_pil = Image.open(BytesIO(image_bytes))
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    if method == "basic":
        processed = _basic_preprocessing(img)
    elif method == "adaptive":
        processed = _adaptive_preprocessing(img)
    elif method == "enhanced":
        processed = _enhanced_preprocessing(img)
    else:
        processed = img
    
    # 바이트로 변환
    _, encoded_img = cv2.imencode('.png', processed)
    return encoded_img.tobytes()


def _basic_preprocessing(img: np.ndarray) -> np.ndarray:
    """
    기본 전처리: 그레이스케일 + Otsu 이진화
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def _adaptive_preprocessing(img: np.ndarray) -> np.ndarray:
    """
    적응형 전처리: 그레이스케일 + 적응형 이진화 + 노이즈 제거
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 적응형 이진화 (조명 변화에 강함)
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # 노이즈 제거
    denoised = cv2.fastNlMeansDenoising(adaptive_thresh, None, 10, 7, 21)
    
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)


def _enhanced_preprocessing(img: np.ndarray) -> np.ndarray:
    """
    향상된 전처리: 스케일링 + 샤프닝 + 이진화
    """
    # 1. 크기 조정 (너무 작은 이미지 확대)
    h, w = img.shape[:2]
    if h < 1000 or w < 1000:
        scale = max(1500 / h, 1500 / w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 2. 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. 샤프닝 필터 적용 (텍스트 선명도 향상)
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    
    # 4. 대비 향상 (CLAHE - Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(sharpened)
    
    # 5. 적응형 이진화
    adaptive_thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    # 6. 모폴로지 연산으로 노이즈 제거
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
    
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

