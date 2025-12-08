# app/ocr_service.py

from __future__ import annotations
from typing import List, Dict, Any, Optional
import os
import json
import base64
from io import BytesIO

from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
from app.image_preprocessor import preprocess_image

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def build_ocr_prompt() -> str:
    """
    VLM OCR을 위한 프롬프트 생성.
    카드 명세서 이미지에서 모든 텍스트를 추출하고 구조화된 형식으로 반환하도록 요청.
    """
    prompt = """
이 이미지는 카드 명세서입니다. 이미지에 있는 모든 텍스트를 추출해주세요.

다음 JSON 형식으로 응답해주세요:
[
  {"text": "추출된 텍스트", "confidence": 0.95},
  ...
]

주의사항:
1. 모든 텍스트를 위에서 아래로, 왼쪽에서 오른쪽 순서로 추출하세요
2. 각 텍스트는 별도의 객체로 분리하세요 (예: "12.04"와 "00:02"는 별도 항목)
3. confidence는 0.0~1.0 사이의 값으로 텍스트 인식 확신도를 나타냅니다 (불확실하면 0.8 정도로 설정)
4. JSON 형식만 출력하고 다른 설명은 하지 마세요
5. 텍스트가 없는 경우 빈 배열 []을 반환하세요
6. 숫자, 날짜, 시간, 상호명, 금액 등 모든 텍스트를 빠짐없이 추출하세요
"""
    return prompt.strip()


def image_to_base64(image_bytes: bytes) -> str:
    """
    이미지 바이트를 base64 인코딩된 data URL로 변환.
    """
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    # 이미지 형식 자동 감지
    try:
        img = Image.open(BytesIO(image_bytes))
        img_format = img.format.lower() if img.format else "png"
        mime_type = f"image/{img_format}"
    except Exception:
        mime_type = "image/png"
    
    return f"data:{mime_type};base64,{image_b64}"


def run_vlm_ocr(
    image_bytes: bytes,
    model: str = "gpt-4o-mini",  # 또는 "gpt-4o" (더 정확하지만 비쌈)
) -> List[Dict[str, Any]]:
    """
    VLM (Vision Language Model)을 사용하여 OCR 수행.
    
    Args:
        image_bytes: 이미지 바이트 데이터
        model: 사용할 OpenAI Vision 모델 (gpt-4o-mini 또는 gpt-4o)
    
    Returns:
        EasyOCR과 동일한 형식의 OCR 결과 리스트
        [{"text": str, "bbox": [[x1, y1], ...], "confidence": float}, ...]
    """
    if not client:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY를 추가하세요.")
    
    prompt = build_ocr_prompt()
    image_url = image_to_base64(image_bytes)
    
    try:
        # OpenAI Vision API 호출
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            },
                        },
                    ],
                }
            ],
            max_tokens=4000,  # OCR 결과가 길 수 있으므로 충분한 토큰 할당
        )
    except Exception as e:
        print(f"[ERROR] VLM OCR API 호출 실패: {e}")
        raise
    
    # 응답에서 텍스트 추출
    try:
        raw_output = response.choices[0].message.content.strip()
    except (AttributeError, IndexError) as e:
        print(f"[ERROR] 응답 파싱 실패: {e}")
        raise
    
    # JSON 파싱 시도
    try:
        # JSON 코드 블록이 있는 경우 제거
        if "```json" in raw_output:
            raw_output = raw_output.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_output:
            raw_output = raw_output.split("```")[1].split("```")[0].strip()
        
        ocr_results = json.loads(raw_output)
        
        # 결과 형식 검증 및 정규화
        normalized_results = []
        for item in ocr_results:
            if isinstance(item, dict) and "text" in item:
                text = item["text"].strip()
                if not text:  # 빈 텍스트는 건너뛰기
                    continue
                
                # bbox가 없으면 빈 리스트로 설정 (VLM은 bbox를 제공하지 않을 수 있음)
                bbox = item.get("bbox", [])
                # confidence가 없으면 기본값 0.9로 설정
                confidence = item.get("confidence", 0.9)
                
                normalized_results.append({
                    "text": text,
                    "bbox": bbox,
                    "confidence": float(confidence)
                })
        
        return normalized_results
        
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON 파싱 실패. 원문 출력:")
        print(raw_output[:500])  # 처음 500자만 출력
        raise ValueError(f"VLM OCR 응답을 JSON으로 파싱할 수 없습니다: {e}")
    except Exception as e:
        print(f"[ERROR] OCR 결과 처리 중 오류: {e}")
        raise


def run_ocr(
    image_bytes: bytes,
    preprocess: bool = True,
    preprocess_method: str = "enhanced"
) -> List[Dict[str, Any]]:
    """
    메인 OCR 함수. VLM OCR을 사용합니다.
    
    Args:
        image_bytes: 이미지 바이트 데이터
        preprocess: 전처리 적용 여부
        preprocess_method: 전처리 방법 ("none", "basic", "adaptive", "enhanced")
    
    Returns:
        OCR 결과 리스트 (EasyOCR 형식과 호환)
    """
    # 전처리 적용
    if preprocess:
        image_bytes = preprocess_image(image_bytes, method=preprocess_method)
    
    return run_vlm_ocr(image_bytes)

