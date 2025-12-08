# app/multimodal.py

from __future__ import annotations
from typing import Dict, Optional
import os
import json
import base64

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)

CATEGORIES = ["식비", "카페/간식", "쇼핑", "교통", "생활/공과금", "기타"]


def build_multimodal_category_prompt(transaction: Dict) -> str:
    """
    거래 정보를 텍스트로 설명하고, 카테고리 하나를 JSON으로 고르게 하는 프롬프트.
    (이미지는 responses API input_image로 따로 넣는다)
    """
    payload = {
        "date": transaction.get("date"),
        "merchant": transaction.get("merchant"),
        "amount": transaction.get("amount"),
    }
    payload_str = json.dumps(payload, ensure_ascii=False)

    prompt = f"""
아래는 카드 명세서의 한 거래 내역이다.

{payload_str}

카테고리 후보:
{CATEGORIES}

이 거래를 가장 잘 설명하는 카테고리를 한 개만 선택해서
다음 JSON 형식으로만 답해라.

예시:
{{ "category": "쇼핑" }}

설명 문장 없이, JSON 한 줄만 출력해라.
"""
    return prompt.strip()


def classify_transaction_multimodal(
    image_path: str,
    transaction: Dict,
    model: str = "gpt-4o-mini",  # vision 지원 모델
) -> Optional[str]:
    """
    로컬 카드명세서 이미지 + 거래 텍스트를 함께 넣어서
    멀티모달 LLM이 카테고리를 고르게 한다.
    """

    if not OPENAI_API_KEY:
        print("[WARN] OPENAI_API_KEY가 없어서 multimodal 분류를 건너뜁니다.")
        return None

    prompt = build_multimodal_category_prompt(transaction)

    # 1) 로컬 이미지 → base64 data URL
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:image/png;base64,{image_b64}"  # jpg면 image/jpeg

    # 2) Chat Completions API 멀티모달 호출
    #    일반적인 Vision API 형식 사용
    try:
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
        )
    except Exception as e:
        print("[ERROR] chat.completions.create 실패:", e)
        return None

    # 3) 텍스트 출력 추출
    try:
        raw = response.choices[0].message.content
    except (AttributeError, IndexError) as e:
        print("[ERROR] 응답 파싱 실패:", e)
        return None

    raw = raw.strip()
    # print("RAW LLM OUTPUT:", raw)  # 디버깅용

    # 4) JSON 파싱
    try:
        obj = json.loads(raw)
        cat = obj.get("category")
        return cat
    except Exception:
        print("[WARN] JSON 파싱 실패, 원문:", raw)
        return None
