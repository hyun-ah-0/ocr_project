import re

# -----------------------------
# 1. 카테고리/키워드 정의
# -----------------------------

CATEGORIES = ["식비", "카페/간식", "쇼핑", "교통", "생활/공과금", "이체", "수입", "기타"]

# 상호명 안에 들어갈 수 있는 키워드들
MERCHANT_KEYWORDS = {
    "카페/간식": [
        "스타벅스", "투썸", "메가커피", "이디야", "폴바셋", "빽다방", "할리스",
        "공차", "던킨", "배스킨"
    ],
    "식비": [
        "김밥", "버거", "맥도날드", "롯데리아", "버거킹", "피자", "치킨",
        "한식", "분식", "도시락", "설렁탕", "국수", "포차"
    ],
    "교통": [
        "택시", "고속", "버스", "KTX", "SRT", "지하철", "코레일", "티머니"
    ],
    "쇼핑": [
        "쿠팡", "마켓컬리", "무신사", "11번가", "G마켓", "옥션", "SSG", "위메프",
        "카카오스타일", "지그재그", "브랜디", "네이버페이", "컬리페이", "올리브"
    ],
    "생활/공과금": [
        "도시가스", "전기요금", "수도요금", "관리비", "통신", "SKT", "KT", "LGU",
        "인터넷", "아파트관리", "경조사"
    ],
    "이체": [
        "카뱅오픈", "카카오뱅크", "이체", "송금", "계좌이체", "무통장입금", "입금"
    ],
}

# -----------------------------
# 2. 상호명 정리 함수
# -----------------------------

LEGAL_PREFIXES = ["(주)", "주식회사", "유한회사", "(유)"]
STORE_SUFFIXES = ["점", "지점", "영업소"]

def normalize_merchant_name(name: str) -> str:
    """
    상호명에서 (주), 주식회사, ~~점 같은 노이즈를 제거하고
    대소문자/공백을 정리한 버전 반환.
    """
    if not name:
        return ""

    text = name.strip()

    # 1. 앞쪽 법인표시 제거
    for prefix in LEGAL_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # 2. 뒤쪽 'OO점' 같은 표현 일부 정리 (예: '스타벅스 강남역점' → '스타벅스 강남역')
    #    너무 과하게 자르면 정보가 사라지니, '점'으로 끝날 때만 1글자 제거
    for suffix in STORE_SUFFIXES:
        if text.endswith(suffix):
            text = text[:-len(suffix)].strip()

    # 3. 연속 공백 하나로 줄이기
    text = re.sub(r"\s+", " ", text)

    return text

# -----------------------------
# 3. 규칙 기반 카테고리 분류
# -----------------------------

def rule_based_category(transaction: dict) -> str:
    merchant_raw = transaction.get("merchant", "") or ""
    merchant = normalize_merchant_name(merchant_raw)
    amount_str = transaction.get("amount", "") or ""
    
    # 수입 관련 키워드 확인 (이자, 캐시백 등)
    income_keywords = ["이자", "캐시백", "적립", "리워드", "환불", "취소"]
    if any(keyword in merchant for keyword in income_keywords):
        return "수입"
    
    # 금액이 양수인 경우 (수입)
    # amount_str에서 + 기호나 양수 패턴 확인
    if amount_str.startswith("+") or (not amount_str.startswith("-") and not amount_str.startswith("~")):
        # 이체 관련 키워드가 있으면서 양수면 수입
        if any(word in merchant for word in ["이체", "송금", "입금", "카뱅오픈", "카카오뱅크"]):
            return "수입"
        # 이자, 캐시백 등이 명시적으로 있으면 수입
        if any(keyword in merchant for keyword in income_keywords):
            return "수입"
    
    # 일반 카테고리 분류
    for category, keywords in MERCHANT_KEYWORDS.items():
        for kw in keywords:
            if kw and kw in merchant:
                # 이체 키워드인데 금액이 양수면 수입으로 처리
                if category == "이체" and (amount_str.startswith("+") or (not amount_str.startswith("-") and not amount_str.startswith("~"))):
                    return "수입"
                return category

    # 편의점
    if any(word in merchant for word in ["CU", "GS25", "세븐일레븐", "이마트24", "편의점"]):
        return "식비"

    # 배달앱
    if any(word in merchant for word in ["배달의민족", "요기요", "배민", "쿠팡이츠"]):
        return "식비"

    return "기타"

# -----------------------------
# 4. 여러 거래 한 번에 태깅하는 함수
# -----------------------------

from typing import List, Dict

def categorize_transactions_rule_based(transactions: List[Dict]) -> List[Dict]:
    """
    여러 거래에 대해 rule_based_category를 적용하고,
    각 dict에 'category_rule' 필드를 추가해서 반환.
    """
    categorized = []
    for tx in transactions:
        tx_copy = tx.copy()
        tx_copy["category_rule"] = rule_based_category(tx_copy)
        categorized.append(tx_copy)
    return categorized


def categorize_transactions_with_llm(transactions: List[Dict]) -> List[Dict]:
    """
    여러 거래에 대해 하이브리드 카테고리 분류를 적용.
    규칙 기반으로 먼저 분류하고, '기타'인 경우 LLM으로 보완.
    각 dict에 'category_rule' 필드를 추가해서 반환.
    """
    categorized = []
    for tx in transactions:
        tx_copy = tx.copy()
        tx_copy["category_rule"] = hybrid_category(tx_copy)
        categorized.append(tx_copy)
    return categorized

# -----------------------------
# 5. LLM 기반 카테고리 분류
# -----------------------------

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
llm_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def build_category_prompt(transaction: dict) -> str:
    """
    거래 정보를 기반으로 카테고리 분류를 위한 프롬프트 생성.
    """
    merchant = transaction.get("merchant", "미상")
    amount = transaction.get("amount", "")
    date = transaction.get("date", "")
    
    prompt = f"""
다음은 카드 거래 내역입니다. 이 거래를 다음 카테고리 중 하나로 분류해주세요.

카테고리 목록:
- 식비: 식당, 음식점, 배달, 편의점 음식 등
- 카페/간식: 카페, 커피숍, 디저트, 간식 등
- 쇼핑: 온라인 쇼핑몰, 백화점, 의류, 생활용품 등
- 교통: 택시, 버스, 지하철, 기차, 주유소 등
- 생활/공과금: 전기, 가스, 수도, 통신비, 관리비 등
- 이체: 계좌이체, 송금, 입금, 카뱅오픈, 카카오뱅크 등 계좌 간 이동 (단, 금액이 양수(+)이거나 수입인 경우는 "수입" 카테고리)
- 수입: 이자, 캐시백, 적립, 리워드, 환불, 취소 등 수입성 거래 (금액이 양수(+)인 경우)
- 기타: 위 카테고리에 해당하지 않는 모든 거래

거래 정보:
- 상호명: {merchant}
- 금액: {amount}
- 날짜: {date}

중요: 금액이 양수(+)로 시작하거나 "이자", "캐시백" 등의 키워드가 있으면 "수입" 카테고리로 분류하세요.

위 거래를 가장 적합한 카테고리 하나만 선택해서 JSON 형식으로만 답해주세요.

예시:
{{ "category": "식비" }}

설명 없이 JSON 한 줄만 출력해주세요.
"""
    return prompt.strip()


def llm_category(transaction: dict) -> str:
    """
    LLM을 사용하여 거래의 카테고리를 분류.
    API 키가 없거나 오류 발생 시 "기타" 반환.
    """
    if not llm_client:
        return "기타"
    
    prompt = build_category_prompt(transaction)
    
    try:
        response = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # 일관성 있는 분류를 위해 낮은 temperature
        )
        
        raw_output = response.choices[0].message.content.strip()
        
        # JSON 파싱
        if "```json" in raw_output:
            raw_output = raw_output.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_output:
            raw_output = raw_output.split("```")[1].split("```")[0].strip()
        
        result = json.loads(raw_output)
        category = result.get("category", "기타")
        
        # 유효한 카테고리인지 확인
        if category in CATEGORIES:
            return category
        else:
            return "기타"
            
    except Exception as e:
        print(f"[WARN] LLM 카테고리 분류 실패: {e}")
        return "기타"


def hybrid_category(transaction: dict) -> str:
    """
    하이브리드 카테고리 분류:
    1) 규칙 기반으로 먼저 분류
    2) '기타'인 경우에만 LLM으로 보완
    """
    rule_cat = rule_based_category(transaction)
    if rule_cat != "기타":
        return rule_cat
    
    # 규칙 기반으로 분류되지 않은 경우 LLM 사용
    return llm_category(transaction)
