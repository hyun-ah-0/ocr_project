import re
from datetime import datetime
from typing import List, Dict

# -----------------------------
# 1. 거래 파싱 함수
# -----------------------------

DATE_PATTERN = r'\d{2}[./-]\d{2}'
TIME_PATTERN = r'\d{2}:\d{2}'
# AMOUNT_PATTERN = r'[~+\-]?\s*\d{1,3}(,\d{3})*원'

def parse_statement_items(ocr_items: List[Dict]) -> List[Dict]:
    """
    OCR 결과에서 여러 거래를 파싱하여 리스트로 반환.
    
    Args:
        ocr_items: OCR 결과 리스트 [{"text": str, ...}, ...]
    
    Returns:
        거래 리스트 [{"date": str, "merchant": str, "amount": str}, ...]
    """
    transactions = []
    current_year = datetime.now().year
    
    # 현재 거래 정보
    current_tx = {
        "date": None,
        "time": None,
        "merchant": None,
        "amount": None
    }
    
    for item in ocr_items:
        text = item['text'].strip()

        # 1) 잔액 제거
        if "잔액" in text:
            continue
        
        # 2) 날짜 (MM.DD)를 만나면 새로운 거래 시작
        if re.fullmatch(DATE_PATTERN, text):
            # 이전 거래가 완성되었으면 저장
            if current_tx["date"] or current_tx["merchant"] or current_tx["amount"]:
                tx = _finalize_transaction(current_tx, current_year)
                if tx:  # 유효한 거래만 추가
                    transactions.append(tx)
            
            # 새 거래 시작
            text = text.replace("-", ".").replace("/", ".")
            current_tx = {
                "date": f"{current_year}.{text}",
                "time": None,
                "merchant": None,
                "amount": None
            }
            continue

        # 3) 시간
        if re.fullmatch(TIME_PATTERN, text):
            if current_tx["date"]:  # 날짜가 있는 경우에만 시간 추가
                current_tx["time"] = text
            continue
        
        # 4) 금액 (원 포함, 숫자 포함)
        if ("원" in text) and any(ch.isdigit() for ch in text):
            # 잔액이 아닌 경우에만 (잔액은 이미 필터링됨)
            if current_tx["date"]:  # 날짜가 있는 경우에만 금액 추가
                current_tx["amount"] = text
            continue
        
        # 5) 상호명 (길이가 2 이상이고, 날짜/시간/금액 패턴이 아닌 경우)
        if len(text) > 2:
            # 날짜나 시간 패턴이 아니고, 금액도 아닌 경우
            if not re.fullmatch(DATE_PATTERN, text) and not re.fullmatch(TIME_PATTERN, text):
                if "원" not in text or not any(ch.isdigit() for ch in text):
                    if current_tx["date"]:  # 날짜가 있는 경우에만 상호명 추가
                        # 이미 상호명이 있으면 업데이트하지 않음 (첫 번째가 가장 정확할 가능성)
                        if not current_tx["merchant"]:
                            current_tx["merchant"] = text
    
    # 마지막 거래 저장
    if current_tx["date"] or current_tx["merchant"] or current_tx["amount"]:
        tx = _finalize_transaction(current_tx, current_year)
        if tx:
            transactions.append(tx)
    
    return transactions


def _finalize_transaction(tx: Dict, current_year: int) -> Dict:
    """
    거래 정보를 최종 형식으로 변환.
    
    Args:
        tx: 거래 정보 딕셔너리
        current_year: 현재 연도
    
    Returns:
        최종 거래 딕셔너리 또는 None (유효하지 않은 경우)
    """
    # 날짜와 금액이 있어야 유효한 거래
    if not tx["date"] or not tx["amount"]:
        return None
    
    # 날짜 + 시간 합치기
    full_date = tx["date"]
    if tx["time"]:
        full_date = f"{full_date} {tx['time']}"
    
    return {
        "date": full_date,
        "merchant": tx["merchant"] or "미상",
        "amount": tx["amount"]
    }

