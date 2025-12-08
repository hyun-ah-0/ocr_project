# app/reporter.py

from __future__ import annotations
from typing import List, Dict, Optional
from datetime import datetime
import re
import pandas as pd
import json

# -------------------------------------------------------------------
# 1. 금액/날짜 헬퍼
# -------------------------------------------------------------------

# 숫자와 콤마만 뽑기
SIGN_AMOUNT_PATTERN = re.compile(r"[\d,]+")


def parse_amount_with_sign(amount_str: str) -> int:
    """
    '-31,410원', '+100,000원', '~31,410원' 같은 문자열을
    -31410, +100000, -31410 형태의 int로 변환.

    ⚠ OCR이 '-'를 '~'로 잘못 읽는 경우가 있어서
       문자열 처음이 '~'인 경우도 음수(-)로 취급한다.
    """
    if not amount_str:
        return 0

    text = amount_str.strip()

    # 유니코드 마이너스(−)를 일반 '-'로 통일
    # 다양한 유니코드 마이너스를 일반 '-'로 통일
    text = (
        text
        .replace("−", "-")  # U+2212
        .replace("—", "-")  # U+2014
        .replace("–", "-")  # U+2013
    )

    # 다양한 틸드 계열 문자를 모두 '~'로 통일
    text = (
        text
        .replace("∼", "~")  # U+223C
        .replace("～", "~")  # U+FF5E
    )

    # 부호 판정
    sign = 1
    if text.startswith("-") or text.startswith("~"):
        # '-' 또는 '~'로 시작하면 '지출'로 보고 음수
        sign = -1
    elif text.startswith("+"):
        sign = 1

    # "+ 31,410원", "~ 31,410원" 같은 것 처리: 부호 제거 후 숫자만 추출
    text_no_sign = text.lstrip("+-~").strip()

    m = SIGN_AMOUNT_PATTERN.search(text_no_sign)
    if not m:
        return 0

    digits = m.group().replace(",", "")
    if not digits:
        return 0

    return sign * int(digits)


def extract_year_month(date_str: str) -> Optional[str]:
    """
    '2025.12.04 00:02', '2025-12-04', '2025/12/04' 같은 문자열에서
    '2025-12' 형식의 year-month를 뽑아낸다.
    """
    if not date_str:
        return None

    parts = date_str.split()
    date_part = parts[0]

    date_part = date_part.replace(".", "-").replace("/", "-")

    try:
        dt = datetime.strptime(date_part, "%Y-%m-%d")
    except ValueError:
        return None

    return f"{dt.year:04d}-{dt.month:02d}"


# -------------------------------------------------------------------
# 2. 거래 리스트 → DataFrame 변환
# -------------------------------------------------------------------

def transactions_to_dataframe(
    transactions: List[Dict],
    category_field: str = "category_rule"
) -> pd.DataFrame:
    """
    거래 리스트를 분석하기 편한 pandas DataFrame으로 변환.

    각 거래 dict 예시:
    {
      "date": "2025.12.04 00:02",
      "merchant": "(주)카카오스타일",
      "amount": "-31,410원" 혹은 "~31,410원" 혹은 "+100,000원",
      "category_rule": "쇼핑"
    }

    생성되는 컬럼:
    - amount_signed: 부호 포함 금액 (int)
    - amount_abs: 절댓값
    - is_expense: 지출 여부 (amount_signed < 0)
    - is_income: 수입/환불/충전 여부 (amount_signed > 0)
    """
    rows = []
    for tx in transactions:
        amount_str = tx.get("amount", "")
        amount_signed = parse_amount_with_sign(amount_str)
        amount_abs = abs(amount_signed)

        date_str = tx.get("date", "")
        ym = extract_year_month(date_str)

        rows.append({
            "date": date_str,
            "year_month": ym,
            "merchant": tx.get("merchant", ""),
            "amount_signed": amount_signed,
            "amount_abs": amount_abs,
            "amount_str": amount_str,
            "category": tx.get(category_field, "기타"),
            "is_expense": amount_signed < 0,
            "is_income": amount_signed > 0,
        })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["year_month"])
    return df


# -------------------------------------------------------------------
# 3. 월간 요약 (지출/수입 분리)
# -------------------------------------------------------------------

def monthly_summary(
    df: pd.DataFrame,
    month: Optional[str] = None
) -> Dict:
    """
    특정 month(예: '2025-12')에 대한
    - 총 지출 (total_spent, 절댓값 합)
    - 총 수입/환불/충전 (total_income, 절댓값 합)
    - 지출 카테고리별 합계/비율(by_category_expense)
    를 계산해 JSON으로 반환.
    
    month가 None이면 모든 월의 거래를 합쳐서 계산합니다.
    """
    if df.empty:
        return {
            "month": month or "전체",
            "total_spent": 0,
            "total_income": 0,
            "by_category_expense": {},
        }

    # month 미지정 시, 모든 월의 거래를 합쳐서 계산
    if month is None:
        df_m = df.copy()
        month = "전체"
        # 여러 월이 있으면 범위 표시
        unique_months = sorted(df["year_month"].unique())
        if len(unique_months) > 1:
            month = f"{unique_months[0]} ~ {unique_months[-1]}"
        elif len(unique_months) == 1:
            month = unique_months[0]
    else:
        df_m = df[df["year_month"] == month].copy()

    # 지출/수입 분리
    df_exp = df_m[df_m["is_expense"]]
    df_inc = df_m[df_m["is_income"]]

    total_spent = int(df_exp["amount_abs"].sum())
    total_income = int(df_inc["amount_abs"].sum())

    # 지출 카테고리별 합계
    cat_sum = df_exp.groupby("category")["amount_abs"].sum().to_dict()

    by_category_expense = {}
    for cat, s in cat_sum.items():
        ratio = (float(s) / total_spent * 100.0) if total_spent > 0 else 0.0
        by_category_expense[cat] = {
            "amount": int(s),
            "ratio": ratio,
        }

    summary = {
        "month": month,
        "total_spent": total_spent,
        "total_income": total_income,
        "by_category_expense": by_category_expense,
    }
    return summary


# -------------------------------------------------------------------
# 4. 전월 대비 증감 (선택)
# -------------------------------------------------------------------

def month_over_month_change(
    df: pd.DataFrame,
    month: str
) -> Optional[Dict]:
    """
    month(예: '2025-12')와 그 이전 달의 총 지출을 비교.
    이전 달 데이터가 없으면 None.
    """
    months = sorted(df["year_month"].unique())
    if month not in months:
        return None

    idx = months.index(month)
    if idx == 0:
        return None  # 이전 달 없음

    prev_month = months[idx - 1]

    cur_summary = monthly_summary(df, month)
    prev_summary = monthly_summary(df, prev_month)

    result = {
        "current_month": cur_summary["month"],
        "previous_month": prev_month,
        "current_total_spent": cur_summary["total_spent"],
        "previous_total_spent": prev_summary["total_spent"],
    }

    prev_total = prev_summary["total_spent"]
    if prev_total > 0:
        diff = cur_summary["total_spent"] - prev_total
        rate = diff / prev_total * 100.0
    else:
        diff = cur_summary["total_spent"]
        rate = None

    result["diff_total_spent"] = diff
    result["diff_rate_spent"] = rate
    return result


# -------------------------------------------------------------------
# 5. LLM 요약용 payload & prompt & stub
# -------------------------------------------------------------------

def build_summary_payload(summary: Dict) -> Dict:
    """
    LLM에 넘길 최소 JSON.
    수입은 참고용으로 포함하고, 본문 요약은 지출 중심으로 하게 설계.
    """
    return {
        "month": summary["month"],
        "total_spent": summary["total_spent"],
        "total_income": summary["total_income"],
        "by_category_expense": {
            cat: data["amount"]
            for cat, data in summary["by_category_expense"].items()
        },
    }


def build_summary_prompt(payload: Dict) -> str:
    """
    LLM 프롬프트 문자열 생성.
    """
    payload_str = json.dumps(payload, ensure_ascii=False)

    prompt = f"""
아래 JSON은 어떤 사용자의 카드 소비 내역을 요약한 것이다.

{payload_str}

- 어떤 카테고리에 지출이 가장 많은지
- 줄이면 좋을 것 같은 지출 항목 1~2개
- 수입/환불(총 income)이 있다면, 지출 대비 어느 정도인지

를 간단한 한국어 문장으로 3줄 이내로 설명해줘.
숫자에 기반해서 담백하게 말해줘.
"""
    return prompt.strip()


def summarize_with_llm_stub(summary: Dict) -> str:
    """
    나중에 OpenAI 등 LLM API 붙일 자리.
    지금은 summary를 기반으로 간단한 규칙형 텍스트만 생성.
    """
    payload = build_summary_payload(summary)
    month = payload["month"]
    total_spent = payload["total_spent"]
    total_income = payload["total_income"]
    by_cat = payload["by_category_expense"]

    if total_spent == 0 and total_income == 0:
        return f"{month}에는 카드 사용 및 입금 내역이 거의 없습니다."

    lines = []
    if total_spent > 0:
        lines.append(f"{month} 총 카드 지출 금액은 {total_spent:,}원입니다.")
    else:
        lines.append(f"{month}에는 지출 내역이 없습니다.")

    if by_cat:
        top_cat = max(by_cat.items(), key=lambda x: x[1])[0]
        lines.append(f"가장 많이 지출한 카테고리는 '{top_cat}'입니다.")
    else:
        lines.append("지출 카테고리가 뚜렷하지 않아, 개별 거래를 한 번 더 확인해보는 것이 좋습니다.")

    if total_income > 0:
        lines.append(f"같은 기간 동안 수입·환불 등으로 {total_income:,}원이 들어왔습니다.")

    return "\n".join(lines)
