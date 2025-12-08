from __future__ import annotations
from typing import Dict
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)


def build_summary_payload(summary: Dict) -> Dict:
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
    payload_str = json.dumps(payload, ensure_ascii=False)

    prompt = f"""
아래 JSON은 어떤 사용자의 카드 소비 내역을 요약한 것이다.

{payload_str}

다음 내용을 각각 "- " 형식으로 나열하여 상세하고 친절한 한국어 요약을 작성해주세요:

- **총 지출 금액**을 강조하여 언급
- **가장 많이 지출한 카테고리**와 그 금액을 구체적으로 설명
- **줄이면 좋을 지출 항목** 1~2개를 구체적으로 제안
- **수입/환불**이 있다면, 지출 대비 비율과 의미를 설명
- **전반적인 소비 패턴**에 대한 간단한 평가

각 항목은 반드시 "- "로 시작하고, 중요한 숫자와 카테고리는 **강조**하기 위해 **로 감싸서 표시해주세요.
예: 
- **총 지출은 150,000원**입니다.
- **식비**에 **80,000원**을 사용하여 가장 많은 지출을 했습니다.
"""
    return prompt.strip()


def summarize_monthly_report_with_llm(summary: Dict) -> str:
    """
    reporter.monthly_summary 결과를 받아
    OpenAI ChatCompletion 기반으로 자연어 요약 생성.
    """
    payload = build_summary_payload(summary)
    prompt = build_summary_prompt(payload)

    # 🔁 API 키 없으면 그냥 규칙 기반 fallback
    if not OPENAI_API_KEY:
        if not payload["by_category_expense"]:
            return f"{payload['month']}에는 지출 내역이 거의 없습니다."

        top_cat = max(
            payload["by_category_expense"].items(),
            key=lambda x: x[1]
        )[0]

        lines = []
        lines.append(f"- {payload['month']} 총 카드 지출 금액은 **{payload['total_spent']:,}원**입니다.")
        lines.append(f"- 가장 많이 지출한 카테고리는 **{top_cat}**입니다.")
        if payload["total_income"] > 0:
            lines.append(f"- 같은 기간 수입·환불로 **{payload['total_income']:,}원**이 들어왔습니다.")
        else:
            lines.append("- 수입·환불 내역은 거의 없습니다.")
        return "\n".join(lines)

    # ✅ 여기서부터는 진짜 LLM 호출 (chat.completions)
    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # 또는 gpt-4.1
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    return response.choices[0].message.content.strip()