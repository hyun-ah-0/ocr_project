# scripts/create_ground_truth.py

"""
Ground truth 데이터 생성 도우미 스크립트
이미지의 정확한 OCR 결과를 수동으로 입력하여 Ground truth 생성
"""

import json
from pathlib import Path

def create_ground_truth_template(image_name: str, texts: list) -> dict:
    """
    Ground truth 템플릿 생성
    """
    return [
        {"text": text} for text in texts
    ]

if __name__ == "__main__":
    # 예시: sample_statement_02.png의 Ground truth
    # 실제로는 이미지를 보고 수동으로 입력해야 함
    
    gt_data = [
        "3:33",
        "6",
        "생활통장",
        "관리",
        "3개월 전체 최신순",
        "12.04",
        "00:02",
        "(주)카카오스타일",
        "-31,410원",
        "12.02",
        "13:04",
        "오가다 제주공항점",
        "-6,500원",
        "12.01",
        "20:50",
        "주식회사 컬리페이",
        "-54,990원",
        "12.01",
        "14:48",
        "카뱅오픈박정심",
        "-3,000원",
        "12.01",
        "02:38",
        "이자",
        "+9원",
        "11.29",
        "16:44",
        "우무 제주국제공항 도",
        "-14,600원"
    ]
    
    project_root = Path(__file__).parent.parent
    gt_dir = project_root / "data" / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    gt_file = gt_dir / "sample_statement_02_gt.json"
    
    gt_json = create_ground_truth_template("sample_statement_02", gt_data)
    
    with open(gt_file, 'w', encoding='utf-8') as f:
        json.dump(gt_json, f, ensure_ascii=False, indent=2)
    
    print(f"Ground truth 생성 완료: {gt_file}")

