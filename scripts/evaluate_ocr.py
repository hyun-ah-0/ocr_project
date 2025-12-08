# scripts/evaluate_ocr.py

"""
OCR 성능 평가 스크립트
Accuracy, Precision, Recall, F1-score 측정
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.ocr_service import run_vlm_ocr
from app.image_preprocessor import preprocess_image


def load_ground_truth(gt_path: Path) -> List[Dict]:
    """
    Ground truth 데이터 로드
    """
    with open(gt_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    """
    텍스트 정규화 (비교를 위해)
    - 공백 제거
    - 소문자 변환 (영문의 경우)
    """
    return text.strip().replace(" ", "").replace("\n", "")


def extract_texts(ocr_results: List[Dict]) -> List[str]:
    """
    OCR 결과에서 텍스트만 추출
    """
    return [normalize_text(item['text']) for item in ocr_results if item.get('text')]


def calculate_metrics(
    predicted: List[str],
    ground_truth: List[str]
) -> Dict[str, float]:
    """
    OCR 성능 지표 계산
    
    Args:
        predicted: 예측된 텍스트 리스트
        ground_truth: 실제 텍스트 리스트
    
    Returns:
        성능 지표 딕셔너리
    """
    # 집합으로 변환하여 중복 제거
    pred_set = set(predicted)
    gt_set = set(ground_truth)
    
    # 교집합 (정확히 일치하는 텍스트)
    intersection = pred_set & gt_set
    
    # Precision: 예측한 것 중 맞은 것
    precision = len(intersection) / len(pred_set) if len(pred_set) > 0 else 0.0
    
    # Recall: 실제 것 중 찾은 것
    recall = len(intersection) / len(gt_set) if len(gt_set) > 0 else 0.0
    
    # F1-score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Accuracy: 전체 텍스트 중 정확히 일치한 비율
    all_texts = pred_set | gt_set
    accuracy = len(intersection) / len(all_texts) if len(all_texts) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "predicted_count": len(pred_set),
        "ground_truth_count": len(gt_set),
        "matched_count": len(intersection)
    }


def evaluate_single_image(
    image_path: Path,
    gt_path: Path,
    preprocess_method: str = "none"
) -> Dict[str, float]:
    """
    단일 이미지에 대한 OCR 성능 평가
    
    Args:
        image_path: 이미지 파일 경로
        gt_path: Ground truth JSON 파일 경로
        preprocess_method: 전처리 방법
    
    Returns:
        성능 지표
    """
    # 이미지 로드
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    # 전처리 적용
    if preprocess_method != "none":
        image_bytes = preprocess_image(image_bytes, method=preprocess_method)
    
    # OCR 수행
    ocr_results = run_vlm_ocr(image_bytes)
    
    # Ground truth 로드
    ground_truth = load_ground_truth(gt_path)
    
    # 텍스트 추출
    predicted_texts = extract_texts(ocr_results)
    gt_texts = extract_texts(ground_truth)
    
    # 성능 계산
    metrics = calculate_metrics(predicted_texts, gt_texts)
    
    return metrics


def evaluate_all_images(
    data_dir: Path = Path("data/raw"),
    gt_dir: Path = Path("data/ground_truth"),
    preprocess_method: str = "none"
) -> Dict[str, any]:
    """
    모든 테스트 이미지에 대한 성능 평가
    
    Returns:
        전체 성능 지표 및 이미지별 상세 결과
    """
    results = {
        "preprocess_method": preprocess_method,
        "image_results": [],
        "overall_metrics": {}
    }
    
    # 모든 이미지 파일 찾기
    image_files = list(data_dir.glob("*.png")) + list(data_dir.glob("*.jpg"))
    
    all_predicted = []
    all_ground_truth = []
    
    for img_path in image_files:
        gt_path = gt_dir / f"{img_path.stem}_gt.json"
        
        if not gt_path.exists():
            print(f"[WARN] Ground truth 없음: {gt_path}")
            continue
        
        print(f"평가 중: {img_path.name}")
        
        try:
            metrics = evaluate_single_image(img_path, gt_path, preprocess_method)
            results["image_results"].append({
                "image": img_path.name,
                "metrics": metrics
            })
            
            # 전체 집계를 위한 데이터 수집
            # (실제로는 각 이미지별로 계산하지만, 전체 평균도 계산)
            
        except Exception as e:
            print(f"[ERROR] {img_path.name} 평가 실패: {e}")
            continue
    
    # 전체 평균 계산
    if results["image_results"]:
        avg_accuracy = sum(r["metrics"]["accuracy"] for r in results["image_results"]) / len(results["image_results"])
        avg_precision = sum(r["metrics"]["precision"] for r in results["image_results"]) / len(results["image_results"])
        avg_recall = sum(r["metrics"]["recall"] for r in results["image_results"]) / len(results["image_results"])
        avg_f1 = sum(r["metrics"]["f1_score"] for r in results["image_results"]) / len(results["image_results"])
        
        results["overall_metrics"] = {
            "accuracy": avg_accuracy,
            "precision": avg_precision,
            "recall": avg_recall,
            "f1_score": avg_f1,
            "num_images": len(results["image_results"])
        }
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OCR 성능 평가")
    parser.add_argument("--method", type=str, default="none",
                       choices=["none", "basic", "adaptive", "enhanced"],
                       help="전처리 방법")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                       help="결과 저장 파일")
    
    args = parser.parse_args()
    
    data_dir = project_root / "data" / "raw"
    gt_dir = project_root / "data" / "ground_truth"
    
    # Ground truth 디렉토리 생성
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"전처리 방법: {args.method}")
    print("=" * 50)
    
    results = evaluate_all_images(data_dir, gt_dir, args.method)
    
    # 결과 저장
    output_path = project_root / args.output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 결과 출력
    if results["overall_metrics"]:
        print("\n전체 성능 지표:")
        print(f"  Accuracy:  {results['overall_metrics']['accuracy']:.4f}")
        print(f"  Precision: {results['overall_metrics']['precision']:.4f}")
        print(f"  Recall:    {results['overall_metrics']['recall']:.4f}")
        print(f"  F1-score:  {results['overall_metrics']['f1_score']:.4f}")
        print(f"\n결과 저장: {output_path}")

