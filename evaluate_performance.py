"""
OCR 성능 평가 스크립트
전처리 전/후 성능을 비교하고 정량적 지표를 측정합니다.
"""

import os
import json
from typing import List, Dict, Tuple
from pathlib import Path
from difflib import SequenceMatcher
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from app.ocr_service import run_ocr
from app.parser import parse_statement_items


def load_ground_truth(gt_path: str) -> List[Dict]:
    """Ground truth 데이터 로드"""
    with open(gt_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_image(image_path: str) -> bytes:
    """이미지 파일을 바이트로 로드"""
    with open(image_path, 'rb') as f:
        return f.read()


def normalize_text(text: str) -> str:
    """텍스트 정규화 (공백 제거, 대소문자 통일 등)"""
    if not text:
        return ""
    # 공백 제거 및 소문자 변환
    normalized = text.replace(" ", "").replace("\n", "").strip()
    return normalized


def calculate_text_similarity(text1: str, text2: str) -> float:
    """두 텍스트 간의 유사도 계산 (0.0 ~ 1.0)"""
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    if not norm1 and not norm2:
        return 1.0
    if not norm1 or not norm2:
        return 0.0
    return SequenceMatcher(None, norm1, norm2).ratio()


def extract_texts_from_ocr(ocr_results: List[Dict]) -> List[str]:
    """OCR 결과에서 텍스트만 추출"""
    return [item.get('text', '').strip() for item in ocr_results if item.get('text', '').strip()]


def extract_texts_from_gt(gt_data: List[Dict]) -> List[str]:
    """Ground truth에서 텍스트만 추출"""
    return [item.get('text', '').strip() for item in gt_data if item.get('text', '').strip()]


def calculate_ocr_metrics(
    ocr_results: List[Dict],
    gt_data: List[Dict]
) -> Dict[str, float]:
    """
    OCR 성능 지표 계산
    
    Returns:
        {
            "text_accuracy": float,  # 텍스트 정확도 (전체 매칭)
            "character_accuracy": float,  # 문자 단위 정확도
            "precision": float,  # 정밀도
            "recall": float,  # 재현율
            "f1_score": float,  # F1 점수
            "avg_similarity": float,  # 평균 유사도
            "total_texts": int,  # 전체 텍스트 수
            "matched_texts": int,  # 매칭된 텍스트 수
        }
    """
    ocr_texts = extract_texts_from_ocr(ocr_results)
    gt_texts = extract_texts_from_gt(gt_data)
    
    # 텍스트 정확도 (정확히 일치하는 텍스트 비율)
    ocr_normalized = [normalize_text(t) for t in ocr_texts]
    gt_normalized = [normalize_text(t) for t in gt_texts]
    
    # 최대 길이로 맞춤
    max_len = max(len(ocr_normalized), len(gt_normalized))
    ocr_padded = ocr_normalized + [''] * (max_len - len(ocr_normalized))
    gt_padded = gt_normalized + [''] * (max_len - len(gt_normalized))
    
    # 정확히 일치하는 텍스트 수
    exact_matches = sum(1 for o, g in zip(ocr_padded, gt_padded) if o == g)
    text_accuracy = exact_matches / max_len if max_len > 0 else 0.0
    
    # 문자 단위 정확도
    ocr_combined = ''.join(ocr_normalized)
    gt_combined = ''.join(gt_normalized)
    char_accuracy = calculate_text_similarity(ocr_combined, gt_combined)
    
    # 유사도 기반 매칭
    similarities = []
    matched_indices = set()  # GT에서 매칭된 인덱스
    ocr_matched_indices = set()  # OCR에서 매칭된 인덱스
    
    for i, ocr_text in enumerate(ocr_normalized):
        if not ocr_text:
            continue
        best_sim = 0.0
        best_idx = -1
        for j, gt_text in enumerate(gt_normalized):
            if j in matched_indices:
                continue
            sim = calculate_text_similarity(ocr_text, gt_text)
            if sim > best_sim:
                best_sim = sim
                best_idx = j
        if best_sim > 0.7:  # 70% 이상 유사도면 매칭으로 간주
            matched_indices.add(best_idx)
            ocr_matched_indices.add(i)
            similarities.append(best_sim)
    
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    matched_texts = len(similarities)
    
    # sklearn을 사용한 Precision, Recall, F1 계산
    # GT 텍스트 기준으로 각 텍스트가 OCR에서 올바르게 인식되었는지 이진 분류
    y_true = [1] * len(gt_normalized)  # GT는 모두 존재하므로 1
    y_pred = [1 if j in matched_indices else 0 for j in range(len(gt_normalized))]  # OCR에서 매칭 여부
    
    # OCR에만 있고 GT에는 없는 텍스트는 False Positive로 처리
    for i, ocr_text in enumerate(ocr_normalized):
        if i not in ocr_matched_indices and ocr_text:
            y_true.append(0)  # 실제로는 존재하지 않음 (False Positive)
            y_pred.append(1)  # OCR은 인식함
    
    # sklearn 메트릭 사용
    if len(y_true) > 0 and len(y_pred) > 0 and len(y_true) == len(y_pred):
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
    else:
        # 수동 계산 (fallback)
        tp = matched_texts
        fp = len(ocr_normalized) - matched_texts
        fn = len(gt_normalized) - matched_texts
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "text_accuracy": text_accuracy,
        "character_accuracy": char_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "avg_similarity": avg_similarity,
        "total_texts": len(gt_normalized),
        "matched_texts": matched_texts,
    }


def evaluate_preprocessing_methods(
    image_path: str,
    gt_path: str,
    methods: List[str] = ["none", "basic", "adaptive", "enhanced"]
) -> Dict[str, Dict[str, float]]:
    """
    여러 전처리 방법에 대한 성능 평가
    
    Returns:
        {
            "none": {metrics...},
            "basic": {metrics...},
            "adaptive": {metrics...},
            "enhanced": {metrics...},
        }
    """
    image_bytes = load_image(image_path)
    gt_data = load_ground_truth(gt_path)
    
    results = {}
    
    for method in methods:
        print(f"\n[평가 중] 전처리 방법: {method}")
        try:
            # OCR 수행
            ocr_results = run_ocr(
                image_bytes,
                preprocess=(method != "none"),
                preprocess_method=method if method != "none" else "enhanced"
            )
            
            # 성능 지표 계산
            metrics = calculate_ocr_metrics(ocr_results, gt_data)
            results[method] = metrics
            
            print(f"  - 텍스트 정확도: {metrics['text_accuracy']:.3f}")
            print(f"  - 문자 정확도: {metrics['character_accuracy']:.3f}")
            print(f"  - F1 점수: {metrics['f1_score']:.3f}")
            print(f"  - 평균 유사도: {metrics['avg_similarity']:.3f}")
            
        except Exception as e:
            print(f"  [오류] {method} 전처리 실패: {e}")
            results[method] = {
                "text_accuracy": 0.0,
                "character_accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "avg_similarity": 0.0,
                "total_texts": len(gt_data),
                "matched_texts": 0,
            }
    
    return results


def main():
    """메인 평가 함수"""
    print("=" * 60)
    print("OCR 성능 평가 시작")
    print("=" * 60)
    
    # 데이터 경로 설정
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    gt_dir = data_dir / "ground_truth"
    
    # 평가할 이미지와 Ground Truth 찾기
    test_cases = []
    for img_file in raw_dir.glob("*.png"):
        gt_file = gt_dir / f"{img_file.stem}_gt.json"
        if gt_file.exists():
            test_cases.append((str(img_file), str(gt_file)))
    
    for img_file in raw_dir.glob("*.jpg"):
        gt_file = gt_dir / f"{img_file.stem}_gt.json"
        if gt_file.exists():
            test_cases.append((str(img_file), str(gt_file)))
    
    if not test_cases:
        print("\n[경고] 평가할 이미지 파일을 찾을 수 없습니다.")
        print("\n이 스크립트를 실행하려면 다음이 필요합니다:")
        print("  1. data/raw/ 폴더에 테스트 이미지 파일 (.png 또는 .jpg)")
        print("  2. data/ground_truth/ 폴더에 해당하는 Ground Truth JSON 파일")
        print("     (예: sample_statement_02.png → sample_statement_02_gt.json)")
        print("\n참고: 개인정보 보호를 위해 실제 이미지는 GitHub에 업로드되지 않습니다.")
        print("      로컬에서 평가를 수행하려면 data/raw/ 폴더에 이미지를 추가하세요.")
        print("\n[정보] 기존 평가 결과는 PERFORMANCE_EVALUATION.md 파일을 참고하세요.")
        return
    
    print(f"\n총 {len(test_cases)}개의 테스트 케이스를 평가합니다.\n")
    
    all_results = {}
    
    for img_path, gt_path in test_cases:
        print(f"\n{'='*60}")
        print(f"테스트 케이스: {Path(img_path).name}")
        print(f"{'='*60}")
        
        results = evaluate_preprocessing_methods(img_path, gt_path)
        all_results[Path(img_path).stem] = results
    
    # 전체 평균 계산
    print(f"\n{'='*60}")
    print("전체 평균 성능")
    print(f"{'='*60}")
    
    methods = ["none", "basic", "adaptive", "enhanced"]
    avg_results = {method: {
        "text_accuracy": [],
        "character_accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "avg_similarity": [],
    } for method in methods}
    
    for case_results in all_results.values():
        for method in methods:
            if method in case_results:
                for metric in avg_results[method].keys():
                    avg_results[method][metric].append(case_results[method].get(metric, 0.0))
    
    # 평균 계산 및 출력
    final_avg = {}
    for method in methods:
        final_avg[method] = {
            metric: sum(values) / len(values) if values else 0.0
            for metric, values in avg_results[method].items()
        }
        print(f"\n[{method.upper()}]")
        print(f"  텍스트 정확도: {final_avg[method]['text_accuracy']:.3f}")
        print(f"  문자 정확도: {final_avg[method]['character_accuracy']:.3f}")
        print(f"  Precision: {final_avg[method]['precision']:.3f}")
        print(f"  Recall: {final_avg[method]['recall']:.3f}")
        print(f"  F1 Score: {final_avg[method]['f1_score']:.3f}")
        print(f"  평균 유사도: {final_avg[method]['avg_similarity']:.3f}")
    
    # 결과를 JSON으로 저장
    output_path = "performance_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "individual_results": all_results,
            "average_results": final_avg
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n결과가 {output_path}에 저장되었습니다.")
    
    return final_avg, all_results


if __name__ == "__main__":
    main()

