# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import os

from app.parser import parse_statement_items
from app.categorizer import categorize_transactions_with_llm
from app.reporter import transactions_to_dataframe, monthly_summary
from app.llm_client import summarize_monthly_report_with_llm
from app.multimodal import classify_transaction_multimodal
from app.ocr_service import run_ocr
import pandas as pd


app = FastAPI()

# 정적 파일 서빙 (HTML 파일)
@app.get("/")
async def read_root():
    """루트 경로에서 index.html 반환"""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "HTML 파일을 찾을 수 없습니다. index.html이 프로젝트 루트에 있는지 확인하세요."}

# 로컬 개발용 CORS 허용 (React dev server: http://localhost:5173 같은 것)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 단계에서는 *로 두고, 배포할 때 도메인 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeResponse(BaseModel):
    transactions: List[Dict[str, Any]]
    summary: Dict[str, Any]
    llm_summary: str


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_statement(file: UploadFile = File(...)):
    """
    1) 이미지 업로드
    2) OCR → items
    3) parser → 거래 dict
    4) categorizer → 규칙 기반 카테고리
    5) reporter → 월간 요약
    6) llm_client → 자연어 요약
    """
    image_bytes = await file.read()

    # 1. OCR (VLM OCR 사용, 전처리 적용)
    ocr_items = run_ocr(image_bytes, preprocess=True, preprocess_method="enhanced")

    # 2. parser (여러 거래 파싱)
    tx_list = parse_statement_items(ocr_items)

    # 3. LLM 기반 카테고리 분류 (규칙 기반 + LLM 하이브리드)
    tx_list = categorize_transactions_with_llm(tx_list)

    # 4. reporter (DataFrame 변환 + 요약)
    df = transactions_to_dataframe(tx_list, category_field="category_rule")
    summary = monthly_summary(df, month=None)  # month는 df에서 자동 추론하도록 해도 됨

    # 5. LLM 텍스트 요약
    llm_text = summarize_monthly_report_with_llm(summary)

    # 6. 멀티모달 카테고리 (원하면 transaction별로 돌면서 붙이기)
    #    여기서는 첫 거래에만 예시로 붙여볼게
    #    이미지 파일 자체는 지금 이 엔드포인트에서 이미 bytes로 있고, 저장해서 path 넘기거나
    #    multimodal을 bytes 버전으로 바꿔도 됨.
    # tx_list[0]["category_mm"] = classify_transaction_multimodal("path/to/tmp.png", tx_list[0])

    return AnalyzeResponse(
        transactions=tx_list,
        summary=summary,
        llm_summary=llm_text,
    )


@app.post("/api/analyze-multiple", response_model=AnalyzeResponse)
async def analyze_statements_multiple(files: List[UploadFile] = File(...)):
    """
    여러 이미지를 한 번에 분석하여 모든 거래를 합쳐서 반환.
    
    1) 여러 이미지 업로드
    2) 각 이미지마다 OCR → parser → categorizer
    3) 모든 거래를 합쳐서 reporter → 월간 요약
    4) llm_client → 자연어 요약
    """
    all_transactions = []
    
    # 각 이미지 처리
    for file in files:
        image_bytes = await file.read()
        
        # 1. OCR (VLM OCR 사용, 전처리 적용)
        ocr_items = run_ocr(image_bytes, preprocess=True, preprocess_method="enhanced")
        
        # 2. parser (여러 거래 파싱)
        tx_list = parse_statement_items(ocr_items)
        
        # 3. LLM 기반 카테고리 분류 (규칙 기반 + LLM 하이브리드)
        tx_list = categorize_transactions_with_llm(tx_list)
        
        # 거래 리스트에 추가
        all_transactions.extend(tx_list)
    
    # 4. 모든 거래를 합쳐서 reporter (DataFrame 변환 + 요약)
    df = transactions_to_dataframe(all_transactions, category_field="category_rule")
    summary = monthly_summary(df, month=None)  # month는 df에서 자동 추론하도록 해도 됨
    
    # 5. LLM 텍스트 요약
    llm_text = summarize_monthly_report_with_llm(summary)
    
    return AnalyzeResponse(
        transactions=all_transactions,
        summary=summary,
        llm_summary=llm_text,
    )
