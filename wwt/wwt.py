__doc__ = """WWT 공정과 관련된 라우트

- 기능 목록:
  - 데이터 업로드 및 관리
  - 추론 실행 및 결과 조회
- 담당 작업자: 반세현
"""
import re
from pathlib import Path
import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from flask import Blueprint, redirect, render_template, url_for, request, jsonify, current_app

from app.services.wwt_service import WWTBlowerInferenceService
from app.services.tabular_service import TabularService


# TODO: 나중에는 app.logger 에서 접근하여 로그 남기도록 수정정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NOTE: 내년도 과제 목록을 알 수 없어, 우선 공정을 기준으로 routes를 만들어둠
wwt_bp = Blueprint("wwt", __name__)

@wwt_bp.route("/")
def render_dashboard():
    """WWT 대시보드 화면"""
    return render_template("wwt.html")

@wwt_bp.route("/upload", methods=["POST"])
def upload_data():
    """WWT 데이터 업로드"""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "파일이 선택되지 않았습니다."}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "파일이 선택되지 않았습니다."}), 400
        
        # TabularService를 통해 데이터 업로드
        tabular_data = TabularService.upload_csv(
            file=file,
            category="WWT",
            description=request.form.get('description', 'WWT 데이터'),
            uploaded_by=request.form.get('uploaded_by', 'anonymous')
        )
        
        return jsonify({
            "success": True,
            "data_id": tabular_data.id,
            "filename": tabular_data.original_filename,
            "row_count": tabular_data.row_count,
            "uploaded_at": tabular_data.uploaded_at.isoformat()
        })
        
    except Exception as e:
        logger.error(f"데이터 업로드 오류: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@wwt_bp.route("/data", methods=["GET"])
def list_data():
    """WWT 데이터 목록 조회"""
    try:
        wwt_data_list = TabularService.list_by_category("WWT")
        data_list = []
        
        for data in wwt_data_list:
            data_list.append({
                "id": data.id,
                "filename": data.original_filename,
                "row_count": data.row_count,
                "column_count": data.column_count,
                "uploaded_at": data.uploaded_at.isoformat(),
                "description": data.description
            })
        
        return jsonify({
            "success": True,
            "data": data_list,
            "count": len(data_list)
        })
        
    except Exception as e:
        logger.error(f"데이터 목록 조회 오류: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@wwt_bp.route("/predict_do", methods=["POST"])
def inference_do():
    """송풍량 예측"""
    try:
        # 요청 데이터 파싱
        data = request.get_json() or {}
        data_ids = data.get('data_ids')
        target_time = data.get('target_time')
        
        # WWT 추론 서비스 초기화 및 실행
        inference_service = WWTBlowerInferenceService()
        result = inference_service.run(data_ids=data_ids, target_time=target_time)
        
        if result["success"]:
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"추론 실행 오류: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }), 500

@wwt_bp.route("/data/<data_id>", methods=["GET"])
def get_data_info(data_id):
    """특정 데이터 정보 조회"""
    try:
        tabular_data = TabularService.get_by_id(data_id)
        if tabular_data is None:
            return jsonify({"success": False, "error": "데이터를 찾을 수 없습니다."}), 404
        
        # 데이터 미리보기
        preview = TabularService.get_data_preview(tabular_data, nrows=5)
        
        return jsonify({
            "success": True,
            "data": {
                "id": tabular_data.id,
                "filename": tabular_data.original_filename,
                "row_count": tabular_data.row_count,
                "column_count": tabular_data.column_count,
                "columns": tabular_data.columns,
                "uploaded_at": tabular_data.uploaded_at.isoformat(),
                "description": tabular_data.description,
                "preview": preview
            }
        })
        
    except Exception as e:
        logger.error(f"데이터 정보 조회 오류: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
