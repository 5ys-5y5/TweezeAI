# app/collector/__init__.py
"""
Collector Module
- 재무 데이터 수집 통합 모듈
"""
from .orchestrator import CollectorOrchestrator
from .models import (
    CompanyIdentity,
    IndicatorDataWithMeta,
    FieldMetadata,
    CollectionAttemptSummary,
    DataQualityMetrics,
    ValuationResults,
    TechnicalIndicators,
    AnalystActivity
)

__all__ = [
    'CollectorOrchestrator',
    'CompanyIdentity',
    'IndicatorDataWithMeta',
    'FieldMetadata',
    'CollectionAttemptSummary',
    'DataQualityMetrics',
    'ValuationResults',
    'TechnicalIndicators',
    'AnalystActivity',
]

__version__ = "25.0.0"