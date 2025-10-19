# app/collector/models.py
"""
Collector 데이터 모델 통합
- FieldMetadata, IndicatorDataWithMeta, CompanyIdentity 등
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
from collections import defaultdict


# ======== 필드 중요도 ========
class FieldImportance(Enum):
    CRITICAL = "critical"
    EV_CRITICAL = "ev_critical"
    DCF_CRITICAL = "dcf_critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


FIELD_IMPORTANCE_MAP = {
    'share_price': FieldImportance.CRITICAL,
    'shares_outstanding': FieldImportance.CRITICAL,
    'cash_and_equivalents': FieldImportance.EV_CRITICAL,
    'total_debt': FieldImportance.EV_CRITICAL,
    'operating_cash_flow': FieldImportance.DCF_CRITICAL,
    'capex': FieldImportance.DCF_CRITICAL,
    'risk_free_rate': FieldImportance.DCF_CRITICAL,
    'wacc': FieldImportance.DCF_CRITICAL,
    'ltm_revenue': FieldImportance.HIGH,
    'ltm_ebitda': FieldImportance.HIGH,
    'ltm_ebit': FieldImportance.HIGH,
    'ltm_eps': FieldImportance.HIGH,
    'interest_expense': FieldImportance.HIGH,
    'effective_tax_rate': FieldImportance.HIGH,
    'revenue_growth_rate': FieldImportance.HIGH,
    'current_ratio': FieldImportance.HIGH,
    'roic': FieldImportance.HIGH,
    'ntm_eps': FieldImportance.MEDIUM,
    'beta': FieldImportance.MEDIUM,
    'dividend_yield': FieldImportance.MEDIUM,
    # ... 나머지 필드들
}


YAHOO_FIELD_MAPPING = {
    'fiftyTwoWeekHigh': 'fifty_two_week_high',
    'trailingPE': 'trailing_pe',
    'forwardPE': 'forward_pe',
    'pegRatio': 'peg_ratio',
    'priceToBook': 'price_to_book',
    'targetMeanPrice': 'target_mean_price',
    'beta': 'beta',
    # ... 전체 40+ 매핑
}


# ======== 메타데이터 ========
@dataclass
class FieldMetadata:
    """필드 메타데이터 (출처, 신뢰도, XBRL 정보)"""
    value: Any
    source: str
    source_link: Optional[str] = None
    
    # XBRL provenance
    xbrl_tag: Optional[str] = None
    xbrl_unit: Optional[str] = None
    frame: Optional[str] = None
    accession: Optional[str] = None
    filing_date: Optional[str] = None
    filing_url: Optional[str] = None
    
    # Standard metadata
    statement: Optional[str] = None
    period: Optional[str] = None
    date_retrieved: Optional[str] = None
    reliability: float = 0.0
    verified: bool = False
    cross_check_sources: List[str] = field(default_factory=list)
    unit: Optional[str] = None
    currency: Optional[str] = None
    importance: FieldImportance = FieldImportance.MEDIUM
    
    # Traceability
    raw_json_path: Optional[str] = None
    snapshot_sha256: Optional[str] = None
    parser_version: Optional[str] = None
    conversion_policy: Optional[str] = None


@dataclass
class AttemptLog:
    """수집 시도 로그"""
    source: str
    method: str
    success: bool
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class CollectionAttemptSummary:
    """수집 시도 요약"""
    ticker: str
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    field_attempts: Dict[str, List[AttemptLog]] = field(default_factory=dict)
    source_stats: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {'attempted': 0, 'succeeded': 0})
    )
    valuation_readiness: Optional['ValuationReadiness'] = None


@dataclass
class ValuationReadiness:
    """밸류에이션 준비 상태"""
    ev_ready: bool = False
    dcf_ready: bool = False
    multiples_ready: bool = False
    ev_missing: List[str] = field(default_factory=list)
    dcf_missing: List[str] = field(default_factory=list)
    multiples_missing: List[str] = field(default_factory=list)


# ======== 회사 정보 ========
@dataclass
class CompanyIdentity:
    """회사 기본 정보"""
    ticker: str
    company_name: Optional[str] = None
    cik: Optional[str] = None
    exchange: Optional[str] = None
    country: Optional[str] = None
    sic_code: Optional[str] = None
    sic_description: Optional[str] = None
    website: Optional[str] = None
    accounting_standard: Optional[str] = None


# ======== 재무 지표 ========
@dataclass
class IndicatorDataWithMeta:
    """재무 지표 (메타데이터 포함)"""
    # Market
    share_price: Optional[FieldMetadata] = None
    shares_outstanding: Optional[FieldMetadata] = None
    market_cap: Optional[FieldMetadata] = None
    
    # Income Statement
    ltm_revenue: Optional[FieldMetadata] = None
    ltm_ebitda: Optional[FieldMetadata] = None
    ltm_ebit: Optional[FieldMetadata] = None
    ltm_eps: Optional[FieldMetadata] = None
    ntm_eps: Optional[FieldMetadata] = None
    
    # Cost Structure
    cost_of_revenue: Optional[FieldMetadata] = None
    gross_profit: Optional[FieldMetadata] = None
    research_development: Optional[FieldMetadata] = None
    selling_general_admin: Optional[FieldMetadata] = None
    
    # Balance Sheet
    total_assets: Optional[FieldMetadata] = None
    total_equity: Optional[FieldMetadata] = None
    total_debt: Optional[FieldMetadata] = None
    current_debt: Optional[FieldMetadata] = None
    long_term_debt: Optional[FieldMetadata] = None
    cash_and_equivalents: Optional[FieldMetadata] = None
    
    # Working Capital
    current_assets: Optional[FieldMetadata] = None
    current_liabilities: Optional[FieldMetadata] = None
    inventory: Optional[FieldMetadata] = None
    accounts_receivable: Optional[FieldMetadata] = None
    
    # Cash Flow
    operating_cash_flow: Optional[FieldMetadata] = None
    capex: Optional[FieldMetadata] = None
    free_cash_flow: Optional[FieldMetadata] = None
    depreciation_amortization: Optional[FieldMetadata] = None
    
    # Ratios
    interest_expense: Optional[FieldMetadata] = None
    tax_expense: Optional[FieldMetadata] = None
    effective_tax_rate: Optional[FieldMetadata] = None
    debt_to_equity: Optional[FieldMetadata] = None
    current_ratio: Optional[FieldMetadata] = None
    quick_ratio: Optional[FieldMetadata] = None
    interest_coverage: Optional[FieldMetadata] = None
    roic: Optional[FieldMetadata] = None
    roce: Optional[FieldMetadata] = None
    
    # Growth
    revenue_growth_rate: Optional[FieldMetadata] = None
    ebitda_growth_rate: Optional[FieldMetadata] = None
    eps_growth_rate: Optional[FieldMetadata] = None
    
    # Yahoo Extended (40+ fields)
    fifty_two_week_high: Optional[FieldMetadata] = None
    fifty_two_week_low: Optional[FieldMetadata] = None
    trailing_pe: Optional[FieldMetadata] = None
    forward_pe: Optional[FieldMetadata] = None
    peg_ratio: Optional[FieldMetadata] = None
    price_to_book: Optional[FieldMetadata] = None
    target_mean_price: Optional[FieldMetadata] = None
    beta: Optional[FieldMetadata] = None
    dividend_yield: Optional[FieldMetadata] = None
    # ... (나머지 Yahoo 필드)
    
    # DCF Parameters
    risk_free_rate: Optional[FieldMetadata] = None
    market_risk_premium: Optional[FieldMetadata] = None
    cost_of_equity: Optional[FieldMetadata] = None
    wacc: Optional[FieldMetadata] = None
    terminal_growth_rate: Optional[FieldMetadata] = None
    
    # Meta
    sector: Optional[FieldMetadata] = None
    industry: Optional[FieldMetadata] = None
    currency: Optional[str] = None
    derived_calculations: Dict[str, 'DerivedCalculation'] = field(default_factory=dict)


@dataclass
class DerivedCalculation:
    """파생 계산 결과"""
    result: float
    formula: str
    components: Dict[str, Dict[str, Any]]
    calculation_method: Optional[str] = None


# ======== 분석 결과 ========
@dataclass
class TechnicalIndicators:
    """기술적 지표"""
    rsi_14: Optional[float] = None
    atr_14: Optional[float] = None
    price_vs_ma50_pct: Optional[float] = None
    price_vs_ma200_pct: Optional[float] = None
    macd_signal: Optional[str] = None
    bollinger_position: Optional[float] = None


@dataclass
class AnalystActivity:
    """애널리스트 활동 (14일)"""
    upgrades_count: int = 0
    downgrades_count: int = 0
    new_coverage_count: int = 0
    target_raises_count: int = 0
    target_lowers_count: int = 0
    avg_target_change_pct: Optional[float] = None
    recent_actions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ValuationResults:
    """밸류에이션 결과"""
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    ev_calculation: Optional[DerivedCalculation] = None
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    ev_sales: Optional[float] = None  # ✅ 추가
    dcf_equity_value: Optional[float] = None
    dcf_value_per_share: Optional[float] = None
    dcf_upside_pct: Optional[float] = None
    dcf_sensitivity: Optional[Dict[str, Any]] = None
    calculation_status: Dict[str, str] = field(default_factory=dict)


@dataclass
class DataQualityMetrics:
    """데이터 품질 지표"""
    coverage_pct: float = 0.0
    verified_pct: float = 0.0
    reliability_score: float = 0.0
    total_measurable_fields: int = 0
    filled_fields: int = 0
    verified_fields: int = 0
    critical_coverage_pct: float = 0.0
    high_coverage_pct: float = 0.0
    field_verification_details: Dict[str, Any] = field(default_factory=dict)