# main.py
"""
Financial Analysis Platform - Main API Server
v25.0.0
"""
# .env 파일에서 환경변수 자동 로드
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import logging

from app.collector import CollectorOrchestrator
from app.collector.models import CompanyIdentity, DataQualityMetrics

# ========================================================================
# Logging Setup
# ========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("api_server")

# ========================================================================
# FastAPI App
# ========================================================================
app = FastAPI(
    title="Financial Analysis Platform",
    version="25.0.0",
    description="""
    Enterprise-grade financial data collection & analysis platform.
    
    ## Features
    - **Multi-source Data Collection**: SEC, Yahoo, FMP, AlphaVantage, Polygon, NewsAPI
    - **Cross Verification**: Multi-source validation with dynamic thresholds
    - **Valuation Models**: DCF, EV, Multiples
    - **Peer Analysis**: SIC-based industry benchmarking
    - **Technical Indicators**: RSI, ATR, MACD, Bollinger Bands
    - **News Sentiment**: Real-time sentiment analysis
    - **Batch Processing**: Parallel collection for multiple tickers
    
    ## Data Sources
    - SEC Edgar (XBRL metadata included)
    - Yahoo Finance (40+ extended fields)
    - Financial Modeling Prep
    - AlphaVantage
    - Polygon.io
    - NewsAPI
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================================================
# Global State
# ========================================================================
orchestrator: Optional[CollectorOrchestrator] = None


# ========================================================================
# Pydantic Models
# ========================================================================
class CollectRequest(BaseModel):
    """단일 티커 수집 요청"""
    ticker: str = Field(..., description="Stock ticker symbol", example="AAPL")
    include_diagnostics: bool = Field(True, description="Include detailed diagnostics")
    include_xbrl_metadata: bool = Field(False, description="Include XBRL metadata samples")
    include_peer_analysis: bool = Field(True, description="Include peer benchmarking")
    include_news_sentiment: bool = Field(True, description="Include news sentiment analysis")


class BatchCollectRequest(BaseModel):
    """배치 티커 수집 요청"""
    tickers: List[str] = Field(..., description="List of ticker symbols", example=["AAPL", "GOOGL", "MSFT"])
    max_workers: int = Field(10, ge=1, le=20, description="Number of parallel workers")
    timeout_per_ticker: int = Field(120, ge=30, le=300, description="Timeout per ticker (seconds)")


class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str
    version: str
    orchestrator_initialized: bool
    api_keys_configured: Dict[str, bool]


# ========================================================================
# Startup/Shutdown Events
# ========================================================================
@app.on_event("startup")
async def startup_event():
    """앱 시작 시 오케스트레이터 초기화"""
    global orchestrator
    
    logger.info("🚀 Starting Financial Analysis Platform v25.0.0")
    
    # API 키 로드
    fmp_key = os.getenv("FMP_API_KEY")
    av_key = os.getenv("ALPHAVANTAGE_API_KEY")
    polygon_key = os.getenv("POLYGON_API_KEY")
    news_api_key = os.getenv("NEWS_API_KEY")
    
    # 설정된 API 키 로깅
    logger.info(f"📋 API Keys Configuration:")
    logger.info(f"  - FMP: {'✓ Configured' if fmp_key else '✗ Missing'}")
    logger.info(f"  - AlphaVantage: {'✓ Configured' if av_key else '✗ Missing'}")
    logger.info(f"  - Polygon: {'✓ Configured' if polygon_key else '✗ Missing'}")
    logger.info(f"  - NewsAPI: {'✓ Configured' if news_api_key else '✗ Missing'}")
    
    try:
        orchestrator = CollectorOrchestrator(
            fmp_key=fmp_key,
            av_key=av_key,
            polygon_key=polygon_key,
            news_api_key=news_api_key,
            enable_peer_analysis=True,
            enable_news_sentiment=bool(news_api_key),
            max_rounds=3,
            target_coverage=0.90
        )
        logger.info("✅ Orchestrator initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize orchestrator: {e}")
        # orchestrator는 None으로 유지 (health check에서 확인 가능)


@app.on_event("shutdown")
async def shutdown_event():
    """앱 종료 시 정리"""
    logger.info("🛑 Shutting down Financial Analysis Platform")


# ========================================================================
# API Endpoints
# ========================================================================
@app.get("/", tags=["Health"])
def root():
    """루트 경로 - 기본 정보"""
    return {
        "service": "Financial Analysis Platform",
        "version": "25.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """헬스 체크 - 시스템 상태 확인"""
    return HealthResponse(
        status="healthy" if orchestrator else "degraded",
        version="25.0.0",
        orchestrator_initialized=orchestrator is not None,
        api_keys_configured={
            "fmp": bool(os.getenv("FMP_API_KEY")),
            "alphavantage": bool(os.getenv("ALPHAVANTAGE_API_KEY")),
            "polygon": bool(os.getenv("POLYGON_API_KEY")),
            "newsapi": bool(os.getenv("NEWS_API_KEY"))
        }
    )


@app.post("/api/v1/collect", tags=["Data Collection"])
async def collect_ticker(request: CollectRequest):
    """
    단일 티커 데이터 수집
    
    ## 수집 항목
    - 회사 기본 정보 (CIK, SIC, 거래소 등)
    - 재무제표 (손익계산서, 재무상태표, 현금흐름표)
    - 시장 데이터 (주가, 시가총액, 거래량 등)
    - 밸류에이션 (DCF, EV, Multiples)
    - 성장률 (매출, EBITDA, EPS)
    - 기술적 지표 (RSI, ATR, MACD 등)
    - 애널리스트 활동 (최근 14일)
    - 뉴스 감성 분석 (최근 7일)
    - 동종업계 벤치마킹
    
    ## 데이터 소스
    - SEC Edgar (XBRL)
    - Yahoo Finance
    - Financial Modeling Prep
    - AlphaVantage
    - Polygon.io
    - NewsAPI
    """
    if not orchestrator:
        raise HTTPException(
            status_code=503,
            detail="Orchestrator not initialized. Check logs for API key configuration."
        )
    
    logger.info(f"📊 Collecting data for ticker: {request.ticker}")
    
    try:
        # 데이터 수집
        ident, data, attempts, quality, valuation, diag = orchestrator.collect(request.ticker)
        
        # 응답 구성
        response = {
            "ticker": request.ticker.upper(),
            "timestamp": os.getenv("TZ", "UTC"),
            
            # 회사 정보
            "identity": {
                "company_name": ident.company_name,
                "cik": ident.cik,
                "exchange": ident.exchange,
                "country": ident.country,
                "sic_code": ident.sic_code,
                "sic_description": ident.sic_description,
                "website": ident.website,
            },
            
            # 데이터 품질
            "quality": {
                "coverage_pct": round(quality.coverage_pct, 2),
                "verified_pct": round(quality.verified_pct, 2),
                "critical_coverage_pct": round(quality.critical_coverage_pct, 2),
                "high_coverage_pct": round(quality.high_coverage_pct, 2),
                "reliability_score": round(quality.reliability_score, 2),
                "total_fields": quality.total_measurable_fields,
                "filled_fields": quality.filled_fields,
                "verified_fields": quality.verified_fields,
            },
            
            # 핵심 재무 지표
            "financials": {
                "share_price": data.share_price.value if data.share_price else None,
                "shares_outstanding": data.shares_outstanding.value if data.shares_outstanding else None,
                "market_cap": data.market_cap.value if data.market_cap else None,
                "ltm_revenue": data.ltm_revenue.value if data.ltm_revenue else None,
                "ltm_ebitda": data.ltm_ebitda.value if data.ltm_ebitda else None,
                "ltm_eps": data.ltm_eps.value if data.ltm_eps else None,
                "total_debt": data.total_debt.value if data.total_debt else None,
                "cash_and_equivalents": data.cash_and_equivalents.value if data.cash_and_equivalents else None,
                "free_cash_flow": data.free_cash_flow.value if data.free_cash_flow else None,
            },
            
            # 밸류에이션
            "valuation": {
                "market_cap": valuation.market_cap,
                "enterprise_value": valuation.enterprise_value,
                "pe_ratio": valuation.pe_ratio,
                "pb_ratio": valuation.pb_ratio,
                "ev_ebitda": valuation.ev_ebitda,
                "ev_sales": valuation.ev_sales,
                "dcf_equity_value": valuation.dcf_equity_value,
                "dcf_value_per_share": valuation.dcf_value_per_share,
                "dcf_upside_pct": valuation.dcf_upside_pct,
                "calculation_status": valuation.calculation_status,
            },
            
            # 성장률
            "growth": {
                "revenue_growth_rate": data.revenue_growth_rate.value if data.revenue_growth_rate else None,
                "ebitda_growth_rate": data.ebitda_growth_rate.value if data.ebitda_growth_rate else None,
                "eps_growth_rate": data.eps_growth_rate.value if data.eps_growth_rate else None,
            },
            
            # 소스 통계
            "collection_stats": {
                "total_attempts": attempts.total_attempts,
                "successful_attempts": attempts.successful_attempts,
                "failed_attempts": attempts.failed_attempts,
                "sources": dict(attempts.source_stats),
            }
        }
        
        # 상세 진단 정보
        if request.include_diagnostics:
            response["diagnostics"] = {
                "missing_critical_fields": diag.get("missing_critical", []),
                "source_summary": diag.get("source_summary", {}),
                "growth_metrics": diag.get("growth_metrics", {}),
                "advanced_ratios": diag.get("advanced_ratios", {}),
            }
        
        # XBRL 메타데이터
        if request.include_xbrl_metadata:
            response["xbrl_metadata_sample"] = diag.get("xbrl_metadata_sample", {})
        
        # 동종업계 분석
        if request.include_peer_analysis:
            response["peer_analysis"] = diag.get("peer_analysis", {})
        
        # 뉴스 감성
        if request.include_news_sentiment:
            response["news_sentiment"] = diag.get("news_sentiment", {})
        
        # 기술적 지표
        response["technical_indicators"] = diag.get("technical_indicators", {})
        
        # 애널리스트 활동
        response["analyst_activity"] = diag.get("analyst_activity_14d", {})
        
        logger.info(f"✅ Collection complete for {request.ticker}: {quality.coverage_pct:.1f}% coverage")
        return response
    
    except Exception as e:
        logger.error(f"❌ Collection failed for {request.ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Collection failed: {str(e)}")


@app.post("/api/v1/collect/batch", tags=["Data Collection"])
async def collect_batch(request: BatchCollectRequest, background_tasks: BackgroundTasks):
    """
    배치 티커 데이터 수집 (병렬 처리)
    
    ## 특징
    - 최대 20개 티커 동시 처리
    - ThreadPoolExecutor 기반 병렬 처리
    - 티커별 타임아웃 설정 가능
    - 실패한 티커는 에러 메시지와 함께 반환
    
    ## 성능
    - 평균 처리 시간: 10-30초 per ticker (병렬)
    - 10개 티커 = 약 1-3분 (순차: 5-10분)
    """
    if not orchestrator:
        raise HTTPException(
            status_code=503,
            detail="Orchestrator not initialized"
        )
    
    logger.info(f"📊 Batch collection started: {len(request.tickers)} tickers")
    
    try:
        # 배치 수집
        results = orchestrator.collect_batch(
            tickers=request.tickers,
            max_workers=request.max_workers,
            timeout_per_ticker=request.timeout_per_ticker
        )
        
        # 응답 구성
        response = {
            "total_tickers": len(request.tickers),
            "successful": sum(1 for r in results.values() if r.get('success')),
            "failed": sum(1 for r in results.values() if not r.get('success')),
            "results": {}
        }
        
        for ticker, result in results.items():
            if result['success']:
                response["results"][ticker] = {
                    "status": "success",
                    "coverage_pct": result['quality'].coverage_pct,
                    "market_cap": result['valuation'].market_cap,
                    "pe_ratio": result['valuation'].pe_ratio,
                    "dcf_upside_pct": result['valuation'].dcf_upside_pct,
                }
            else:
                response["results"][ticker] = {
                    "status": "failed",
                    "error": result['error']
                }
        
        logger.info(f"✅ Batch collection complete: {response['successful']}/{response['total_tickers']} succeeded")
        return response
    
    except Exception as e:
        logger.error(f"❌ Batch collection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch collection failed: {str(e)}")


@app.get("/api/v1/sources", tags=["System"])
async def list_sources():
    """
    사용 가능한 데이터 소스 목록
    """
    return {
        "sources": [
            {
                "name": "SEC Edgar",
                "type": "official",
                "coverage": ["US stocks"],
                "data": ["financials", "filings", "XBRL metadata"],
                "cost": "free",
                "rate_limit": "10 req/sec"
            },
            {
                "name": "Yahoo Finance",
                "type": "free",
                "coverage": ["global stocks"],
                "data": ["market data", "financials", "analyst ratings"],
                "cost": "free",
                "rate_limit": "2000 req/hour"
            },
            {
                "name": "Financial Modeling Prep",
                "type": "paid",
                "coverage": ["global stocks"],
                "data": ["financials", "ratios", "DCF"],
                "cost": "$14-299/month",
                "rate_limit": "250-750 req/min",
                "required": bool(os.getenv("FMP_API_KEY"))
            },
            {
                "name": "AlphaVantage",
                "type": "freemium",
                "coverage": ["global stocks"],
                "data": ["market data", "technical indicators"],
                "cost": "free (5 req/min) or $49.99/month",
                "rate_limit": "5 req/min (free)",
                "required": bool(os.getenv("ALPHAVANTAGE_API_KEY"))
            },
            {
                "name": "Polygon.io",
                "type": "paid",
                "coverage": ["US stocks"],
                "data": ["market data", "financials"],
                "cost": "$29-399/month",
                "rate_limit": "5-unlimited req/min",
                "required": bool(os.getenv("POLYGON_API_KEY"))
            },
            {
                "name": "NewsAPI",
                "type": "freemium",
                "coverage": ["global news"],
                "data": ["news articles", "sentiment"],
                "cost": "free (100 req/day) or $449/month",
                "rate_limit": "100 req/day (free)",
                "required": bool(os.getenv("NEWS_API_KEY"))
            }
        ]
    }


# ========================================================================
# Main Entry Point
# ========================================================================
if __name__ == "__main__":
    import uvicorn
    
    # 환경변수에서 포트 읽기 (기본값: 8000)
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,  # 개발 모드에서만 True
        log_level="info"
    )