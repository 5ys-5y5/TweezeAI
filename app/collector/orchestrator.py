# app/collector/orchestrator.py
"""
Collector Orchestrator
- 모든 소스를 조율하여 데이터 수집
- 병렬 처리 (ThreadPoolExecutor)
- Cascade filling (다단계 fallback)
- 교차 검증 및 품질 관리
"""
import time
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from collections import defaultdict
from dataclasses import asdict

from .utils import (
    dbg, normalize_ticker, today_kst, create_session, CurrencyNormalizer,
    validate_number  # ✅ 추가
)
from .models import (
    CompanyIdentity, IndicatorDataWithMeta, CollectionAttemptSummary,
    DataQualityMetrics, ValuationResults, FieldMetadata, AttemptLog,
    FIELD_IMPORTANCE_MAP, FieldImportance  # 이 줄 추가
)
from .cache import CIKMapper
from .sources import (
    SECEdgarSource, YahooFinanceSource, FinancialModelingPrepSource,
    AlphaVantageSource, PolygonIOSource, NewsAPISource, MacroFinanceSource,
    IndustryAverageEstimator, HistoricalTrendEstimator, FinnhubSource
)
from .analyzers import (
    PeerAnalyzer, TechnicalIndicatorsCalculator,
    AnalystActivityTracker, GrowthRateCalculator
)
from .calculators import (
    CrossVerificationEngine, ValuationCalculator, DerivedRatiosCalculator
)


# ========================================================================
# Collector Orchestrator
# ========================================================================
class CollectorOrchestrator:
    """데이터 수집 오케스트레이터"""
    
    SOURCE_PRIORITY = {
        "SEC": 0,
        "Yahoo": 1,
        "Finnhub": 1,
        "FMP": 2,
        "AlphaVantage": 3,
        "Polygon": 3,
        "Estimated(SectorAvg)": 4,
        "Estimated(Historical)": 5,
    }

    def __init__(
        self,
        fmp_key: Optional[str] = None,
        av_key: Optional[str] = None,
        polygon_key: Optional[str] = None,
        news_api_key: Optional[str] = None,
        finnhub_key: Optional[str] = None,
        enable_peer_analysis: bool = True,
        enable_news_sentiment: bool = True,
        peer_limit: int = 10,
        news_days: int = 7,
        max_rounds: int = 3,
        target_coverage: float = 0.90,
        time_budget_sec: int = 7200
    ):
        """
        초기화
        
        Args:
            fmp_key: Financial Modeling Prep API 키
            av_key: AlphaVantage API 키
            polygon_key: Polygon.io API 키
            news_api_key: NewsAPI 키
            enable_peer_analysis: 동종업계 분석 활성화
            enable_news_sentiment: 뉴스 감성 분석 활성화
            peer_limit: 동종업계 회사 수
            news_days: 뉴스 수집 기간 (일)
            max_rounds: 최대 cascade 라운드
            target_coverage: 목표 커버리지 (%)
            time_budget_sec: 시간 제한 (초)
        """
        self.session = create_session()
        self.cik_mapper = CIKMapper()
        
        # Core sources
        self.sec = SECEdgarSource(self.cik_mapper, self.session)
        self.yahoo = YahooFinanceSource()
        self.finnhub = FinnhubSource(finnhub_key, self.session)
        self.fmp = FinancialModelingPrepSource(fmp_key, self.session)
        self.av = AlphaVantageSource(av_key, self.session)
        self.polygon = PolygonIOSource(polygon_key, self.session)
        self.macro = MacroFinanceSource(self.session)
        
        # Estimators
        self.industry_est = IndustryAverageEstimator()
        self.hist_est = HistoricalTrendEstimator()
        
        # Analyzers
        self.peer_analyzer = PeerAnalyzer(self.session, self.cik_mapper) if enable_peer_analysis else None
        self.technical_calc = TechnicalIndicatorsCalculator()
        self.analyst_tracker = AnalystActivityTracker(self.session)
        self.news_api = NewsAPISource(news_api_key, self.session) if (news_api_key and enable_news_sentiment) else None
        self.growth_calc = GrowthRateCalculator()
        
        # Utils
        self.fx = CurrencyNormalizer(self.session, "USD")
        
        # Config
        self.enable_peer_analysis = enable_peer_analysis
        self.enable_news_sentiment = enable_news_sentiment
        self.peer_limit = peer_limit
        self.news_days = news_days
        self.max_rounds = max_rounds
        self.target_coverage = target_coverage
        self.time_budget_sec = time_budget_sec

    def _ensure_periods_for_ttm(self, data: IndicatorDataWithMeta):
        """TTM 필드에 기간 정보 보강"""
        for fname in dir(data):
            if not fname.startswith(('ltm_', 'ntm_')):
                continue
            fm = getattr(data, fname, None)
            if isinstance(fm, FieldMetadata) and not fm.period:
                fm.period = fm.date_retrieved or today_kst()

    def enrich_identity(self, ticker: str, base: Optional[CompanyIdentity]) -> CompanyIdentity:
        """회사 정보 보강 (Yahoo 추가 정보)"""
        ident = base or CompanyIdentity(ticker=ticker)
        
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info
            ident.website = ident.website or info.get("website") or info.get("websiteUrl")
            ident.exchange = ident.exchange or info.get("exchange")
            ident.country = ident.country or info.get("country")
        except Exception as e:
            dbg(1100, f"enrich_identity warn: {e}")
        
        # Accounting standard 추론
        if not ident.accounting_standard:
            if ident.country in (None, "", "United States", "USA", "US"):
                ident.accounting_standard = "US GAAP"
        
        return ident

    def _merge_field(
        self,
        current: Optional[FieldMetadata],
        new: Optional[FieldMetadata]
    ) -> Optional[FieldMetadata]:
        """두 FieldMetadata 병합 (우선순위: 검증 > 신뢰도 > 신선도 > 소스)"""
        if new is None:
            return current
        if current is None:
            return new
        
        try:
            # 검증 여부
            if new.verified and not current.verified:
                return new
            if current.verified and not new.verified:
                return current
            
            # 가중 신뢰도 (신선도 반영)
            from .utils import freshness_weight
            cw = (current.reliability or 0.0) * freshness_weight(current.period)
            nw = (new.reliability or 0.0) * freshness_weight(new.period)
            
            if abs(nw - cw) > 0.05:
                return new if nw > cw else current
            
            # 기간 비교
            c_date = (current.period or "")[:10]
            n_date = (new.period or "")[:10]
            if n_date != c_date:
                return new if n_date > c_date else current
            
            # 신뢰도 비교
            if (new.reliability or 0.0) > (current.reliability or 0.0):
                return new
            
            # 소스 우선순위
            sp_c = self.SOURCE_PRIORITY.get(current.source, 999)
            sp_n = self.SOURCE_PRIORITY.get(new.source, 999)
            if sp_n != sp_c:
                return new if sp_n < sp_c else current

            return current
        except Exception as e:
            dbg(1150, f"merge_field error: {e}")
            return new

    def _cascade_fill_missing(
        self,
        field_name: str,
        existing: Optional[FieldMetadata],
        ticker: str,
        ctx: IndicatorDataWithMeta,
        attempts: CollectionAttemptSummary
    ) -> Optional[FieldMetadata]:
        """Cascade fallback으로 누락 필드 채우기"""

        # ✅ 이미 값이 있고 신뢰도가 높으면 스킵
        if existing and existing.value is not None:
            reliability = existing.reliability or 0.0
            if reliability >= 0.75:  # 신뢰도 75% 이상이면 스킵
                return existing

        sources = [
            self.finnhub, self.fmp, self.av, self.polygon,
            self.industry_est, self.hist_est
        ]
        
        chosen = existing

        for src in sources:

            # ✅ API 키가 없는 소스는 미리 스킵
            if hasattr(src, 'api_key') and not src.api_key:
                continue

            try:
                # FMP는 배치 페칭
                if src == self.fmp:
                    all_fields = src.fetch_all_available_fields(ticker)
                    val = all_fields.get(field_name)
                else:
                    val = src.fetch_field(ticker, field_name, ctx)
                
                attempts.total_attempts += 1
                attempts.source_stats.setdefault(src.name, {'attempted': 0, 'succeeded': 0})
                attempts.source_stats[src.name]['attempted'] += 1
                attempts.field_attempts.setdefault(field_name, [])
                
                if val is not None:
                    attempts.successful_attempts += 1
                    attempts.source_stats[src.name]['succeeded'] += 1
                    attempts.field_attempts[field_name].append(
                        AttemptLog(source=src.name, method="cascade", success=True)
                    )
                    chosen = self._merge_field(chosen, val)
                else:
                    attempts.failed_attempts += 1
                    attempts.field_attempts[field_name].append(
                        AttemptLog(source=src.name, method="cascade", success=False, error="None")
                    )
            except Exception as e:
                attempts.failed_attempts += 1
                attempts.field_attempts.setdefault(field_name, [])
                attempts.field_attempts[field_name].append(
                    AttemptLog(source=src.name, method="cascade", success=False, error=str(e))
                )
                continue
        
        return chosen

    def _collect_candidates_and_cascade(
        self,
        ticker: str,
        ctx: IndicatorDataWithMeta,
        attempts: CollectionAttemptSummary
    ) -> Dict[str, List[FieldMetadata]]:
        """후보 수집 및 cascade filling"""
        dbg(1200, f"Starting candidate collection: {ticker}")
        
        # Schema fields 추출
        schema_fields = []
        annotations = getattr(IndicatorDataWithMeta, "__annotations__", {})
        for fname, anno in annotations.items():
            if "FieldMetadata" in str(anno):
                schema_fields.append(fname)
        
        candidates: Dict[str, List[FieldMetadata]] = defaultdict(list)
        
        # 기존 값 수집
        for fname in schema_fields:
            val = getattr(ctx, fname, None)
            if isinstance(val, FieldMetadata) and val.value is not None:
                candidates[fname].append(val)
        
        # Cascade filling (최대 max_rounds)
        start_ts = time.time()
        prev_crit = -1.0
        
        for round_num in range(self.max_rounds):
            if (time.time() - start_ts) > self.time_budget_sec:
                dbg(1201, "Time budget exceeded")
                break
            
            dbg(1201, f"Cascade round {round_num + 1}/{self.max_rounds}")
            
            for fname in schema_fields:
                base_val = getattr(ctx, fname, None)
                if not isinstance(base_val, FieldMetadata):
                    base_val = None
                
                try:
                    new_val = self._cascade_fill_missing(fname, base_val, ticker, ctx, attempts)
                    if new_val is not None and new_val.value is not None:
                        candidates[fname].append(new_val)
                        if getattr(ctx, fname, None) is None:
                            setattr(ctx, fname, new_val)
                except Exception as e:
                    dbg(1202, f"cascade {round_num} {fname} error: {e}")
            
            # 품질 체크
            try:
                temp_quality = CrossVerificationEngine.calculate_quality_metrics(ctx)
                dbg(1203, f"Round {round_num + 1}: coverage={temp_quality.coverage_pct:.1f}%, critical={temp_quality.critical_coverage_pct:.1f}%")
                
                # 개선 없으면 조기 종료
                if prev_crit >= 0 and (temp_quality.critical_coverage_pct <= prev_crit + 1e-6):
                    dbg(1204, "No improvement, stopping early")
                    break
                prev_crit = temp_quality.critical_coverage_pct

                # 목표 달성 시 종료
                if (temp_quality.coverage_pct >= self.target_coverage * 100) and \
                   (temp_quality.critical_coverage_pct >= 95.0):
                    dbg(1204, "Target coverage reached")
                    break
            except Exception as e:
                dbg(1204, f"Quality check error: {e}")
        
        return candidates

    def _collect_single(self, ticker: str) -> Tuple[
        CompanyIdentity, IndicatorDataWithMeta, CollectionAttemptSummary,
        DataQualityMetrics, ValuationResults, Dict[str, Any]
    ]:
        """단일 티커 수집 (내부용)"""
        return self.collect(ticker)

    def collect_batch(
        self,
        tickers: List[str],
        max_workers: int = 10,
        timeout_per_ticker: int = 120
    ) -> Dict[str, Dict[str, Any]]:
        """배치 수집 (병렬 처리)"""
        dbg(1300, f"Batch collection: {len(tickers)} tickers, {max_workers} workers")
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(self._collect_single, ticker): ticker
                for ticker in tickers
            }
            
            for future in as_completed(future_to_ticker, timeout=timeout_per_ticker * len(tickers)):
                ticker = future_to_ticker[future]
                try:
                    result = future.result(timeout=timeout_per_ticker)
                    results[ticker] = {
                        'success': True,
                        'identity': result[0],
                        'data': result[1],
                        'attempts': result[2],
                        'quality': result[3],
                        'valuation': result[4],
                        'diagnostics': result[5]
                    }
                    dbg(1301, f"{ticker}: ✓ SUCCESS - coverage {result[3].coverage_pct:.1f}%")
                except TimeoutError:
                    results[ticker] = {
                        'success': False,
                        'error': f'Timeout after {timeout_per_ticker}s'
                    }
                    dbg(1302, f"{ticker}: ✗ TIMEOUT")
                except Exception as e:
                    results[ticker] = {
                        'success': False,
                        'error': str(e)
                    }
                    dbg(1303, f"{ticker}: ✗ ERROR - {e}")
        
        success_count = sum(1 for r in results.values() if r.get('success'))
        dbg(1304, f"Batch complete: {success_count}/{len(tickers)} succeeded")
        return results

    def collect(self, ticker: str) -> Tuple[
        CompanyIdentity, IndicatorDataWithMeta, CollectionAttemptSummary,
        DataQualityMetrics, ValuationResults, Dict[str, Any]
    ]:
        """
        단일 티커 수집 (메인 메서드)
        
        Returns:
            (identity, data, attempts, quality, valuation, diagnostics)
        """
        t = normalize_ticker(ticker)
        attempts = CollectionAttemptSummary(ticker=t)
        ctx = IndicatorDataWithMeta()
        
        dbg(1400, f"=== Collection start: {ticker} ===")

        # ===== 1. Company Identity =====
        ident, errs = self.sec.fetch_company_info(t, attempts)
        if errs:
            dbg(1401, f"SEC identity errors: {errs}")

        # ===== 2. SEC Indicators =====
        sec_ind, errs = self.sec.fetch_indicators(t, ident.cik if ident else None, attempts)
        if errs:
            dbg(1402, f"SEC indicators errors: {errs}")
        if sec_ind:
            ctx = sec_ind

        # ===== 3. Yahoo Finance =====
        yah_ind, yerrs = self.yahoo.fetch_indicators(t, attempts)
        if yerrs:
            dbg(1403, f"Yahoo errors: {yerrs}")
        
        if yah_ind:
            # Merge Yahoo data
            for fn in dir(ctx):
                if fn.startswith("_"):
                    continue
                cv = getattr(ctx, fn, None)
                yv = getattr(yah_ind, fn, None)
                if isinstance(cv, FieldMetadata) or isinstance(yv, FieldMetadata):
                    merged = self._merge_field(
                        cv if isinstance(cv, FieldMetadata) else None,
                        yv if isinstance(yv, FieldMetadata) else None
                    )
                    if merged is not None:
                        setattr(ctx, fn, merged)

        # ===== 3.5. Finnhub (Yahoo 대체) ===== ✅ 새로 추가
        if yerrs and self.finnhub and self.finnhub.api_key:
            dbg(1403.5, f"Yahoo failed, trying Finnhub fallback...")
            fh_ind, fh_errs = self.finnhub.fetch_indicators(t, attempts)
            
            if fh_errs:
                dbg(1403.6, f"Finnhub errors: {fh_errs}")
            
            if fh_ind:
                # Merge Finnhub data
                for fn in dir(ctx):
                    if fn.startswith("_"):
                        continue
                    cv = getattr(ctx, fn, None)
                    fv = getattr(fh_ind, fn, None)
                    if isinstance(cv, FieldMetadata) or isinstance(fv, FieldMetadata):
                        merged = self._merge_field(
                            cv if isinstance(cv, FieldMetadata) else None,
                            fv if isinstance(fv, FieldMetadata) else None
                        )
                        if merged is not None:
                            setattr(ctx, fn, merged)
                dbg(1403.7, "✓ Finnhub data merged")

        # ===== 4. Macro DCF Parameters =====
        try:
            self.macro.populate_dcf_params(ident, ctx, attempts)
        except Exception as e:
            dbg(1404, f"Macro DCF param error: {e}")

        # ===== 5. TTM Period 보강 =====
        self._ensure_periods_for_ttm(ctx)

        # ===== 6. Candidate Collection & Cascade =====
        dbg(1405, "Starting cascade filling")
        field_candidates = self._collect_candidates_and_cascade(t, ctx, attempts)
        
        # ===== 7. Cross Verification =====
        dbg(1406, "Starting cross-verification")
        final_data = IndicatorDataWithMeta()
        
        # Shares 추출 (per-share 변환용)
        shares = None
        if field_candidates.get('shares_outstanding'):
            for cand in field_candidates['shares_outstanding']:
                shares_val = validate_number(cand.value)
                if shares_val is not None:
                    shares = shares_val
                    break
        
        # 교차 검증
        for fname, vals in field_candidates.items():
            if not vals:
                continue
            
            first = next((c for c in vals if c and c.value is not None), None)
            if first is None:
                continue
            
            if len(vals) == 1:
                setattr(final_data, fname, vals[0])
                continue
            
            try:
                if isinstance(first.value, (int, float)):
                    verified_field = CrossVerificationEngine.verify_numeric_field(
                        fname, vals, tolerance_pct=5.0, shares=shares
                    )
                else:
                    verified_field = CrossVerificationEngine.verify_string_field(fname, vals)
                
                setattr(final_data, fname, verified_field)
            except Exception as e:
                dbg(1407, f"Verification error for {fname}: {e}")
                best = max(vals, key=lambda x: x.reliability or 0.0)
                setattr(final_data, fname, best)
        
        # ===== 8. Currency Normalization =====
        dbg(1408, "Currency normalization")
        try:
            annotations = getattr(IndicatorDataWithMeta, "__annotations__", {})
            for fname, anno in annotations.items():
                if fname in ('shares_outstanding', 'shares_float'):
                    continue
                
                fval = getattr(final_data, fname, None)
                if isinstance(fval, FieldMetadata) and isinstance(fval.value, (int, float)):
                    normalized = self.fx.normalize_field(fval, shares=shares)
                    if normalized:
                        setattr(final_data, fname, normalized)
        except Exception as e:
            dbg(1409, f"Normalization error: {e}")
        
        # ===== 9. Enrich Identity =====
        ident = self.enrich_identity(t, ident)
        
        # ===== 10. Derived Ratios =====
        dbg(1410, "Calculating derived ratios")
        final_data = DerivedRatiosCalculator.calculate_all(final_data)
        
        # ===== 11. Growth Rates =====
        dbg(1411, "Calculating growth rates")
        if self.sec and ident and ident.cik:
            try:
                historical = self.sec.fetch_historical_data(t, ident.cik, periods=8)
                if historical:
                    growth_fields = self.growth_calc.calculate_growth_rates(historical, final_data)
                    for fname, fm in growth_fields.items():
                        setattr(final_data, fname, fm)
                    dbg(1412, f"Growth rates: {list(growth_fields.keys())}")
            except Exception as e:
                dbg(1412, f"Growth rate error: {e}")
        
        # ===== 12. Quality Metrics =====
        quality = CrossVerificationEngine.calculate_quality_metrics(final_data)
        dbg(1413, f"Quality: coverage={quality.coverage_pct:.1f}%, verified={quality.verified_pct:.1f}%")
        
        # ===== 13. Valuation =====
        readiness = ValuationCalculator.check_readiness(final_data)
        attempts.valuation_readiness = readiness
        valuation = ValuationCalculator.calculate_all(final_data, readiness)
        
        # ===== 14. Advanced Analytics =====
        
        # Peer Analysis
        peer_analysis = {}
        if self.peer_analyzer and ident and ident.sic_code:
            try:
                mcap = validate_number(final_data.market_cap.value) if final_data.market_cap else None
                peers = self.peer_analyzer.get_peer_companies(t, ident.sic_code, mcap, limit=self.peer_limit)
                if peers:
                    peer_medians = self.peer_analyzer.calculate_peer_metrics(peers)
                    
                    # Relative valuation
                    relative_val = {}
                    if final_data.trailing_pe and peer_medians.get('pe_ratio'):
                        current_pe = validate_number(final_data.trailing_pe.value)
                        peer_pe = peer_medians['pe_ratio']
                        if current_pe and peer_pe:
                            relative_val['pe_discount_to_peers'] = ((current_pe / peer_pe) - 1) * 100
                    
                    peer_analysis = {
                        'peer_companies': peers,
                        'peer_medians': peer_medians,
                        'relative_valuation': relative_val
                    }
                    dbg(1414, f"Peer analysis: {len(peers)} peers")
            except Exception as e:
                dbg(1415, f"Peer analysis error: {e}")
        
        # Technical Indicators
        technical_indicators = self.technical_calc.calculate_all(t)
        
        # Analyst Activity
        analyst_activity = self.analyst_tracker.fetch_recent_activity(t, days=14)
        
        # News Sentiment
        news_sentiment = {}
        if self.news_api:
            try:
                news_sentiment = self.news_api.fetch_recent_news_sentiment(
                    t,
                    ident.company_name if ident else None,
                    days=self.news_days
                )
                if news_sentiment:
                    dbg(1416, f"News sentiment: {news_sentiment.get('total_articles', 0)} articles")
            except Exception as e:
                dbg(1417, f"News sentiment error: {e}")
        
        # ===== 15. Diagnostics =====
        diag = {
            "missing_critical": [
                k for k, v in FIELD_IMPORTANCE_MAP.items()
                if v in (FieldImportance.CRITICAL, FieldImportance.EV_CRITICAL, FieldImportance.DCF_CRITICAL)
                and (not getattr(final_data, k, None) or getattr(final_data, k).value is None)
            ],
            "data_quality": {
                "coverage_pct": quality.coverage_pct,
                "verified_pct": quality.verified_pct,
                "critical_coverage_pct": quality.critical_coverage_pct,
                "high_coverage_pct": quality.high_coverage_pct,
            },
            "source_summary": dict(attempts.source_stats),
            "xbrl_metadata_sample": self._extract_xbrl_sample(final_data),
            "growth_metrics": {
                "revenue_growth": getattr(final_data.revenue_growth_rate, 'value', None) if final_data.revenue_growth_rate else None,
                "ebitda_growth": getattr(final_data.ebitda_growth_rate, 'value', None) if final_data.ebitda_growth_rate else None,
                "eps_growth": getattr(final_data.eps_growth_rate, 'value', None) if final_data.eps_growth_rate else None,
            },
            "advanced_ratios": {
                "roic": getattr(final_data.roic, 'value', None) if final_data.roic else None,
                "roce": getattr(final_data.roce, 'value', None) if final_data.roce else None,
                "interest_coverage": getattr(final_data.interest_coverage, 'value', None) if final_data.interest_coverage else None,
            },
            "peer_analysis": peer_analysis,
            "news_sentiment": news_sentiment,
            "technical_indicators": asdict(technical_indicators),
            "analyst_activity_14d": asdict(analyst_activity),
        }

        dbg(1499, f"=== Collection complete: {ticker} ===")
        return ident, final_data, attempts, quality, valuation, diag

    def _extract_xbrl_sample(self, data: IndicatorDataWithMeta) -> Dict[str, Any]:
        """XBRL 메타데이터 샘플 추출"""
        sample = {}
        key_fields = [
            'ltm_revenue', 'ltm_ebitda', 'shares_outstanding',
            'cash_and_equivalents', 'total_debt'
        ]
        
        for fname in key_fields:
            fm = getattr(data, fname, None)
            if isinstance(fm, FieldMetadata) and fm.xbrl_tag:
                sample[fname] = {
                    'xbrl_tag': fm.xbrl_tag,
                    'xbrl_unit': fm.xbrl_unit,
                    'frame': fm.frame,
                    'accession': fm.accession,
                    'filing_date': fm.filing_date,
                    'filing_url': fm.filing_url,
                }
        
        return sample