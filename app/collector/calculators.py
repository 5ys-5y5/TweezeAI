# app/collector/calculators.py
"""
계산 엔진 통합
- CrossVerificationEngine (교차 검증)
- ValuationCalculator (밸류에이션)
- DerivedRatiosCalculator (파생 비율 계산)
"""
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import asdict
from collections import defaultdict, Counter

from .utils import (
    dbg, validate_number, today_kst, freshness_weight,
    is_per_share, convert_per_share
)
from .models import (
    FieldMetadata, IndicatorDataWithMeta, ValuationReadiness,
    ValuationResults, DataQualityMetrics, DerivedCalculation,
    FIELD_IMPORTANCE_MAP, FieldImportance
)

try:
    import pandas as pd
except ImportError:
    pd = None


# ========================================================================
# Cross Verification Engine (교차 검증)
# ========================================================================
class CrossVerificationEngine:
    """교차 검증 엔진 (다중 소스 데이터 검증)"""
    
    @staticmethod
    def _same_period(a: Optional[str], b: Optional[str]) -> bool:
        """기간 동일 여부 판단 (±15일 + 분기 일치)"""
        try:
            if not a or not b:
                return False
            da = datetime.strptime(a[:10], "%Y-%m-%d")
            db = datetime.strptime(b[:10], "%Y-%m-%d")
            
            # ±15일 이내
            if abs((da - db).days) <= 15:
                return True
            
            # 같은 분기 (YYYY-Qn)
            qa = (da.year, (da.month - 1)//3 + 1)
            qb = (db.year, (db.month - 1)//3 + 1)
            return qa == qb
        except Exception:
            return False

    @staticmethod
    def verify_numeric_field(
        field_name: str,
        values: List[FieldMetadata],
        tolerance_pct: float = 5.0,
        shares: Optional[float] = None
    ) -> FieldMetadata:
        """숫자 필드 교차 검증"""
        dbg(700, f"verify_numeric_field: {field_name}, n={len(values)}")
        
        if not values:
            return FieldMetadata(value=None, source="none", verified=False)

        # 1) Per-share 정규화
        normed: List[FieldMetadata] = []
        for v in values:
            try:
                val = validate_number(v.value)
                if val is None:
                    continue
                
                if v.unit and is_per_share(v.unit):
                    conv = convert_per_share(val, v.unit, "USD", shares)
                    if conv is not None:
                        vv = FieldMetadata(
                            **{**asdict(v), "value": conv, "unit": "USD", "currency": v.currency or "USD"}
                        )
                        normed.append(vv)
                        continue
                
                normed.append(v)
            except Exception as e:
                dbg(701, f"normalize error: {e}")

        if not normed:
            return values[0]

        # 2) Cohort 그룹화 (statement, period, currency)
        def cohort_key(v: FieldMetadata) -> Tuple[str, str, str]:
            st = (v.statement or "").split("(")[0].upper()
            pe = (v.period or "")[:10]
            cc = (v.currency or "USD").upper()
            return (st, pe, cc)

        cohorts: Dict[Tuple[str,str,str], List[FieldMetadata]] = defaultdict(list)
        for v in normed:
            cohorts[cohort_key(v)].append(v)

        def cohort_sort_key(k: Tuple[str,str,str]) -> Tuple[int, str]:
            st, pe, _ = k
            rank = 0 if st in ("IS","BS","CF") else 1
            return (rank, pe)

        sorted_cohorts = sorted(cohorts.items(), key=lambda kv: cohort_sort_key(kv[0]), reverse=True)
        best_meta: Optional[FieldMetadata] = None

        # 3) Dynamic threshold
        thresholds = {
            'ltm_eps': {'relative': 10.0, 'absolute': 0.10},
            'share_price': {'relative': 2.0, 'absolute': 0.50},
            'ltm_revenue': {'relative': 5.0, 'absolute': 1_000_000.0},
            'effective_tax_rate': {'relative': 10.0, 'absolute': 0.05},
        }
        thr = thresholds.get(field_name, {'relative': tolerance_pct, 'absolute': float('inf')})

        for key, cohort in sorted_cohorts:
            if len(cohort) == 1:
                cand = cohort[0]
                if not best_meta or (cand.reliability > best_meta.reliability):
                    best_meta = FieldMetadata(**{**asdict(cand), "verified": False})
                continue
            
            try:
                # Weighted mean
                weighted_sum = sum(float(v.value) * (v.reliability or 0.5) for v in cohort)
                weight_total = sum((v.reliability or 0.5) for v in cohort) or 1.0
                mean_value = weighted_sum / weight_total
                matches = []
                
                for v in cohort:
                    val = float(v.value)
                    rel = abs(val - mean_value) / (abs(mean_value) + 1e-12) * 100.0
                    absdev = abs(val - mean_value)
                    
                    # Dynamic absolute threshold
                    base_abs = float(thr.get('absolute', float('inf')))
                    rel_win = float(thr.get('relative', tolerance_pct)) / 100.0
                    log_adj = max(1.0, 10 ** max(0.0, math.log10(abs(mean_value) + 1e-9) - 6))
                    abs_dyn = max(base_abs, abs(mean_value) * rel_win * 0.5)
                    abs_thr = max(base_abs, abs_dyn) * log_adj
                    
                    if (rel <= thr.get('relative', tolerance_pct)) or (absdev <= abs_thr):
                        matches.append(v)
                
                if len(matches) >= 2:
                    best = max(matches, key=lambda x: x.reliability or 0.0)
                    best.verified = True
                    best.cross_check_sources = [v.source for v in matches]
                    dbg(702, f"{field_name} ✓ verified with {len(matches)} matches")
                    return best
                else:
                    cand = max(cohort, key=lambda x: x.reliability or 0.0)
                    if not best_meta or cand.reliability > best_meta.reliability:
                        best_meta = FieldMetadata(**{**asdict(cand), "verified": False})
                    
            except Exception as e:
                dbg(703, f"cohort verify error: {e}")

        return best_meta or normed[0]

    @staticmethod
    def verify_string_field(field_name: str, values: List[FieldMetadata]) -> FieldMetadata:
        """문자열 필드 검증 (다수결)"""
        dbg(710, f"verify_string_field: {field_name}, n={len(values)}")
        
        if not values:
            return FieldMetadata(value=None, source="none", verified=False)
        if len(values) == 1:
            return values[0]
        
        value_counts = Counter(v.value for v in values if v.value)
        if not value_counts:
            return values[0]
        
        most_common_value, count = value_counts.most_common(1)[0]
        if count >= 2:
            matching = [v for v in values if v.value == most_common_value]
            best = max(matching, key=lambda x: x.reliability or 0.0)
            best.verified = True
            best.cross_check_sources = [v.source for v in matching]
            dbg(711, f"{field_name} ✓ verified by majority ({count})")
            return best
        
        best = max(values, key=lambda x: x.reliability or 0.0)
        best.verified = False
        dbg(712, f"{field_name} picked most reliable")
        return best

    @staticmethod
    def calculate_quality_metrics(data: IndicatorDataWithMeta) -> DataQualityMetrics:
        """데이터 품질 지표 계산"""
        dbg(720, "calculate_quality_metrics")
        
        metrics = DataQualityMetrics()
        
        # Measurable fields 추출
        from typing import get_origin, get_args
        measurable_fields = []
        annotations = getattr(IndicatorDataWithMeta, "__annotations__", {})
        for fname, anno in annotations.items():
            origin = get_origin(anno)
            args = get_args(anno)
            is_fm = (anno is FieldMetadata) or (origin and FieldMetadata in args)
            if is_fm:
                measurable_fields.append(fname)
        
        metrics.total_measurable_fields = len(measurable_fields)
        
        filled = verified = 0
        reliability_sum = 0.0
        reliability_count = 0
        freshness_scores: List[float] = []
        
        critical_filled = critical_total = 0
        high_filled = high_total = 0
        
        verification_details: Dict[str, Any] = {}

        for fname in measurable_fields:
            field = getattr(data, fname, None)
            importance = FIELD_IMPORTANCE_MAP.get(fname, FieldImportance.MEDIUM)
            
            if importance in (FieldImportance.CRITICAL, FieldImportance.EV_CRITICAL, FieldImportance.DCF_CRITICAL):
                critical_total += 1
            if importance == FieldImportance.HIGH:
                high_total += 1

            if isinstance(field, FieldMetadata) and field.value is not None:
                filled += 1
                if field.verified:
                    verified += 1
                if field.reliability is not None:
                    reliability_sum += float(field.reliability)
                    reliability_count += 1
                
                if field.date_retrieved:
                    try:
                        dt = datetime.strptime(field.date_retrieved[:10], "%Y-%m-%d")
                        days = max(0, (datetime.utcnow() - dt).days)
                        freshness_scores.append(max(0.0, 1.0 - min(days, 365) / 365.0))
                    except Exception:
                        pass

                if importance in (FieldImportance.CRITICAL, FieldImportance.EV_CRITICAL, FieldImportance.DCF_CRITICAL):
                    critical_filled += 1
                if importance == FieldImportance.HIGH:
                    high_filled += 1

                verification_details[fname] = {
                    "source": field.source,
                    "verified": field.verified,
                    "reliability": field.reliability,
                    "importance": importance.value,
                }

        metrics.filled_fields = filled
        metrics.verified_fields = verified
        metrics.coverage_pct = (filled / metrics.total_measurable_fields * 100) if metrics.total_measurable_fields else 0.0
        metrics.verified_pct = (verified / filled * 100) if filled else 0.0
        metrics.reliability_score = (reliability_sum / reliability_count) if reliability_count else 0.0
        metrics.freshness_score = sum(freshness_scores) / len(freshness_scores) if freshness_scores else 0.0
        metrics.critical_coverage_pct = (critical_filled / critical_total * 100) if critical_total else 0.0
        metrics.high_coverage_pct = (high_filled / high_total * 100) if high_total else 0.0
        metrics.field_verification_details = verification_details
        
        dbg(721, f"Quality: coverage={metrics.coverage_pct:.1f}%, verified={metrics.verified_pct:.1f}%, critical={metrics.critical_coverage_pct:.1f}%")
        return metrics


# ========================================================================
# Valuation Calculator (밸류에이션 계산)
# ========================================================================
class ValuationCalculator:
    """밸류에이션 계산기 (EV, DCF, Multiples)"""
    
    @staticmethod
    def extract_value(field: Optional[FieldMetadata]) -> Optional[float]:
        """FieldMetadata에서 값 추출"""
        if field and field.value is not None:
            return validate_number(field.value)
        return None

    @staticmethod
    def check_readiness(data: IndicatorDataWithMeta) -> ValuationReadiness:
        """밸류에이션 준비 상태 체크"""
        readiness = ValuationReadiness()
        
        price = ValuationCalculator.extract_value(data.share_price)
        shares = ValuationCalculator.extract_value(data.shares_outstanding)
        mcap_direct = ValuationCalculator.extract_value(data.market_cap)
        
        has_market_cap = (price and shares) or mcap_direct
        has_debt = any(ValuationCalculator.extract_value(x) for x in 
                      [data.total_debt, data.current_debt, data.long_term_debt])
        has_cash = ValuationCalculator.extract_value(data.cash_and_equivalents) is not None

        # EV 준비도
        if has_market_cap and has_debt and has_cash:
            readiness.ev_ready = True
        else:
            if not has_market_cap:
                readiness.ev_missing.append("market_cap")
            if not has_debt:
                readiness.ev_missing.append("debt")
            if not has_cash:
                readiness.ev_missing.append("cash")

        # DCF 준비도
        has_fcf = (data.free_cash_flow) or (data.operating_cash_flow and data.capex)
        if has_fcf and shares:
            readiness.dcf_ready = True
        else:
            if not has_fcf:
                readiness.dcf_missing.append("free_cash_flow")
            if not shares:
                readiness.dcf_missing.append("shares_outstanding")

        # Multiples 준비도
        has_ltm = any(ValuationCalculator.extract_value(x) for x in 
                     [data.ltm_revenue, data.ltm_ebitda, data.ltm_eps])
        if price and has_ltm:
            readiness.multiples_ready = True
        else:
            if not price:
                readiness.multiples_missing.append("share_price")
            if not has_ltm:
                readiness.multiples_missing.append("ltm_metrics")
        
        return readiness

    @staticmethod
    def _cost_of_debt(data: IndicatorDataWithMeta) -> float:
        """부채비용 계산"""
        ie = ValuationCalculator.extract_value(data.interest_expense)
        td = ValuationCalculator.extract_value(data.total_debt)
        if ie and td and td > 0:
            cod = max(0.0, min(0.20, float(ie/td)))
            return cod if cod > 0 else 0.06
        return 0.06

    @staticmethod
    def _wacc(data: IndicatorDataWithMeta, market_cap: Optional[float]) -> float:
        """WACC 계산"""
        if data.wacc and data.wacc.value:
            return float(data.wacc.value)
        
        coe = ValuationCalculator.extract_value(data.cost_of_equity) or 0.10
        cod = ValuationCalculator._cost_of_debt(data)
        etr = ValuationCalculator.extract_value(data.effective_tax_rate) or 0.25
        
        E = float(market_cap or 0.0)
        D = float(ValuationCalculator.extract_value(data.total_debt) or 0.0)
        V = max(1.0, E + D)
        
        return float((E/V)*coe + (D/V)*cod*(1.0 - etr))

    @staticmethod
    def _latest_fcff(data: IndicatorDataWithMeta) -> Optional[float]:
        """최신 FCFF 조회"""
        if data.free_cash_flow and data.free_cash_flow.value:
            return float(data.free_cash_flow.value)
        
        ocf = ValuationCalculator.extract_value(data.operating_cash_flow)
        capex = ValuationCalculator.extract_value(data.capex)
        if ocf and capex:
            return float(ocf - capex)
        
        return None

    @staticmethod
    def _compute_dcf_equity(
        data: IndicatorDataWithMeta,
        market_cap: Optional[float]
    ) -> Tuple[Optional[float], Optional[float], Optional[Dict[str,Any]]]:
        """DCF 주식가치 계산"""
        base_fcff = ValuationCalculator._latest_fcff(data)
        if base_fcff is None:
            return None, None, None
        
        # 성장률 추정
        g_hint = None
        for f in (data.revenue_growth_rate, data.ebitda_growth_rate, data.eps_growth_rate):
            if f and f.value:
                g_hint = float(f.value) / 100.0
                break
        g_mid = g_hint if g_hint else 0.03
        
        wacc = ValuationCalculator._wacc(data, market_cap)
        g_term = ValuationCalculator.extract_value(data.terminal_growth_rate) or 0.025
        
        # 5년 예측
        years = 5
        fcffs = []
        f = base_fcff
        for _ in range(years):
            f = f * (1.0 + g_mid)
            fcffs.append(float(f))
        
        # PV 계산
        ev_sum = sum(f / pow(1.0 + wacc, i) for i, f in enumerate(fcffs, 1))
        tv = fcffs[-1] * (1.0 + g_term) / max(1e-6, (wacc - g_term))
        pv_tv = tv / pow(1.0 + wacc, years)
        ev_dcf = ev_sum + pv_tv
        
        # 주식가치 = EV - Net Debt - Minority - Preferred
        cash = ValuationCalculator.extract_value(data.cash_and_equivalents) or 0.0
        td = ValuationCalculator.extract_value(data.total_debt) or 0.0
        net_debt = float(td - cash)
        minority = ValuationCalculator.extract_value(data.minority_interest) or 0.0
        pref = ValuationCalculator.extract_value(data.preferred_equity) or 0.0
        
        eq = float(ev_dcf - net_debt - minority - pref)
        
        # Sensitivity analysis
        sens = {}
        for dw in (-0.01, 0.0, 0.01):
            for dg in (-0.005, 0.0, 0.005):
                w = max(0.01, wacc + dw)
                g = max(-0.05, min(0.05, g_term + dg))
                ev_s = sum(f/pow(1.0 + w, i) for i, f in enumerate(fcffs, 1))
                tv_s = (fcffs[-1]*(1.0 + g))/max(1e-6, (w - g))
                eq_s = (ev_s + tv_s/pow(1.0 + w, years)) - net_debt - minority - pref
                sens[f"W={w:.3f},g={g:.3f}"] = float(eq_s)
        
        return float(eq), float(wacc), {"grid": sens, "assumptions":{"g_mid":g_mid,"g_term":g_term}}

    @staticmethod
    def calculate_all(data: IndicatorDataWithMeta, readiness: ValuationReadiness) -> ValuationResults:
        """모든 밸류에이션 계산"""
        results = ValuationResults()
        
        price = ValuationCalculator.extract_value(data.share_price)
        shares = ValuationCalculator.extract_value(data.shares_outstanding)
        mcap_direct = ValuationCalculator.extract_value(data.market_cap)

        # Market Cap
        if price and shares:
            results.market_cap = price * shares
            results.calculation_status['market_cap'] = 'calculated'
        elif mcap_direct:
            results.market_cap = mcap_direct
            results.calculation_status['market_cap'] = 'direct'
        else:
            results.calculation_status['market_cap'] = 'unavailable'

        # Enterprise Value
        if readiness.ev_ready and results.market_cap:
            td = ValuationCalculator.extract_value(data.total_debt) or 0.0
            cash = ValuationCalculator.extract_value(data.cash_and_equivalents) or 0.0
            net_debt = td - cash
            
            results.enterprise_value = results.market_cap + net_debt
            results.calculation_status['ev'] = 'calculated'

        # Multiples
        try:
            if results.enterprise_value and data.ltm_ebitda:
                ebitda_val = validate_number(data.ltm_ebitda.value)
                if ebitda_val and ebitda_val != 0:
                    results.ev_ebitda = float(results.enterprise_value / ebitda_val)
            
            if price and data.ltm_eps:
                eps_val = validate_number(data.ltm_eps.value)
                if eps_val and eps_val != 0:
                    results.pe_ratio = float(price / eps_val)
        except Exception as e:
            dbg(730, f"Multiples calculation error: {e}")

        # DCF
        try:
            eq_dcf, wacc_used, sens = ValuationCalculator._compute_dcf_equity(data, results.market_cap)
            if eq_dcf and shares:
                results.dcf_equity_value = float(eq_dcf)
                results.dcf_value_per_share = float(eq_dcf / shares)
                if price:
                    results.dcf_upside_pct = float((results.dcf_value_per_share / price - 1.0) * 100.0)
                results.dcf_sensitivity = sens
                results.calculation_status['dcf'] = f'calculated(wacc={wacc_used:.3f})'
            else:
                results.calculation_status['dcf'] = 'unavailable'
        except Exception as e:
            dbg(731, f"DCF error: {e}")
            results.calculation_status['dcf'] = 'error'

        return results


# ========================================================================
# Derived Ratios Calculator (파생 비율 계산)
# ========================================================================
class DerivedRatiosCalculator:
    """파생 비율 계산기"""
    
    @staticmethod
    def calculate_all(data: IndicatorDataWithMeta) -> IndicatorDataWithMeta:
        """모든 파생 비율 계산"""
        asof = today_kst()
        
        # Current Ratio
        if data.current_assets and data.current_liabilities:
            ca = validate_number(data.current_assets.value)
            cl = validate_number(data.current_liabilities.value)
            if ca and cl and cl != 0:
                data.current_ratio = FieldMetadata(
                    value=ca / cl,
                    source="Derived",
                    statement="BS",
                    date_retrieved=asof,
                    reliability=min(data.current_assets.reliability, data.current_liabilities.reliability)
                )
        
        # Interest Coverage
        if data.ltm_ebit and data.interest_expense:
            ebit = validate_number(data.ltm_ebit.value)
            ie = validate_number(data.interest_expense.value)
            if ebit and ie and ie != 0:
                data.interest_coverage = FieldMetadata(
                    value=ebit / ie,
                    source="Derived",
                    statement="IS",
                    date_retrieved=asof,
                    reliability=min(data.ltm_ebit.reliability, data.interest_expense.reliability)
                )
        
        # Free Cash Flow
        if data.operating_cash_flow and data.capex:
            ocf = validate_number(data.operating_cash_flow.value)
            capex = validate_number(data.capex.value)
            if ocf and capex:
                fcf = ocf - capex
                data.free_cash_flow = FieldMetadata(
                    value=fcf,
                    source="Derived",
                    statement="CF",
                    date_retrieved=asof,
                    reliability=min(data.operating_cash_flow.reliability, data.capex.reliability),
                    currency="USD",
                    unit="USD"
                )
                
                data.derived_calculations['free_cash_flow'] = DerivedCalculation(
                    result=fcf,
                    formula="FCF = OCF - CapEx",
                    components={
                        'ocf': {'value': ocf, 'source': data.operating_cash_flow.source},
                        'capex': {'value': capex, 'source': data.capex.source}
                    }
                )
        
        # ROIC
        if data.ltm_ebit and data.effective_tax_rate and data.total_equity and data.total_debt:
            ebit = validate_number(data.ltm_ebit.value)
            etr = validate_number(data.effective_tax_rate.value)
            equity = validate_number(data.total_equity.value)
            debt = validate_number(data.total_debt.value)
            cash = validate_number(data.cash_and_equivalents.value) if data.cash_and_equivalents else 0.0
            
            if all([ebit, etr is not None, equity, debt]):
                nopat = ebit * (1 - etr)
                invested_capital = equity + debt - (cash or 0.0)
                
                if invested_capital > 0:
                    roic = nopat / invested_capital
                    data.roic = FieldMetadata(
                        value=roic,
                        source="Derived",
                        statement="Mixed",
                        date_retrieved=asof,
                        reliability=0.75,
                        unit="%"
                    )
        
        dbg(740, "Derived ratios calculated")
        return data