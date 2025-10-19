# app/collector/sources.py
"""
데이터 소스 통합 모듈
- SEC Edgar, Yahoo Finance, FMP, AlphaVantage, Polygon, NewsAPI, Macro
- Estimators (Industry Average, Historical Trend)
"""
import re
import json
import hashlib
from functools import lru_cache
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import Counter

from .utils import (
    dbg, validate_number, validate_date, today_kst,
    normalize_ticker, normalize_cik, is_per_share
)
from .models import (
    FieldMetadata, IndicatorDataWithMeta, CompanyIdentity,
    CollectionAttemptSummary, AttemptLog, YAHOO_FIELD_MAPPING
)
from .cache import CIKMapper

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import feedparser
except ImportError:
    feedparser = None


# ========================================================================
# SEC Edgar Source (XBRL Metadata + Historical + Caching)
# ========================================================================
class SECEdgarSource:
    """SEC Edgar API 래퍼 (XBRL 메타데이터 포함)"""
    
    def __init__(self, cik_mapper: CIKMapper, session):
        self.cik_mapper = cik_mapper
        self.session = session

    @staticmethod
    def _parse_quarter_from_frame(frame: str) -> Optional[Tuple[int, int]]:
        """프레임에서 분기 추출 (YYYY-Qn)"""
        if not frame or not isinstance(frame, str):
            return None
        m = re.search(r'(\d{4}).*Q([1-4])', frame)
        if m:
            return int(m.group(1)), int(m.group(2))
        return None

    @staticmethod
    def _generate_filing_url(cik: str, accession: str) -> Optional[str]:
        """SEC EDGAR 뷰어 URL 생성"""
        if not accession or not cik:
            return None
        try:
            return f"https://www.sec.gov/cgi-bin/viewer?action=view&cik={cik}&accession_number={accession}&xbrl_type=v"
        except Exception:
            return None

    @staticmethod
    def _currency_unit_from_meta(meta: Dict[str, Any]) -> Tuple[Optional[str], str]:
        """XBRL unit에서 통화/단위 추출"""
        unit = (meta or {}).get('xbrl_unit') or ''
        u = unit.lower()
        if 'share' in u:
            return None, 'shares'
        m = re.search(r'(?:iso4217:)?([A-Z]{3})', unit, re.IGNORECASE)
        cur = (m.group(1).upper() if m else 'USD')
        return cur, cur

    @staticmethod
    def _select_12m_value(values: List[Dict]) -> Optional[Dict]:
        """12개월 값 선택 (P12M > 4Q 합산 > FY)"""
        if not values:
            return None
        
        # Priority 1: P12M frames
        frames_12m = [v for v in values if isinstance(v.get('frame'), str) and 'P12M' in v['frame']]
        if frames_12m:
            frames_12m.sort(key=lambda x: x.get('end', ''), reverse=True)
            return {
                'val': frames_12m[0]['val'],
                'end': frames_12m[0].get('end'),
                'method': 'P12M',
                'confidence': 1.0,
                'accn': frames_12m[0].get('accn'),
                'filed': frames_12m[0].get('filed'),
                'frame': frames_12m[0].get('frame'),
            }
        
        # Priority 2: Sum last 4 quarters
        quarterlies = [v for v in values if isinstance(v.get('frame'), str) and 'Q' in v['frame'] and 'YTD' not in v['frame']]
        quarterlies = [v for v in quarterlies if SECEdgarSource._parse_quarter_from_frame(v['frame']) is not None]
        
        if quarterlies:
            quarterlies.sort(key=lambda x: x.get('end', ''), reverse=True)
            picked, seen = [], set()
            for v in quarterlies:
                yq = SECEdgarSource._parse_quarter_from_frame(v['frame'])
                if yq and yq not in seen:
                    picked.append(v)
                    seen.add(yq)
                if len(picked) == 4:
                    break
            
            if len(picked) >= 3:
                total = sum(v['val'] for v in picked if isinstance(v.get('val'), (int, float)))
                end = max(v.get('end', '') for v in picked)
                method = f"sum_last_{len(picked)}_quarters"
                confidence = len(picked) / 4.0
                return {
                    'val': total,
                    'end': end,
                    'method': method,
                    'confidence': confidence,
                    'accn': picked[0].get('accn'),
                    'filed': picked[0].get('filed'),
                    'frame': f"Computed_{method}",
                }
        
        # Priority 3: FY
        frames_fy = [v for v in values if isinstance(v.get('frame'), str) and v['frame'].endswith('FY')]
        if frames_fy:
            frames_fy.sort(key=lambda x: x.get('end', ''), reverse=True)
            return {
                'val': frames_fy[0]['val'],
                'end': frames_fy[0].get('end'),
                'method': 'FY',
                'confidence': 0.90,
                'accn': frames_fy[0].get('accn'),
                'filed': frames_fy[0].get('filed'),
                'frame': frames_fy[0].get('frame'),
            }
        
        return None

    @lru_cache(maxsize=100)
    def _get_company_facts_cached(self, cik: str, date: str) -> Optional[Dict]:
        """캐시된 company facts 조회"""
        return self._get_company_facts(cik)

    def _get_company_facts(self, cik: str) -> Optional[Dict]:
        """Company facts API 호출"""
        try:
            url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
            r = self.session.get(url, timeout=15)
            if r.ok:
                dbg(501, f"✓ SEC facts loaded for CIK {cik}")
                return r.json()
        except Exception as e:
            dbg(502, f"✗ SEC facts fetch failed: {e}")
        return None

    def _get_submissions(self, cik: str) -> Optional[Dict]:
        """Submissions API 호출"""
        try:
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            r = self.session.get(url, timeout=15)
            if r.ok:
                return r.json()
        except Exception as e:
            dbg(503, f"Submissions fetch failed: {e}")
        return None

    @staticmethod
    def _get_facts_units(facts: Dict, tag: str) -> Dict[str, List[Dict]]:
        """XBRL 태그의 units 반환 (US GAAP -> IFRS 순서)"""
        try:
            return facts['facts']['us-gaap'][tag]['units']
        except Exception:
            try:
                return facts['facts']['ifrs-full'][tag]['units']
            except Exception:
                return {}

    def fetch_company_info(
        self, 
        ticker: str, 
        attempt_log: CollectionAttemptSummary
    ) -> Tuple[Optional[CompanyIdentity], List[str]]:
        """회사 기본 정보 조회"""
        dbg(510, f"SEC fetch_company_info: {ticker}")
        
        try:
            cik = self.cik_mapper.get_cik(ticker, self.session)
            if not cik:
                return None, ["CIK not found"]
            
            ident = CompanyIdentity(ticker=ticker, cik=cik)
            sub = self._get_submissions(cik)
            
            if sub:
                ident.company_name = sub.get('name')
                ident.sic_code = str(sub.get('sic', ''))
                ident.sic_description = sub.get('sicDescription')
                dbg(511, f"✓ Company info: {ident.company_name}, SIC={ident.sic_code}")
            
            return ident, []
        except Exception as e:
            return None, [str(e)]

    def fetch_indicators(
        self,
        ticker: str,
        cik: Optional[str],
        attempt_log: CollectionAttemptSummary
    ) -> Tuple[Optional[IndicatorDataWithMeta], List[str]]:
        """재무 지표 수집 (XBRL 메타데이터 포함)"""
        dbg(520, f"SEC fetch_indicators: {ticker}")
        
        if not cik:
            ident, errs = self.fetch_company_info(ticker, attempt_log)
            if ident:
                cik = ident.cik
            else:
                return None, errs
        
        asof = today_kst()
        facts = self._get_company_facts_cached(cik, asof)
        
        if not facts:
            return None, ["SEC facts unavailable"]
        
        ind = IndicatorDataWithMeta()
        companyfacts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

        def latest_instant(tag_list: List[str]) -> Optional[Tuple[float, str, Dict[str, Any]]]:
            """최신 instant 값 조회"""
            for tag in tag_list:
                units = self._get_facts_units(facts, tag)
                for unit, vals in units.items():
                    inst = [v for v in vals if (v.get('frame') is None) or 
                           (isinstance(v.get('frame'), str) and v['frame'].endswith('I'))]
                    inst = sorted(inst, key=lambda x: x.get('end', ''), reverse=True)
                    for v in inst:
                        if isinstance(v.get('val'), (int, float)):
                            metadata = {
                                'xbrl_tag': tag,
                                'xbrl_unit': unit,
                                'frame': v.get('frame'),
                                'accession': v.get('accn'),
                                'filing_date': v.get('filed'),
                                'filing_url': self._generate_filing_url(cik, v.get('accn')),
                            }
                            return float(v['val']), v.get('end'), metadata
            return None

        def ltm_from_tags(tag_list: List[str]) -> Optional[Tuple[float, str, str, float, Dict[str, Any]]]:
            """LTM 값 조회"""
            for tag in tag_list:
                units = self._get_facts_units(facts, tag)
                for unit, vals in units.items():
                    pick = self._select_12m_value(vals)
                    if pick and validate_number(pick['val']) is not None:
                        metadata = {
                            'xbrl_tag': tag,
                            'xbrl_unit': unit,
                            'frame': pick.get('frame'),
                            'accession': pick.get('accn'),
                            'filing_date': pick.get('filed'),
                            'filing_url': self._generate_filing_url(cik, pick.get('accn')),
                        }
                        return (
                            float(pick['val']),
                            pick.get('end'),
                            pick.get('method', ''),
                            float(pick.get('confidence', 1.0)),
                            metadata
                        )
            return None

        # === 주요 지표 수집 ===
        
        # Shares Outstanding
        so = latest_instant(['CommonStockSharesOutstanding', 'EntityCommonStockSharesOutstanding'])
        if so:
            val, end, metadata = so
            _cur, unit = self._currency_unit_from_meta(metadata)
            ind.shares_outstanding = FieldMetadata(
                value=val, source="SEC", source_link=companyfacts_url,
                statement="instant", period=end, date_retrieved=asof,
                reliability=0.9, currency=_cur, unit=unit, **metadata
            )

        # Cash & Debt
        cash = latest_instant(['CashAndCashEquivalentsAtCarryingValue', 'CashCashEquivalentsAndShortTermInvestments'])
        if cash:
            val, end, metadata = cash
            cur, unit = self._currency_unit_from_meta(metadata)
            ind.cash_and_equivalents = FieldMetadata(
                value=val, source="SEC", source_link=companyfacts_url,
                statement="BS", period=end, date_retrieved=asof,
                reliability=0.9, currency=cur, unit=unit, **metadata
            )
        
        total_debt = latest_instant(['Debt'])
        if total_debt:
            val, end, metadata = total_debt
            cur, unit = self._currency_unit_from_meta(metadata)
            ind.total_debt = FieldMetadata(
                value=val, source="SEC", source_link=companyfacts_url,
                statement="BS", period=end, date_retrieved=asof,
                reliability=0.85, currency=cur, unit=unit, **metadata
            )

        # Equity
        equity = latest_instant(['StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'])
        if equity:
            val, end, metadata = equity
            cur, unit = self._currency_unit_from_meta(metadata)
            ind.total_equity = FieldMetadata(
                value=val, source="SEC", source_link=companyfacts_url,
                statement="BS", period=end, date_retrieved=asof,
                reliability=0.85, currency=cur, unit=unit, **metadata
            )

        # Revenue
        rev = ltm_from_tags(['Revenues', 'SalesRevenueNet', 'RevenueFromContractWithCustomerExcludingAssessedTax'])
        if rev:
            v, end, method, conf, metadata = rev
            cur, unit = self._currency_unit_from_meta(metadata)
            ind.ltm_revenue = FieldMetadata(
                value=v, source="SEC", source_link=companyfacts_url,
                statement=f"IS({method})", period=end, date_retrieved=asof,
                reliability=0.8*conf, currency=cur, unit=unit, **metadata
            )

        # EBIT
        ebit_pick = ltm_from_tags(['OperatingIncomeLoss'])
        if ebit_pick:
            v, end, method, conf, metadata = ebit_pick
            cur, unit = self._currency_unit_from_meta(metadata)
            ind.ltm_ebit = FieldMetadata(
                value=v, source="SEC", source_link=companyfacts_url,
                statement=f"IS({method})", period=end, date_retrieved=asof,
                reliability=0.85*conf, currency=cur, unit=unit, **metadata
            )

        # D&A
        da_pick = ltm_from_tags(['DepreciationDepletionAndAmortization'])
        if da_pick:
            v, end, method, conf, metadata = da_pick
            cur, unit = self._currency_unit_from_meta(metadata)
            ind.depreciation_amortization = FieldMetadata(
                value=v, source="SEC", source_link=companyfacts_url,
                statement=f"IS({method})", period=end, date_retrieved=asof,
                reliability=0.85*conf, currency=cur, unit=unit, **metadata
            )

        # EBITDA (EBIT + D&A)
        if ebit_pick and da_pick:
            v_e, end_e, method_e, conf_e, meta_e = ebit_pick
            v_da, end_da, _, conf_da, meta_da = da_pick
            conf_min = min(conf_e, conf_da)
            cur, unit = self._currency_unit_from_meta(meta_e)
            ind.ltm_ebitda = FieldMetadata(
                value=v_e + v_da, source="SEC_CALC", source_link=companyfacts_url,
                statement=f"IS({method_e}+DA)", period=max(end_e, end_da),
                date_retrieved=asof, reliability=0.9*conf_min, currency=cur, unit=unit,
                xbrl_tag=f"{meta_e.get('xbrl_tag')}+{meta_da.get('xbrl_tag')}",
                filing_url=meta_e.get('filing_url')
            )

        # OCF / CapEx
        ocf = ltm_from_tags(['NetCashProvidedByUsedInOperatingActivities'])
        if ocf:
            v, end, method, conf, metadata = ocf
            cur, unit = self._currency_unit_from_meta(metadata)
            ind.operating_cash_flow = FieldMetadata(
                value=v, source="SEC", source_link=companyfacts_url,
                statement=f"CF({method})", period=end, date_retrieved=asof,
                reliability=0.85*conf, currency=cur, unit=unit, **metadata
            )

        capex = ltm_from_tags(['PaymentsToAcquirePropertyPlantAndEquipment'])
        if capex:
            v, end, method, conf, metadata = capex
            cur, unit = self._currency_unit_from_meta(metadata)
            ind.capex = FieldMetadata(
                value=abs(v), source="SEC", source_link=companyfacts_url,
                statement=f"CF({method})", period=end, date_retrieved=asof,
                reliability=0.85*conf, currency=cur, unit=unit, **metadata
            )

        # Interest / Tax
        ie = ltm_from_tags(['InterestExpense', 'InterestAndDebtExpense'])
        if ie:
            v, end, method, conf, metadata = ie
            cur, unit = self._currency_unit_from_meta(metadata)
            ind.interest_expense = FieldMetadata(
                value=v, source="SEC", source_link=companyfacts_url,
                statement=f"IS({method})", period=end, date_retrieved=asof,
                reliability=0.8*conf, currency=cur, unit=unit, **metadata
            )

        dbg(530, f"✓ SEC indicators extracted: {ticker}")
        return ind, []

    def fetch_historical_data(
        self,
        ticker: str,
        cik: Optional[str],
        periods: int = 8
    ) -> Dict[str, List[Tuple[float, str]]]:
        """분기별 과거 데이터 조회 (성장률 계산용)"""
        if not cik:
            ident, _ = self.fetch_company_info(ticker, CollectionAttemptSummary(ticker=ticker))
            if ident:
                cik = ident.cik
            else:
                return {}
        
        asof = today_kst()
        facts = self._get_company_facts_cached(cik, asof)
        if not facts:
            return {}
        
        historical = {}
        
        def get_quarterly_series(tag_list: List[str]) -> List[Tuple[float, str]]:
            for tag in tag_list:
                units = self._get_facts_units(facts, tag)
                for unit, vals in units.items():
                    quarterlies = [
                        v for v in vals
                        if isinstance(v.get('frame'), str) and 'Q' in v['frame'] and 'YTD' not in v['frame']
                    ]
                    
                    if not quarterlies:
                        continue
                    
                    quarterlies.sort(key=lambda x: x.get('end', ''), reverse=True)
                    
                    result = []
                    for v in quarterlies[:periods]:
                        val = validate_number(v.get('val'))
                        period = v.get('end')
                        if val is not None and period:
                            result.append((val, period))
                    
                    if result:
                        return result
            
            return []
        
        rev_series = get_quarterly_series(['Revenues', 'SalesRevenueNet'])
        if rev_series:
            historical['revenue'] = rev_series
        
        ebit_series = get_quarterly_series(['OperatingIncomeLoss'])
        if ebit_series:
            historical['ebit'] = ebit_series
        
        ni_series = get_quarterly_series(['NetIncomeLoss'])
        if ni_series:
            historical['net_income'] = ni_series
        
        dbg(540, f"Historical data: {list(historical.keys())}")
        return historical


# ========================================================================
# Yahoo Finance Source (Extended 40+ Fields)
# ========================================================================
class YahooFinanceSource:
    """Yahoo Finance 데이터 소스 (yfinance + Stooq fallback)"""
    
    @staticmethod
    def _price_from_stooq(ticker: str) -> Optional[float]:
        """Stooq fallback for price"""
        try:
            from .utils import create_session
            sess = create_session()
            sym = ticker.lower()
            if '.' not in sym:
                sym = f"{sym}.us"
            url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
            r = sess.get(url, timeout=10)
            if r.ok and 'Date,Open,High,Low,Close,Volume' in r.text:
                lines = r.text.strip().splitlines()
                if len(lines) > 1:
                    close = validate_number(lines[-1].split(',')[4])
                    return close
        except Exception as e:
            dbg(550, f"Stooq fallback failed: {e}")
        return None

    @staticmethod
    def fetch_indicators(
        ticker: str,
        attempt_log: CollectionAttemptSummary
    ) -> Tuple[Optional[IndicatorDataWithMeta], List[str]]:
        """Yahoo Finance 지표 수집 (40+ 필드)"""
        if not yf:
            return None, ["yfinance not installed"]
        
        dbg(560, f"Yahoo fetch_indicators: {ticker}")
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info if hasattr(stock, "info") else {}
            
            indicator = IndicatorDataWithMeta()
            source_link = f"https://finance.yahoo.com/quote/{ticker}"
            asof = today_kst()

            # Price (다중 fallback)
            price_source_name = "Yahoo"
            price = validate_number(info.get('currentPrice') or info.get('regularMarketPrice'))
            
            if price is None:
                try:
                    hist = stock.history(period="5d")
                    if pd and not hist.empty and 'Close' in hist.columns:
                        price = float(hist['Close'].iloc[-1])
                except Exception:
                    pass
            
            if price is None:
                stq = YahooFinanceSource._price_from_stooq(ticker)
                if stq:
                    price = stq
                    price_source_name = "Stooq"
            
            if price:
                cur = (info.get('currency') or "USD").upper()
                indicator.share_price = FieldMetadata(
                    value=price, source=price_source_name, source_link=source_link,
                    date_retrieved=asof, reliability=0.85, unit=f"{cur}/share", currency=cur
                )

            # Shares Outstanding
            shares = validate_number(info.get('sharesOutstanding'))
            if shares:
                indicator.shares_outstanding = FieldMetadata(
                    value=shares, source="Yahoo", source_link=source_link,
                    unit="shares", date_retrieved=asof, reliability=0.85
                )

            # Market Cap
            market_cap = validate_number(info.get('marketCap'))
            if market_cap:
                indicator.market_cap = FieldMetadata(
                    value=market_cap, source="Yahoo_Direct", source_link=source_link,
                    date_retrieved=asof, reliability=0.90, currency="USD", unit="USD"
                )

            # LTM Metrics
            ltm_rev = validate_number(info.get('totalRevenue'))
            if ltm_rev:
                indicator.ltm_revenue = FieldMetadata(
                    value=ltm_rev, source="Yahoo", source_link=source_link,
                    statement="IS(TTM)", period=asof, date_retrieved=asof,
                    reliability=0.80, currency="USD", unit="USD"
                )

            ltm_ebitda = validate_number(info.get('ebitda'))
            if ltm_ebitda:
                indicator.ltm_ebitda = FieldMetadata(
                    value=ltm_ebitda, source="Yahoo", source_link=source_link,
                    statement="IS(TTM)", period=asof, date_retrieved=asof,
                    reliability=0.80, currency="USD", unit="USD"
                )

            # Extended Yahoo Fields (40+ fields)
            for yf_key, our_key in YAHOO_FIELD_MAPPING.items():
                val = info.get(yf_key)
                if val is not None:
                    validated = validate_number(val) if isinstance(val, (int, float)) else val
                    if validated is not None:
                        # Unit/Currency 판단
                        if 'price' in our_key.lower() or 'rate' in our_key.lower():
                            unit, currency = "USD", "USD"
                        elif 'volume' in our_key:
                            unit, currency = "shares", None
                        elif 'ratio' in our_key or 'margin' in our_key:
                            unit, currency = None, None
                        else:
                            unit = "USD" if isinstance(validated, (int, float)) else None
                            currency = "USD" if isinstance(validated, (int, float)) else None
                        
                        setattr(indicator, our_key, FieldMetadata(
                            value=validated,
                            source="Yahoo",
                            source_link=source_link,
                            date_retrieved=asof,
                            reliability=0.75,
                            unit=unit,
                            currency=currency
                        ))

            # Sector & Industry
            if info.get('sector'):
                indicator.sector = FieldMetadata(
                    value=info.get('sector'), source="Yahoo", source_link=source_link,
                    date_retrieved=asof, reliability=0.7
                )
            if info.get('industry'):
                indicator.industry = FieldMetadata(
                    value=info.get('industry'), source="Yahoo", source_link=source_link,
                    date_retrieved=asof, reliability=0.7
                )

            dbg(570, f"✓ Yahoo collected {ticker}: {len([k for k in YAHOO_FIELD_MAPPING.values() if getattr(indicator, k, None)])} extended fields")
            return indicator, []
            
        except Exception as e:
            dbg(571, f"✗ Yahoo error: {e}")
            return None, [f"Yahoo error: {e}"]


# ========================================================================
# FMP (Financial Modeling Prep)
# ========================================================================
class FinancialModelingPrepSource:
    """FMP API 래퍼 (배치 페칭)"""
    name = "FMP"
    BASE = "https://financialmodelingprep.com/api/v3"
    
    def __init__(self, api_key: Optional[str], session):
        self.api_key = api_key
        self.session = session
    
    def _get(self, path: str, params: Dict[str, Any]) -> Optional[dict]:
        if not self.api_key or not self.session:
            return None
        params = dict(params or {})
        params["apikey"] = self.api_key
        try:
            r = self.session.get(f"{self.BASE}{path}", params=params, timeout=12)
            if r.ok:
                return r.json()
        except Exception:
            return None
        return None
    
    def fetch_all_available_fields(self, ticker: str) -> Dict[str, FieldMetadata]:
        """배치 페칭 (한 번에 여러 필드)"""
        t = normalize_ticker(ticker)
        asof = today_kst()
        fields = {}
        
        try:
            # Income Statement (quarterly, last 8)
            is_data = self._get(f"/income-statement/{t}", {"period": "quarter", "limit": 8})
            if is_data and isinstance(is_data, list) and len(is_data) > 0:
                snap = json.dumps(is_data[:4], sort_keys=True).encode("utf-8")
                sha = hashlib.sha256(snap).hexdigest()
                
                def sum_last4(key: str) -> Optional[float]:
                    vals = [validate_number(row.get(key)) for row in is_data[:4]]
                    vals = [v for v in vals if v is not None]
                    return float(sum(vals)) if vals else None
                
                # EBIT
                ebit_val = sum_last4("operatingIncome")
                if ebit_val:
                    fields['ltm_ebit'] = FieldMetadata(
                        value=ebit_val, source=self.name,
                        source_link=f"{self.BASE}/income-statement/{t}",
                        statement="IS(TTM)", date_retrieved=asof,
                        reliability=0.7, currency="USD", unit="USD",
                        raw_json_path="income-statement[:4].operatingIncome",
                        snapshot_sha256=sha
                    )
                
                # Interest Expense
                ie_val = sum_last4("interestExpense")
                if ie_val:
                    fields['interest_expense'] = FieldMetadata(
                        value=ie_val, source=self.name,
                        source_link=f"{self.BASE}/income-statement/{t}",
                        statement="IS(TTM)", date_retrieved=asof,
                        reliability=0.65, currency="USD", unit="USD",
                        raw_json_path="income-statement[:4].interestExpense",
                        snapshot_sha256=sha
                    )
        
        except Exception as e:
            dbg(600, f"FMP batch fetch error: {e}")
        
        return fields


# ========================================================================
# AlphaVantage
# ========================================================================
class AlphaVantageSource:
    """AlphaVantage API 래퍼"""
    name = "AlphaVantage"
    BASE = "https://www.alphavantage.co/query"
    
    def __init__(self, api_key: Optional[str], session):
        self.api_key = api_key
        self.session = session
    
    def fetch_field(self, ticker: str, field_name: str) -> Optional[FieldMetadata]:
        """단일 필드 조회"""
        if not self.api_key or not self.session:
            return None
        
        t = normalize_ticker(ticker)
        asof = today_kst()
        
        try:
            if field_name == "share_price":
                r = self.session.get(self.BASE, params={
                    "function": "TIME_SERIES_DAILY_ADJUSTED",
                    "symbol": t,
                    "apikey": self.api_key
                }, timeout=12)
                if r.ok:
                    js = r.json()
                    ts = js.get("Time Series (Daily)", {})
                    if ts:
                        last_day = sorted(ts.keys())[-1]
                        close = validate_number(ts[last_day].get("4. close"))
                        if close:
                            return FieldMetadata(
                                value=close, source=self.name,
                                source_link=f"{self.BASE}?function=TIME_SERIES_DAILY_ADJUSTED&symbol={t}",
                                statement="Market", date_retrieved=asof,
                                reliability=0.7, unit="USD", currency="USD"
                            )
        except Exception:
            pass
        
        return None


# ========================================================================
# Polygon.io
# ========================================================================
class PolygonIOSource:
    """Polygon.io API 래퍼"""
    name = "Polygon"
    BASE = "https://api.polygon.io"
    
    def __init__(self, api_key: Optional[str], session):
        self.api_key = api_key
        self.session = session
    
    def fetch_field(self, ticker: str, field_name: str) -> Optional[FieldMetadata]:
        """단일 필드 조회"""
        if not self.api_key or not self.session:
            return None
        
        t = normalize_ticker(ticker)
        asof = today_kst()
        
        try:
            if field_name == "share_price":
                r = self.session.get(
                    f"{self.BASE}/v2/aggs/ticker/{t}/prev",
                    params={"adjusted": "true", "apiKey": self.api_key},
                    timeout=12
                )
                if r.ok:
                    res = r.json().get("results", [])
                    if res:
                        close = validate_number(res[0].get("c"))
                        if close:
                            return FieldMetadata(
                                value=close, source=self.name,
                                source_link=f"{self.BASE}/v2/aggs/ticker/{t}/prev",
                                statement="Market", date_retrieved=asof,
                                reliability=0.65, unit="USD", currency="USD"
                            )
        except Exception:
            pass
        
        return None


# ========================================================================
# NewsAPI (Sentiment Analysis)
# ========================================================================
POSITIVE_WORDS = {
    'beat', 'beats', 'strong', 'growth', 'surge', 'gain', 'rally',
    'bullish', 'positive', 'upgrade', 'outperform', 'buy', 'exceeded'
}

NEGATIVE_WORDS = {
    'miss', 'misses', 'weak', 'decline', 'fall', 'drop',
    'bearish', 'negative', 'downgrade', 'underperform', 'sell', 'lawsuit'
}


class NewsAPISource:
    """NewsAPI 래퍼 (감성 분석)"""
    
    def __init__(self, api_key: Optional[str], session):
        self.api_key = api_key
        self.session = session
        self.base_url = "https://newsapi.org/v2"

    def fetch_recent_news_sentiment(
        self,
        ticker: str,
        company_name: Optional[str] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """최근 뉴스 감성 분석"""
        if not self.api_key or not self.session:
            return {}
        
        try:
            from_date = (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%d')
            query = company_name if company_name else ticker
            
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'publishedAt',
                'language': 'en',
                'apiKey': self.api_key
            }
            
            response = self.session.get(f"{self.base_url}/everything", params=params, timeout=10)
            
            if not response.ok:
                return {}
            
            articles = response.json().get('articles', [])
            if not articles:
                return {'total_articles': 0}
            
            # 감성 분석
            positive_count = negative_count = neutral_count = 0
            sentiment_scores = []
            headlines_sample = []
            topics = Counter()
            
            for article in articles:
                title = (article.get('title', '') or '').lower()
                description = (article.get('description', '') or '').lower()
                full_text = f"{title} {description}"
                
                pos_matches = sum(1 for word in POSITIVE_WORDS if word in full_text)
                neg_matches = sum(1 for word in NEGATIVE_WORDS if word in full_text)
                
                if pos_matches > neg_matches:
                    sentiment = 'positive'
                    positive_count += 1
                    score = min(1.0, pos_matches / 5.0)
                elif neg_matches > pos_matches:
                    sentiment = 'negative'
                    negative_count += 1
                    score = -min(1.0, neg_matches / 5.0)
                else:
                    sentiment = 'neutral'
                    neutral_count += 1
                    score = 0.0
                
                sentiment_scores.append(score)
                
                # 토픽 추출
                words = re.findall(r'\b[a-z]{4,}\b', full_text)
                for word in words:
                    if word not in {'said', 'will', 'also', 'company'}:
                        topics[word] += 1
                
                if len(headlines_sample) < 10:
                    headlines_sample.append({
                        'date': article.get('publishedAt', '')[:10],
                        'title': article.get('title', ''),
                        'sentiment': sentiment
                    })
            
            overall_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
            
            return {
                'period': f'last_{days}_days',
                'total_articles': len(articles),
                'sentiment_score': round(overall_sentiment, 3),
                'sentiment_breakdown': {
                    'positive': positive_count,
                    'neutral': neutral_count,
                    'negative': negative_count
                },
                'key_topics': [word for word, _ in topics.most_common(10)],
                'headline_sample': headlines_sample
            }
            
        except Exception as e:
            dbg(700, f"News sentiment error: {e}")
            return {}

# ========================================================================
# Finnhub Source (Yahoo 대체)
# ========================================================================
class FinnhubSource:
    """Finnhub API 래퍼 (Yahoo 대체용)"""
    name = "Finnhub"
    BASE = "https://finnhub.io/api/v1"
    
    def __init__(self, api_key: Optional[str], session):
        self.api_key = api_key
        self.session = session
    
    def _get(self, endpoint: str, params: Dict[str, Any] = None) -> Optional[dict]:
        """API 호출"""
        if not self.api_key or not self.session:
            return None
        
        params = params or {}
        params['token'] = self.api_key
        
        try:
            url = f"{self.BASE}{endpoint}"
            r = self.session.get(url, params=params, timeout=10)
            if r.ok:
                return r.json()
            else:
                dbg(801, f"Finnhub API error: {r.status_code}")
        except Exception as e:
            dbg(802, f"Finnhub request failed: {e}")
        
        return None
    
    def fetch_indicators(
        self,
        ticker: str,
        attempt_log: CollectionAttemptSummary
    ) -> Tuple[Optional[IndicatorDataWithMeta], List[str]]:
        """재무 지표 수집"""
        if not self.api_key:
            return None, ["Finnhub API key not configured"]
        
        dbg(810, f"Finnhub fetch_indicators: {ticker}")
        
        try:
            indicator = IndicatorDataWithMeta()
            asof = today_kst()
            source_link = f"https://finnhub.io/quote/{ticker}"
            
            # 1. Quote (실시간 가격)
            quote = self._get("/quote", {"symbol": ticker})
            if quote and quote.get('c'):  # current price
                price = validate_number(quote.get('c'))
                if price:
                    indicator.share_price = FieldMetadata(
                        value=price,
                        source=self.name,
                        source_link=source_link,
                        date_retrieved=asof,
                        reliability=0.85,
                        unit="USD",
                        currency="USD"
                    )
                    dbg(811, f"Finnhub price: ${price}")
            
            # 2. Profile (회사 정보)
            profile = self._get("/stock/profile2", {"symbol": ticker})
            if profile:
                # Market Cap
                mcap = validate_number(profile.get('marketCapitalization'))
                if mcap:
                    # Finnhub는 million 단위로 반환
                    indicator.market_cap = FieldMetadata(
                        value=mcap * 1_000_000,
                        source=self.name,
                        source_link=source_link,
                        date_retrieved=asof,
                        reliability=0.85,
                        currency="USD",
                        unit="USD"
                    )
                
                # Shares Outstanding
                shares = validate_number(profile.get('shareOutstanding'))
                if shares:
                    # Finnhub는 million 단위로 반환
                    indicator.shares_outstanding = FieldMetadata(
                        value=shares * 1_000_000,
                        source=self.name,
                        source_link=source_link,
                        date_retrieved=asof,
                        reliability=0.85,
                        unit="shares"
                    )
            
            # 3. Basic Financials (재무 지표)
            financials = self._get("/stock/metric", {"symbol": ticker, "metric": "all"})
            if financials and financials.get('metric'):
                metrics = financials['metric']
                
                # Beta
                beta = validate_number(metrics.get('beta'))
                if beta:
                    indicator.beta = FieldMetadata(
                        value=beta,
                        source=self.name,
                        source_link=source_link,
                        date_retrieved=asof,
                        reliability=0.80
                    )
                
                # PE Ratio
                pe = validate_number(metrics.get('peBasicExclExtraTTM'))
                if pe:
                    indicator.trailing_pe = FieldMetadata(
                        value=pe,
                        source=self.name,
                        source_link=source_link,
                        date_retrieved=asof,
                        reliability=0.80
                    )
                
                # PB Ratio
                pb = validate_number(metrics.get('pbQuarterly'))
                if pb:
                    indicator.price_to_book = FieldMetadata(
                        value=pb,
                        source=self.name,
                        source_link=source_link,
                        date_retrieved=asof,
                        reliability=0.80
                    )
                
                # Dividend Yield
                div_yield = validate_number(metrics.get('dividendYieldIndicatedAnnual'))
                if div_yield:
                    indicator.dividend_yield = FieldMetadata(
                        value=div_yield,
                        source=self.name,
                        source_link=source_link,
                        date_retrieved=asof,
                        reliability=0.80
                    )
                
                # 52-week High/Low
                high_52w = validate_number(metrics.get('52WeekHigh'))
                if high_52w:
                    indicator.fifty_two_week_high = FieldMetadata(
                        value=high_52w,
                        source=self.name,
                        source_link=source_link,
                        date_retrieved=asof,
                        reliability=0.85,
                        currency="USD"
                    )
                
                low_52w = validate_number(metrics.get('52WeekLow'))
                if low_52w:
                    indicator.fifty_two_week_low = FieldMetadata(
                        value=low_52w,
                        source=self.name,
                        source_link=source_link,
                        date_retrieved=asof,
                        reliability=0.85,
                        currency="USD"
                    )
            
            dbg(820, f"✓ Finnhub collected {ticker}")
            return indicator, []
            
        except Exception as e:
            dbg(821, f"✗ Finnhub error: {e}")
            return None, [f"Finnhub error: {e}"]
    
    def fetch_field(self, ticker: str, field_name: str, context=None) -> Optional[FieldMetadata]:
        """단일 필드 조회 (Cascade용)"""
        if not self.api_key:
            return None
        
        asof = today_kst()
        source_link = f"https://finnhub.io/quote/{ticker}"
        
        try:
            # Share Price
            if field_name == "share_price":
                quote = self._get("/quote", {"symbol": ticker})
                if quote and quote.get('c'):
                    price = validate_number(quote.get('c'))
                    if price:
                        return FieldMetadata(
                            value=price,
                            source=self.name,
                            source_link=source_link,
                            date_retrieved=asof,
                            reliability=0.85,
                            unit="USD",
                            currency="USD"
                        )
            
            # Market Cap
            elif field_name == "market_cap":
                profile = self._get("/stock/profile2", {"symbol": ticker})
                if profile:
                    mcap = validate_number(profile.get('marketCapitalization'))
                    if mcap:
                        return FieldMetadata(
                            value=mcap * 1_000_000,
                            source=self.name,
                            source_link=source_link,
                            date_retrieved=asof,
                            reliability=0.85,
                            currency="USD",
                            unit="USD"
                        )
            
            # Shares Outstanding
            elif field_name == "shares_outstanding":
                profile = self._get("/stock/profile2", {"symbol": ticker})
                if profile:
                    shares = validate_number(profile.get('shareOutstanding'))
                    if shares:
                        return FieldMetadata(
                            value=shares * 1_000_000,
                            source=self.name,
                            source_link=source_link,
                            date_retrieved=asof,
                            reliability=0.85,
                            unit="shares"
                        )
        
        except Exception as e:
            dbg(822, f"Finnhub field fetch error: {e}")
        
        return None


# ========================================================================
# Macro Finance Source (DCF Parameters)
# ========================================================================
class MacroFinanceSource:
    """거시경제 데이터 소스 (Risk-free rate, MRP)"""
    
    def __init__(self, session):
        self.session = session

    @staticmethod
    def _country_mrp(country_code: str) -> Tuple[float, str]:
        """국가별 MRP (Damodaran)"""
        table = {
            "US": (0.0475, "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.html"),
            "KR": (0.0640, "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.html"),
            "JP": (0.0550, "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.html"),
        }
        return table.get(country_code, (0.0500, "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.html"))

    def populate_dcf_params(self, ident, data, attempt_log):
        """DCF 파라미터 자동 채우기"""
        country = (ident.country or "US") if ident else "US"
        if "korea" in country.lower():
            country = "KR"
        
        asof = today_kst()

        # Risk-free rate
        if not (data.risk_free_rate and data.risk_free_rate.value):
            rf_val = 0.035
            if yf and country == "US":
                try:
                    tnx = yf.Ticker("^TNX")
                    hist = tnx.history(period="5d")
                    if not hist.empty:
                        last = float(hist["Close"].iloc[-1])
                        rf_val = last / 100.0 if last > 1.0 else last
                except Exception:
                    pass
            
            data.risk_free_rate = FieldMetadata(
                value=rf_val,
                source="Yahoo(^TNX)" if country == "US" else "Heuristic",
                date_retrieved=asof,
                reliability=0.70
            )

        # Market Risk Premium
        if not (data.market_risk_premium and data.market_risk_premium.value):
            mrp_val, mrp_link = self._country_mrp(country)
            data.market_risk_premium = FieldMetadata(
                value=mrp_val,
                source="Damodaran",
                source_link=mrp_link,
                date_retrieved=asof,
                reliability=0.65
            )


# ========================================================================
# Estimators (Industry Average, Historical Trend)
# ========================================================================
class IndustryAverageEstimator:
    """산업 평균 기반 추정"""
    name = "Estimated(SectorAvg)"
    
    SECTOR_DEFAULTS = {
        "Technology": {"etr": 0.20, "cod": 0.06},
        "Utilities": {"etr": 0.24, "cod": 0.05},
        "default": {"etr": 0.21, "cod": 0.06},
    }
    
    def fetch_field(self, ticker: str, field_name: str, context) -> Optional[FieldMetadata]:
        """산업 평균값 반환"""
        asof = today_kst()
        sector = context.sector.value if (context and context.sector) else None
        prof = self.SECTOR_DEFAULTS.get(sector, self.SECTOR_DEFAULTS["default"])
        
        if field_name == "effective_tax_rate":
            return FieldMetadata(
                value=prof["etr"],
                source=self.name,
                statement="IS(estimate)",
                date_retrieved=asof,
                reliability=0.35
            )
        
        return None


class HistoricalTrendEstimator:
    """과거 데이터 기반 추정"""
    name = "Estimated(Historical)"
    
    def fetch_field(self, ticker: str, field_name: str, context) -> Optional[FieldMetadata]:
        """과거 추세 기반 추정"""
        asof = today_kst()
        
        if field_name == "ltm_ebit":
            ebitda = validate_number(context.ltm_ebitda.value) if context.ltm_ebitda else None
            da = validate_number(context.depreciation_amortization.value) if context.depreciation_amortization else None
            
            if ebitda and da:
                return FieldMetadata(
                    value=float(ebitda - da),
                    source=self.name,
                    statement="IS(estimated)",
                    date_retrieved=asof,
                    reliability=0.40,
                    currency="USD",
                    unit="USD"
                )
        
        return None


# ========================================================================
# Base Cascade Source
# ========================================================================
class BaseCascadeSource:
    """Cascade 소스 베이스 클래스"""
    name = "Base"
    
    def __init__(self, session=None):
        self.session = session
    
    def fetch_field(self, ticker: str, field_name: str, context=None) -> Optional[FieldMetadata]:
        """필드 조회 (하위 클래스에서 구현)"""
        return None