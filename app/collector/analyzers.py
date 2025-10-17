# app/collector/analyzers.py
"""
분석 모듈 통합
- PeerAnalyzer (동종업계 비교)
- TechnicalIndicatorsCalculator (기술적 지표)
- AnalystActivityTracker (애널리스트 활동)
- GrowthRateCalculator (성장률 계산)
"""
from typing import Dict, List, Optional, Any, Tuple
from .utils import dbg, validate_number, today_kst, normalize_cik
from .models import TechnicalIndicators, AnalystActivity, FieldMetadata
from .cache import CIKMapper

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import pandas as pd
except ImportError:
    pd = None


# ========================================================================
# Peer Analyzer (SIC 기반 동종업계 분석)
# ========================================================================
class PeerAnalyzer:
    """동종업계 분석기 (SIC 코드 기반)"""
    
    def __init__(self, session, cik_mapper: CIKMapper):
        self.session = session
        self.cik_mapper = cik_mapper
        self._company_tickers_cache: Optional[Dict] = None

    def _get_all_company_tickers(self) -> Dict:
        """SEC company tickers 조회 (캐시)"""
        if self._company_tickers_cache is not None:
            return self._company_tickers_cache
        
        try:
            url = "https://www.sec.gov/files/company_tickers.json"
            r = self.session.get(url, timeout=15)
            if r.ok:
                self._company_tickers_cache = r.json()
                dbg(600, f"Loaded {len(self._company_tickers_cache)} company tickers")
                return self._company_tickers_cache
        except Exception as e:
            dbg(601, f"Failed to fetch company tickers: {e}")
        
        return {}

    def get_peer_companies(
        self,
        ticker: str,
        target_sic: Optional[str],
        target_market_cap: Optional[float],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """동종업계 회사 조회 (SIC + 시가총액 유사도)"""
        if not target_sic:
            dbg(602, f"No SIC code for {ticker}")
            return []
        
        all_companies = self._get_all_company_tickers()
        if not all_companies:
            return []
        
        # SIC 코드 필터링 (4자리)
        target_sic_4digit = str(target_sic)[:4]
        peers = []
        
        for entry in all_companies.values():
            try:
                if entry.get('ticker', '').upper() == ticker.upper():
                    continue
                
                cik = normalize_cik(entry.get('cik_str', ''))
                company_ticker = entry.get('ticker', '').upper()
                company_name = entry.get('title', '')
                
                peers.append({
                    'ticker': company_ticker,
                    'name': company_name,
                    'cik': cik
                })
                
                if len(peers) >= limit * 3:
                    break
                    
            except Exception as e:
                dbg(603, f"Error processing peer: {e}")
                continue
        
        # 시가총액 기준 필터링
        if yf and target_market_cap:
            scored_peers = []
            for peer in peers[:limit * 2]:
                try:
                    stock = yf.Ticker(peer['ticker'])
                    info = stock.info if hasattr(stock, 'info') else {}
                    mcap = validate_number(info.get('marketCap'))
                    
                    if mcap:
                        peer['market_cap'] = mcap
                        mcap_ratio = mcap / target_market_cap if target_market_cap > 0 else 0
                        if 0.5 <= mcap_ratio <= 2.0:
                            peer['similarity_score'] = 1.0 - abs(1.0 - mcap_ratio)
                            scored_peers.append(peer)
                            
                except Exception as e:
                    dbg(604, f"Failed to fetch market cap for {peer['ticker']}: {e}")
                    continue
            
            scored_peers.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            return scored_peers[:limit]
        
        return peers[:limit]

    def calculate_peer_metrics(self, peers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """동종업계 중앙값 계산"""
        if not peers or not yf:
            return {}
        
        metrics = {
            'pe_ratios': [],
            'ps_ratios': [],
            'pb_ratios': [],
            'ev_ebitda': [],
            'gross_margins': [],
            'revenue_growth': [],
            'market_caps': []
        }
        
        for peer in peers:
            try:
                stock = yf.Ticker(peer['ticker'])
                info = stock.info if hasattr(stock, 'info') else {}
                
                pe = validate_number(info.get('trailingPE'))
                if pe: metrics['pe_ratios'].append(pe)
                
                ps = validate_number(info.get('priceToSalesTrailing12Months'))
                if ps: metrics['ps_ratios'].append(ps)
                
                pb = validate_number(info.get('priceToBook'))
                if pb: metrics['pb_ratios'].append(pb)
                
                ev_ebitda = validate_number(info.get('enterpriseToEbitda'))
                if ev_ebitda: metrics['ev_ebitda'].append(ev_ebitda)
                
                gm = validate_number(info.get('grossMargins'))
                if gm: metrics['gross_margins'].append(gm)
                
                rg = validate_number(info.get('revenueGrowth'))
                if rg: metrics['revenue_growth'].append(rg)
                
                mcap = validate_number(info.get('marketCap'))
                if mcap: metrics['market_caps'].append(mcap)
                
            except Exception as e:
                dbg(605, f"Failed to fetch metrics for {peer['ticker']}: {e}")
                continue
        
        # 중앙값 계산
        peer_medians = {}
        for key, values in metrics.items():
            if values:
                median_val = sorted(values)[len(values)//2]
                field_name = key.replace('_ratios', '_ratio').replace('_margins', '_margin').replace('_growth', '_growth_rate')
                peer_medians[field_name] = float(median_val)
        
        return peer_medians


# ========================================================================
# Technical Indicators Calculator (기술적 지표)
# ========================================================================
class TechnicalIndicatorsCalculator:
    """기술적 지표 계산기"""
    
    @staticmethod
    def calculate_rsi(prices: 'pd.Series', period: int = 14) -> Optional[float]:
        """RSI 계산"""
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
        except Exception:
            return None
    
    @staticmethod
    def calculate_atr(
        high: 'pd.Series',
        low: 'pd.Series',
        close: 'pd.Series',
        period: int = 14
    ) -> Optional[float]:
        """ATR 계산"""
        try:
            high_low = high - low
            high_close = abs(high - close.shift())
            low_close = abs(low - close.shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None
        except Exception:
            return None
    
    @staticmethod
    def calculate_all(ticker: str) -> TechnicalIndicators:
        """모든 기술적 지표 계산"""
        indicators = TechnicalIndicators()
        
        if not yf or not pd:
            dbg(610, "yfinance/pandas not available")
            return indicators
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='200d')
            
            if hist.empty or len(hist) < 50:
                dbg(611, f"{ticker}: Insufficient historical data")
                return indicators
            
            close = hist['Close']
            high = hist['High']
            low = hist['Low']
            
            # RSI
            indicators.rsi_14 = TechnicalIndicatorsCalculator.calculate_rsi(close, 14)
            
            # ATR
            indicators.atr_14 = TechnicalIndicatorsCalculator.calculate_atr(high, low, close, 14)
            
            # 이동평균 대비 위치
            current_price = float(close.iloc[-1])
            
            if len(close) >= 50:
                ma50 = close.rolling(window=50).mean().iloc[-1]
                if not pd.isna(ma50) and ma50 > 0:
                    indicators.price_vs_ma50_pct = ((current_price / ma50) - 1) * 100
            
            if len(close) >= 200:
                ma200 = close.rolling(window=200).mean().iloc[-1]
                if not pd.isna(ma200) and ma200 > 0:
                    indicators.price_vs_ma200_pct = ((current_price / ma200) - 1) * 100
            
            # MACD 신호
            if len(close) >= 26:
                ema12 = close.ewm(span=12).mean()
                ema26 = close.ewm(span=26).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9).mean()
                
                if not pd.isna(macd.iloc[-1]) and not pd.isna(signal.iloc[-1]):
                    if macd.iloc[-1] > signal.iloc[-1]:
                        indicators.macd_signal = 'bullish'
                    elif macd.iloc[-1] < signal.iloc[-1]:
                        indicators.macd_signal = 'bearish'
                    else:
                        indicators.macd_signal = 'neutral'
            
            # Bollinger Bands 위치
            if len(close) >= 20:
                ma20 = close.rolling(window=20).mean()
                std20 = close.rolling(window=20).std()
                upper_band = ma20 + (2 * std20)
                lower_band = ma20 - (2 * std20)
                
                if not pd.isna(upper_band.iloc[-1]) and not pd.isna(lower_band.iloc[-1]):
                    band_width = upper_band.iloc[-1] - lower_band.iloc[-1]
                    if band_width > 0:
                        position = (current_price - lower_band.iloc[-1]) / band_width
                        indicators.bollinger_position = max(0.0, min(1.0, float(position)))
            
            dbg(612, f"{ticker}: RSI={indicators.rsi_14}, ATR={indicators.atr_14}")
            
        except Exception as e:
            dbg(613, f"Technical indicators error for {ticker}: {e}")
        
        return indicators


# ========================================================================
# Analyst Activity Tracker (애널리스트 활동 추적)
# ========================================================================
class AnalystActivityTracker:
    """애널리스트 활동 추적기 (14일)"""
    
    def __init__(self, session=None):
        self.session = session
    
    def fetch_recent_activity(self, ticker: str, days: int = 14) -> AnalystActivity:
        """최근 애널리스트 활동 수집"""
        activity = AnalystActivity()
        
        if not yf or not pd:
            dbg(620, "yfinance/pandas not available")
            return activity
        
        try:
            stock = yf.Ticker(ticker)
            recommendations = stock.recommendations
            
            if recommendations is not None and not recommendations.empty:
                # 최근 14일 필터링
                cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
                recent = recommendations[recommendations.index >= cutoff_date]
                
                if not recent.empty:
                    # 업그레이드/다운그레이드 카운트
                    if 'To Grade' in recent.columns and 'From Grade' in recent.columns:
                        grade_scores = {
                            'strong buy': 5, 'buy': 4, 'outperform': 4,
                            'hold': 3, 'neutral': 3,
                            'underperform': 2, 'sell': 1, 'strong sell': 0
                        }
                        
                        for idx, row in recent.iterrows():
                            to_grade = str(row.get('To Grade', '')).lower()
                            from_grade = str(row.get('From Grade', '')).lower()
                            firm = str(row.get('Firm', 'Unknown'))
                            
                            to_score = grade_scores.get(to_grade, 3)
                            from_score = grade_scores.get(from_grade, 3)
                            
                            if to_score > from_score:
                                activity.upgrades_count += 1
                                activity.recent_actions.append({
                                    'date': idx.strftime('%Y-%m-%d'),
                                    'firm': firm,
                                    'action': 'upgrade',
                                    'from': from_grade,
                                    'to': to_grade
                                })
                            elif to_score < from_score:
                                activity.downgrades_count += 1
                                activity.recent_actions.append({
                                    'date': idx.strftime('%Y-%m-%d'),
                                    'firm': firm,
                                    'action': 'downgrade',
                                    'from': from_grade,
                                    'to': to_grade
                                })
                            else:
                                activity.new_coverage_count += 1
                                activity.recent_actions.append({
                                    'date': idx.strftime('%Y-%m-%d'),
                                    'firm': firm,
                                    'action': 'initiate',
                                    'rating': to_grade
                                })
                    
                    dbg(621, f"{ticker}: {activity.upgrades_count} upgrades, {activity.downgrades_count} downgrades")
            
            # 목표가 변경 추적
            analyst_info = stock.info
            if analyst_info:
                target_mean = validate_number(analyst_info.get('targetMeanPrice'))
                current_price = validate_number(analyst_info.get('currentPrice'))
                
                if target_mean and current_price:
                    change_pct = ((target_mean / current_price) - 1) * 100
                    activity.avg_target_change_pct = float(change_pct)
                    
                    if change_pct > 5:
                        activity.target_raises_count = activity.upgrades_count
                    elif change_pct < -5:
                        activity.target_lowers_count = activity.downgrades_count
            
        except Exception as e:
            dbg(622, f"Analyst activity error for {ticker}: {e}")
        
        return activity


# ========================================================================
# Growth Rate Calculator (성장률 계산)
# ========================================================================
class GrowthRateCalculator:
    """성장률 계산기"""
    
    @staticmethod
    def calculate_yoy_growth(current: float, previous: float) -> Optional[float]:
        """YoY 성장률 계산"""
        if previous == 0 or previous is None:
            return None
        try:
            growth = ((current - previous) / abs(previous)) * 100.0
            if abs(growth) > 1000:  # 비정상적 값 필터링
                return None
            return float(growth)
        except Exception:
            return None

    @staticmethod
    def calculate_cagr(values: List[Tuple[float, str]], periods: int) -> Optional[float]:
        """CAGR 계산"""
        if not values or len(values) < 2:
            return None
        try:
            start_val = values[-1][0]
            end_val = values[0][0]
            if start_val <= 0 or end_val <= 0:
                return None
            
            years = periods / 4.0
            cagr = ((end_val / start_val) ** (1.0 / years) - 1.0) * 100.0
            
            if abs(cagr) > 200:
                return None
            return float(cagr)
        except Exception:
            return None

    @staticmethod
    def calculate_growth_rates(
        historical: Dict[str, List[Tuple[float, str]]],
        current_data
    ) -> Dict[str, FieldMetadata]:
        """과거 데이터로부터 성장률 계산"""
        asof = today_kst()
        growth_fields = {}
        
        # Revenue Growth
        if 'revenue' in historical and len(historical['revenue']) >= 8:
            rev_data = historical['revenue']
            recent_4q = sum(v[0] for v in rev_data[:4])
            previous_4q = sum(v[0] for v in rev_data[4:8])
            yoy = GrowthRateCalculator.calculate_yoy_growth(recent_4q, previous_4q)
            
            if yoy is not None:
                growth_fields['revenue_growth_rate'] = FieldMetadata(
                    value=yoy,
                    source="SEC_Calculated",
                    statement="IS(Growth)",
                    date_retrieved=asof,
                    reliability=0.80,
                    unit="%"
                )
                dbg(630, f"Revenue YoY growth: {yoy:.2f}%")
        
        # EBITDA Growth (EBIT 근사치)
        if 'ebit' in historical and len(historical['ebit']) >= 8:
            ebit_data = historical['ebit']
            recent_4q = sum(v[0] for v in ebit_data[:4])
            previous_4q = sum(v[0] for v in ebit_data[4:8])
            yoy = GrowthRateCalculator.calculate_yoy_growth(recent_4q, previous_4q)
            
            if yoy is not None:
                growth_fields['ebitda_growth_rate'] = FieldMetadata(
                    value=yoy,
                    source="SEC_Calculated",
                    statement="IS(Growth)",
                    date_retrieved=asof,
                    reliability=0.75,
                    unit="%"
                )
                dbg(631, f"EBITDA YoY growth (approx): {yoy:.2f}%")
        
        # EPS Growth
        if 'net_income' in historical and len(historical['net_income']) >= 8:
            ni_data = historical['net_income']
            shares = validate_number(current_data.shares_outstanding.value) if current_data.shares_outstanding else None
            
            if shares and shares > 0:
                recent_4q_ni = sum(v[0] for v in ni_data[:4])
                previous_4q_ni = sum(v[0] for v in ni_data[4:8])
                
                recent_eps = recent_4q_ni / shares
                previous_eps = previous_4q_ni / shares
                
                yoy = GrowthRateCalculator.calculate_yoy_growth(recent_eps, previous_eps)
                
                if yoy is not None:
                    growth_fields['eps_growth_rate'] = FieldMetadata(
                        value=yoy,
                        source="SEC_Calculated",
                        statement="IS(Growth)",
                        date_retrieved=asof,
                        reliability=0.80,
                        unit="%"
                    )
                    dbg(632, f"EPS YoY growth: {yoy:.2f}%")
        
        return growth_fields