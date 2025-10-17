# app/collector/utils.py
"""
Collector 유틸리티 통합
- 디버깅, 검증, 정규화, 세션 관리
"""
import math
from datetime import datetime
from typing import Any, Optional, Union, Dict

# ======== 디버깅 ========
def dbg(code: int, *msgs: Any) -> None:
    """통합 디버그 로거"""
    try:
        print(f"[dbg.{code}]", *msgs, flush=True)
    except Exception:
        pass

_dbg = dbg  # 하위 호환

# ======== Optional Dependencies ========
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from dateutil import parser as date_parser
except ImportError:
    date_parser = None

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    requests = None
    HTTPAdapter = None
    Retry = None

# ======== 검증 함수 ========
def validate_number(
    value: Any, 
    min_val: Optional[float] = None, 
    max_val: Optional[float] = None
) -> Optional[float]:
    """숫자 검증"""
    try:
        num = float(value)
        if (pd and pd.isna(num)) or math.isnan(num) or math.isinf(num):
            return None
        if min_val is not None and num < min_val:
            return None
        if max_val is not None and num > max_val:
            return None
        return 0.0 if num == 0 else num
    except Exception:
        return None


def validate_date(date_input: Any) -> Optional[str]:
    """날짜 검증 및 정규화"""
    if not date_input:
        return None
    if isinstance(date_input, datetime):
        return date_input.strftime('%Y-%m-%d')
    if isinstance(date_input, (int, float)):
        try:
            return datetime.fromtimestamp(date_input).strftime('%Y-%m-%d')
        except Exception:
            pass
    if date_parser:
        try:
            return date_parser.parse(str(date_input)).strftime('%Y-%m-%d')
        except Exception:
            pass
    s = str(date_input)[:10]
    for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y']:
        try:
            return datetime.strptime(s, fmt).strftime('%Y-%m-%d')
        except Exception:
            continue
    return None


def today_kst() -> str:
    """현재 날짜 (KST)"""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%d")


def freshness_weight(period: Optional[str], half_life_days: int = 180) -> float:
    """신선도 가중치 (exponential decay)"""
    if not period:
        return 0.5
    try:
        data_date = datetime.strptime(period[:10], '%Y-%m-%d')
        days_old = (datetime.utcnow() - data_date).days
        if days_old <= 0:
            return 1.0
        lam = math.log(2) / max(1, half_life_days)
        return float(math.exp(-lam * days_old))
    except Exception:
        return 0.5


# ======== 정규화 함수 ========
def normalize_ticker(ticker: str) -> str:
    """티커 정규화"""
    return str(ticker).strip().upper().replace(".", "-")


def normalize_cik(cik: Union[str, int]) -> str:
    """CIK 정규화 (10자리)"""
    return str(cik).strip().zfill(10)


def is_per_share(unit: Optional[str]) -> bool:
    """주당 단위 여부"""
    if not unit:
        return False
    u = unit.lower()
    return ('/share' in u) or ('/shares' in u) or ('pershare' in u) or u.endswith('/sh')


def convert_per_share(
    value: float, 
    from_unit: str, 
    to_unit: str, 
    shares: Optional[float]
) -> Optional[float]:
    """주당 <-> 총액 변환"""
    if from_unit == to_unit:
        return value
    if is_per_share(from_unit) and not is_per_share(to_unit):
        return value * shares if shares and shares > 0 else None
    if (not is_per_share(from_unit)) and is_per_share(to_unit):
        return value / shares if shares and shares > 0 else None
    return value


# ======== 세션 관리 ========
def create_session() -> 'requests.Session':
    """Retry 포함 HTTP 세션"""
    if requests is None or HTTPAdapter is None or Retry is None:
        raise RuntimeError("Requests/urllib3 missing")
    
    session = requests.Session()
    try:
        retry = Retry(
            total=3, backoff_factor=1, 
            status_forcelist=[429, 500, 502, 503, 504], 
            allowed_methods={"GET", "POST"}
        )
    except TypeError:
        retry = Retry(
            total=3, backoff_factor=1, 
            status_forcelist=[429, 500, 502, 503, 504], 
            method_whitelist={"GET", "POST"}
        )
    
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({'User-Agent': 'EnterpriseCollector/v25'})
    
    dbg(120, "HTTP session created")
    return session


# ======== 환율 정규화 ========
class CurrencyNormalizer:
    """환율 정규화 (USD 기준)"""
    FX_CACHE: Dict[str, float] = {}

    def __init__(self, session: Optional['requests.Session'] = None, target: str = "USD"):
        self.session = session
        self.target = target.upper()

    def get_rate(self, src: str) -> Optional[float]:
        """환율 조회"""
        src = (src or "").upper()
        if not src or src == self.target:
            return 1.0
        if src in self.FX_CACHE:
            return self.FX_CACHE[src]
        if not self.session:
            dbg(201, f"FX session missing for {src}->{self.target}")
            return None
        
        try:
            url = f"https://api.exchangerate.host/latest?base={src}&symbols={self.target}"
            r = self.session.get(url, timeout=8)
            if r.ok:
                rate = float(r.json().get("rates", {}).get(self.target, 0))
                if rate > 0:
                    self.FX_CACHE[src] = rate
                    dbg(202, f"FX {src}->{self.target} = {rate}")
                    return rate
        except Exception as e:
            dbg(203, f"FX fetch failed: {e}")
        return None

    def normalize_field(self, fm, shares: Optional[float] = None):
        """FieldMetadata 환율 정규화"""
        if not fm or fm.value is None:
            return fm
        
        try:
            val = float(fm.value)
        except Exception:
            return fm

        # Per-share 변환
        if fm.unit:
            conv = convert_per_share(val, fm.unit, "USD", shares)
            if conv is not None:
                val = conv
                fm.unit = "USD"

        # 환율 변환
        src_ccy = (fm.currency or "USD").upper()
        rate = self.get_rate(src_ccy)
        if rate:
            fm.value = val * rate
            fm.currency = self.target
            fm.conversion_policy = f"fx:exchangerate.host->{self.target}"
        
        return fm