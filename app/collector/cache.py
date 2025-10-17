# app/collector/cache.py
"""
캐싱 시스템
- CIK 매핑 캐시
"""
import json
from pathlib import Path
from typing import Dict, Optional
from .utils import normalize_ticker, normalize_cik, dbg


class CIKMapper:
    """티커 <-> CIK 매핑 캐시"""
    
    def __init__(self):
        self.cache: Dict[str, str] = {}
        self.reverse_cache: Dict[str, str] = {}
        self.cache_file = Path("cik_cache.json")
        self._load_cache()

    def _load_cache(self):
        """캐시 파일 로드"""
        if self.cache_file.exists():
            try:
                data = json.loads(self.cache_file.read_text())
                self.cache = data.get('ticker_to_cik', {})
                self.reverse_cache = data.get('cik_to_ticker', {})
                dbg(300, f"Loaded CIK cache: {len(self.cache)} entries")
            except Exception as e:
                dbg(301, f"CIK cache load failed: {e}")

    def _save_cache(self):
        """캐시 파일 저장"""
        try:
            data = {
                'ticker_to_cik': self.cache,
                'cik_to_ticker': self.reverse_cache
            }
            self.cache_file.write_text(json.dumps(data, indent=2))
            dbg(302, f"Saved CIK cache: {len(self.cache)} entries")
        except Exception as e:
            dbg(303, f"CIK cache save failed: {e}")

    def get_cik(self, ticker: str, session) -> Optional[str]:
        """티커로 CIK 조회 (캐시 우선)"""
        ticker = normalize_ticker(ticker)
        
        # 캐시 확인
        if ticker in self.cache:
            return self.cache[ticker]
        
        # SEC API 조회
        try:
            url = "https://www.sec.gov/files/company_tickers.json"
            r = session.get(url, timeout=10)
            if r.ok:
                data = r.json()
                for entry in data.values():
                    t = entry.get('ticker', '').upper()
                    cik = normalize_cik(entry.get('cik_str', ''))
                    self.cache[t] = cik
                    self.reverse_cache[cik] = t
                    
                    if t == ticker:
                        self._save_cache()
                        dbg(304, f"Mapped {ticker} -> CIK {cik}")
                        return cik
        except Exception as e:
            dbg(305, f"CIK lookup failed: {e}")
        
        return None