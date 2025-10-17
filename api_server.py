# api_server.py
from fastapi import FastAPI

app = FastAPI(
    title="My API",
    docs_url="/docs",              # 문서 경로 확정
    redoc_url="/redoc",            # 보조 문서 경로
    openapi_url="/openapi.json"    # 스키마 경로
)

@app.get("/")
def health():
    return {"status": "ok"}        # 홈 경로가 404가 아닌지 확인용

# (기존 라우터/엔드포인트가 있다면 아래에 그대로 두세요)
