# DermaLens 챗봇

피부·성분 기반 AI 뷰티 챗봇 (의도 분류 + 키워드 추출 + DB 기반 응답)

## 기술 스택

- Python + Flask
- sentence-transformers (HuggingFace `jhgan/ko-sroberta-multitask`)
- scikit-learn, numpy
- 외부 LLM API 미사용, 자체 임베딩 모델로 동작

## 로컬 실행

```bash
pip install -r requirements.txt
python chatbot.py
```

→ http://localhost:5000 접속 (데모 UI)

첫 실행 시 모델 다운로드로 1~2분 소요됩니다.

## API

### POST /chat

요청:
```json
{ "message": "민감성 크림 추천해줘" }
```

응답:
```json
{
  "intent": "PRODUCT_RECOMMEND",
  "score": 0.92,
  "keywords": { "skin_type": "민감성", "category": "크림" },
  "message": "민감성 피부에 맞는 제품을 추천해드릴게요.",
  "components": [
    {
      "type": "card",
      "title": "저자극 진정 크림",
      "description": "자극 성분이 적고 진정 성분 포함",
      "riskLevel": "LOW",
      "buttonText": "제품 상세보기"
    }
  ],
  "quickReplies": ["성분 분석", "피부 진단"]
}
```

**intent 종류:** `PRODUCT_RECOMMEND` / `INGREDIENT_RISK` / `INGREDIENT_ANALYSIS` / `SKIN_TYPE_TEST` / `REVIEW_SUMMARY` / `ANALYSIS_HISTORY` / `UNKNOWN`

**components type:** `card` (제품/성분 카드) / `link` (페이지 이동 버튼)

## 배포 (Railway)

- 시작 명령어: `Procfile` 참고 (gunicorn, worker 1개, timeout 300초)
- 메모리: AI 모델 때문에 **최소 1~2GB RAM 필요**
- 첫 부팅 시 모델 다운로드/로딩으로 1~2분 소요 (헬스체크 타임아웃 주의)
- Python 버전: `runtime.txt` 참고

## 환경변수 (선택)

- `FLASK_DEBUG=1` : 로컬 디버그 모드 (기본 OFF)
- `PORT` : 배포 환경에서 자동 주입 (Railway 등)
