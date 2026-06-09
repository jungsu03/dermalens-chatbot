from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import re
import json
import difflib
import requests
from functools import lru_cache

# 백엔드 분석 API (DB 조회·추천·저장 담당). 환경변수로 덮어쓸 수 있음.
BACKEND_URL = os.environ.get(
    "BACKEND_URL",
    "https://dermalens-production.up.railway.app/api/analysis/chat/"
)
BACKEND_TIMEOUT = float(os.environ.get("BACKEND_TIMEOUT", "5"))

# 백엔드 성분 자동완성 검색 API (21,810개 DB 부분일치)
INGREDIENT_SEARCH_URL = os.environ.get(
    "INGREDIENT_SEARCH_URL",
    "https://dermalens-production.up.railway.app/api/analysis/ingredients/search/"
)
INGREDIENT_SEARCH_TIMEOUT = float(os.environ.get("INGREDIENT_SEARCH_TIMEOUT", "1.5"))

# 한글 완성형 음절(가-힣) / 영문 단어 패턴
KOREAN_SYLLABLE_RE = re.compile(r'[가-힣]')
ENGLISH_WORD_RE = re.compile(r'[a-zA-Z]{2,}')

# 사용법·상식 질문 패턴 (의도 분류보다 FAQ를 먼저 시도해야 하는 신호)
# 예: "로션은 언제 발라야해?" → "로션" 키워드로 PRODUCT_RECOMMEND 오인되는 거 방지
USAGE_QUESTION_PATTERNS = [
    re.compile(r'언제.{0,8}(발라|써|쓰|먹|적용|해)'),
    re.compile(r'어떻게.{0,8}(발라|써|쓰|먹|관리|해|골라)'),
    re.compile(r'얼마나.{0,8}(자주|많이|발라|써|쓰|오래)'),
    re.compile(r'왜.{0,8}(발라|써|좋|안\s*좋|필요)'),
    re.compile(r'(몇\s*시간|몇\s*번|몇\s*분|몇\s*살)'),
    re.compile(r'(차이가\s*뭐|차이\s*뭐|뭐가\s*달라)'),
    re.compile(r'(써도\s*돼|발라도\s*돼|먹어도\s*돼|해도\s*돼)'),
    re.compile(r'(좋아\??|좋나|좋은가|효과\s*있)'),
    re.compile(r'(뭐야|뭔가요|뭐예요|이\s*뭐|이게\s*뭐)'),
    re.compile(r'(꼭\s*해야|필수야|필요해)'),
    re.compile(r'(같이\s*써|함께\s*써|섞어\s*써)'),
    re.compile(r'순서가?\s*어'),
    re.compile(r'어떤\s*순서'),
]


def is_usage_question(text):
    """사용법·상식 질문 패턴인지 확인 — FAQ 우선 시도 트리거."""
    for pat in USAGE_QUESTION_PATTERNS:
        if pat.search(text):
            return True
    return False


def is_meaningful_text(text):
    """한글 음절(가-힣)이나 2글자 이상 영문이 있어야 의미 있는 입력으로 간주.
    'ㄱㄴㄷ', 'ㅏㅓㅗ', '...' 같은 자모/기호만 있는 입력은 거름."""
    if KOREAN_SYLLABLE_RE.search(text):
        return True
    if ENGLISH_WORD_RE.search(text):
        return True
    return False

app = Flask(__name__)
# CORS 허용 — 프론트엔드(다른 도메인)에서 /chat, /suggest 호출 가능하게
CORS(app, resources={r"/chat": {"origins": "*"}, r"/suggest": {"origins": "*"}})

# =========================================
# DermaLens 챗봇
# - 외부 API 사용 X
# - HuggingFace 문장 임베딩 기반 intent 분류
# - 더미 DB 기반 제품 추천 / 성분 위험도 응답
# - 성분 + 피부타입 조합 답변 적용
# =========================================

model = SentenceTransformer("jhgan/ko-sroberta-multitask")

RISK_LABEL_KO = {"LOW": "낮음", "MEDIUM": "중간", "HIGH": "높음"}

# 피부 타입 진단 페이지 URL (팀에서 정해지면 여기만 수정)
DIAGNOSE_URL = "/diagnose"

# -------------------------------
# 페이지 라우팅 — 사용자가 관련 키워드 말하면 해당 페이지로 안내
# 프론트엔드가 URL 정하면 여기 url만 수정하면 됨
# -------------------------------
PAGES = {
    "product": {
        "url": "/products",
        "title": "제품 페이지",
        "description": "DermaLens가 추천하는 모든 제품을 둘러보세요.",
        "label": "제품 보러가기",
        "keywords": ["제품 페이지", "제품 목록", "전체 제품", "제품 보러", "상품 페이지"]
    },
    "routine": {
        "url": "/routine",
        "title": "바르는 루틴",
        "description": "올바른 화장품 사용 순서를 단계별로 알려드려요.",
        "label": "루틴 보러가기",
        "keywords": ["바르는 루틴", "사용 루틴", "기초 루틴", "스킨케어 루틴", "루틴 페이지"]
    },
    "ocr": {
        "url": "/ocr",
        "title": "성분 사진 분석",
        "description": "화장품 성분표를 사진으로 촬영하면 성분을 자동 분석해드려요.",
        "label": "사진 분석 시작하기",
        "keywords": ["성분 사진", "사진으로 성분", "이미지로 성분", "OCR", "ocr", "카메라로 성분", "성분 촬영"]
    },
    "diagnose": {
        "url": DIAGNOSE_URL,
        "title": "피부 타입 진단",
        "description": "간단한 질문으로 내 피부 타입을 정확하게 알아봐요.",
        "label": "진단 시작하기",
        "keywords": ["피부 진단 페이지", "진단 받기", "진단 시작"]
    },
    "review": {
        "url": "/reviews",
        "title": "리뷰 페이지",
        "description": "다른 사용자들의 실제 사용 후기를 확인해보세요.",
        "label": "리뷰 보러가기",
        "keywords": ["리뷰 페이지", "리뷰 보기", "후기 보기", "리뷰 모음", "후기 페이지"]
    },
    "allergy": {
        "url": "/allergy",
        "title": "알레르기 관리",
        "description": "내 알레르기·기피 성분을 등록·관리해 맞춤 추천에 반영해요.",
        "label": "알레르기 등록",
        "keywords": ["알레르기", "알러지", "기피 성분", "알레르기 등록", "민감 성분 등록"]
    },
    "register_product": {
        "url": "/register-product",
        "title": "제품 등록",
        "description": "DermaLens DB에 없는 제품을 직접 등록해서 분석받을 수 있어요.",
        "label": "제품 등록하기",
        "keywords": ["제품 등록", "상품 등록", "없는 제품", "신제품 등록", "새 제품 등록"]
    },
    "inquiry": {
        "url": "/inquiry",
        "title": "문의하기",
        "description": "DermaLens에 궁금한 점이나 개선 의견을 남겨주세요.",
        "label": "문의 작성하기",
        "keywords": ["문의", "문의하기", "고객센터", "건의", "질문 보내기"]
    },
    "report": {
        "url": "/report",
        "title": "제품 신고",
        "description": "사용 후 문제가 있었던 제품을 신고해 다른 사용자를 보호해주세요.",
        "label": "신고 작성하기",
        "keywords": ["제품 신고", "신고하기", "부작용 신고", "안 좋은 제품 신고", "유해 제품"]
    },
    "feedback": {
        "url": "/feedback",
        "title": "앱 만족도 평가",
        "description": "DermaLens 사용 경험을 평가해 더 나은 서비스에 도움을 주세요.",
        "label": "평가하러 가기",
        "keywords": ["만족도", "앱 평가", "별점", "앱 만족도", "사용 후기 남기기"]
    }
}

# -------------------------------
# 1. Intent 예시 문장
# -------------------------------
intent_examples = {
    "INGREDIENT_ANALYSIS": [
        "성분 분석해줘",
        "화장품 성분 봐줘",
        "성분표 확인해줘",
        "이 제품 성분 분석하고 싶어",
        "사진으로 성분 확인해줘",
        "화장품 성분표 봐줘",
        "성분이 괜찮은지 봐줘",
        "다른 성분 보여줘",
        "다른 성분 알려줘",
        "성분표 이미지로 분석해줘",
        "이미지로 성분 분석"
    ],
    "INGREDIENT_RISK": [
        "이 성분 위험해?",
        "페녹시에탄올 괜찮아?",
        "안 좋은 성분이야?",
        "피부에 자극 있는 성분이야?",
        "알코올 들어가면 안 좋아?",
        "이 성분 안전해?",
        "성분 위험도 알려줘",
        "이 성분 주의해야 해?",
        "히알루론산 알려줘",
        "나이아신아마이드 뭐야",
        "성분 설명해줘",
        "PDRN 어때?",
        "마데카소사이드 알려줘"
    ],
    "PRODUCT_RECOMMEND": [
        "추천해줘",
        "제품 추천해줘",
        "크림 추천",
        "토너 추천",
        "세럼 추천",
        "에센스 추천",
        "앰플 추천",
        "로션 추천",
        "수분크림 추천해줘",
        "민감성 피부 추천",
        "민감성 피부에 좋은 제품 알려줘",
        "건성 피부에 맞는 화장품 추천해줘",
        "지성 피부 토너 추천해줘",
        "복합성 피부에 맞는 제품 있어?",
        "여드름 피부에 맞는 제품 있어?",
        "피부가 예민한데 뭘 써야 해?",
        "보습에 좋은 제품 추천해줘",
        "진정에 좋은 화장품 알려줘",
        # 자연스러운 회화체 — "[피부타입]피부인데 [카테고리] 추천좀" 패턴
        "지성피부인데 에센스 추천좀",
        "건성피부인데 토너 추천좀",
        "민감성피부인데 세럼 추천좀",
        "복합성피부인데 앰플 추천좀",
        "여드름성피부인데 로션 추천좀",
        "건성인데 크림 추천해줘",
        "지성에 맞는 앰플 알려줘",
        "민감성한테 좋은 에센스",
    ],
    "SKIN_TYPE_TEST": [
        "피부 타입 알려줘",
        "피부 진단",
        "나는 무슨 피부야?",
        "내 피부 타입 검사하고 싶어",
        "피부 타입 테스트 해줘",
        "건성인지 지성인지 모르겠어",
        "내 피부 상태 확인해줘"
    ],
    "REVIEW_SUMMARY": [
        "리뷰 요약",
        "후기 어때",
        "리뷰 정리해줘",
        "사람들이 뭐라고 평가해?",
        "장점 단점 알려줘",
        "이 제품 후기 요약해줘"
    ],
    "ANALYSIS_HISTORY": [
        "기록 보여줘",
        "이전 결과",
        "분석 기록 보여줘",
        "내 분석 내역 확인하고 싶어",
        "마이페이지 기록 보여줘",
        "최근 분석 결과 보여줘"
    ]
}

intent_embeddings = {
    intent: model.encode(examples)
    for intent, examples in intent_examples.items()
}

# -------------------------------
# 1-1. 피부타입 의미 매칭용 표현 (별칭 사전이 못 잡을 때 폴백)
# -------------------------------
skin_type_phrases = {
    "지성": [
        "얼굴에 기름이 많이 나",
        "기름이 줄줄 흘러",
        "피지가 많이 나와",
        "번들거려",
        "번질거려",
        "T존이 번들거려",
        "유분이 많아",
        "모공이 넓어",
        "저녁 되면 떡져",
        "화장이 다 떠올라",
        "얼굴이 번들번들해",
        "피부가 끈적해"
    ],
    "건성": [
        "피부가 건조해",
        "땡겨",
        "당기는 느낌이야",
        "각질이 일어나",
        "메마른 느낌",
        "푸석해",
        "보습이 부족해",
        "트는 느낌",
        "갈라져",
        "얼굴이 까칠해",
        "건조해서 가려워"
    ],
    "민감성": [
        "피부가 예민해",
        "쉽게 붉어져",
        "자극을 잘 받아",
        "발갛게 올라와",
        "쉽게 트러블 생겨",
        "따가워",
        "간지러워",
        "화끈거려",
        "조금만 발라도 빨개져",
        "자극에 약해"
    ],
    "복합성": [
        "T존은 기름지고 볼은 건조해",
        "이마는 번들 볼은 땡김",
        "부분적으로 건조하고 기름져",
        "코는 번들거리는데 볼은 건조해",
        "복합성 같아",
        "부위마다 피부 상태가 달라"
    ],
    "여드름성": [
        "여드름이 많아",
        "트러블이 자주 나",
        "뾰루지가 자꾸 생겨",
        "블랙헤드가 많아",
        "화농성 여드름이 나",
        "좁쌀 여드름이 올라와",
        "턱에 뾰루지가 자주 나"
    ]
}

skin_type_phrase_embeddings = {
    skin_type: model.encode(phrases)
    for skin_type, phrases in skin_type_phrases.items()
}

# 의미 매칭 임계값 (낮으면 false match, 높으면 놓침)
SKIN_TYPE_SEMANTIC_THRESHOLD = 0.55

# -------------------------------
# 1-2. FAQ 지식 베이스 로드 (faq.json)
# 화장품 사용법·상식 Q&A 200개. 임베딩으로 의미 매칭.
# -------------------------------
FAQ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faq.json")
FAQ_MATCH_THRESHOLD = 0.7  # 0.7 이상이면 FAQ 답변 반환 (정확도 우선 — 큐레이션된 72개 Q&A 기준)

try:
    with open(FAQ_PATH, "r", encoding="utf-8") as f:
        faq_bank = json.load(f)
    faq_questions = [item["question"] for item in faq_bank]
    faq_question_embeddings = model.encode(faq_questions)
    print(f"[FAQ] {len(faq_bank)}개 Q&A 로드 완료")
except FileNotFoundError:
    faq_bank = []
    faq_question_embeddings = None
    print("[FAQ] faq.json 없음 - FAQ 기능 비활성화")


def find_faq_match(user_input, embedding=None):
    """사용자 질문을 FAQ 200개와 비교해서 가장 비슷한 답변 반환.
    유사도 < 임계값이면 None 반환 (다른 흐름으로 진행)."""
    if not faq_bank or faq_question_embeddings is None:
        return None
    if embedding is None:
        embedding = model.encode([user_input])

    scores = cosine_similarity(embedding, faq_question_embeddings)[0]
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score < FAQ_MATCH_THRESHOLD:
        return None

    matched = faq_bank[best_idx]
    return {
        "answer": matched["answer"],
        "matched_question": matched["question"],
        "category": matched.get("category", ""),
        "score": best_score
    }


# -------------------------------
# 1-3. 성분명 오타 보정 (fuzzy matching)
# 사용자가 "글리세인", "히아루론산" 같이 오타 쳐도
# 가장 비슷한 DB 성분명에 매칭시킴.
# -------------------------------
FUZZY_INGREDIENT_CUTOFF = 0.7  # 0.7 이상 유사도면 보정 시도


def fuzzy_correct_ingredient(text):
    """텍스트에서 오타가 있는 성분명을 추정해 보정.
    DB에 있는 성분명/별칭과 비교해 가장 비슷한 거 반환."""
    # 비교 대상: DB 성분명 + 모든 별칭
    candidates = list(ingredients_db.keys())
    for aliases in ingredient_aliases.values():
        candidates.extend(aliases)

    # 입력 토큰화 (공백/구두점 기준 단어 분리)
    tokens = re.findall(r'[가-힣a-zA-Z]+', text)

    for token in tokens:
        if len(token) < 2:
            continue
        # 정확히 일치하는 게 있으면 보정 불필요
        if token in candidates:
            return None
        matches = difflib.get_close_matches(
            token, candidates, n=1, cutoff=FUZZY_INGREDIENT_CUTOFF
        )
        if matches:
            corrected_alias = matches[0]
            # 매칭된 별칭이 어느 성분에 속하는지 역추적
            for ingredient, aliases in ingredient_aliases.items():
                if corrected_alias == ingredient or corrected_alias in aliases:
                    return {"original": token, "corrected": ingredient}
            if corrected_alias in ingredients_db:
                return {"original": token, "corrected": corrected_alias}

    return None

# -------------------------------
# 1-4. 성분명 후보 추출 (백엔드 패스스루용)
# 챗봇 사전(27개)에 없는 성분이라도 백엔드 DB(21,810개)에는 있을 수 있음.
# 메시지에서 명사 후보 1개 뽑아 백엔드로 던지면 백엔드가 부분일치 검색.
# 예: "알로에 성분 알려줘" → "알로에" → 백엔드 → "알로에베라잎추출물"
# -------------------------------
INGREDIENT_STOPWORDS = {
    # 동사·서술어
    "알려줘", "알려", "보여줘", "보여", "확인", "분석", "추천",
    "사용", "발라", "써도", "발라도", "써", "쓰",
    # 질문어
    "뭐야", "뭔가요", "뭐예요", "어때", "어떻게", "어떤", "무엇",
    # 평가어
    "위험", "위험해", "괜찮", "괜찮아", "안전", "좋아", "나빠", "주의",
    "효과", "도움", "좋은가", "나쁜가",
    # 일반 명사
    "성분", "피부", "제품", "화장품", "정보",
    # 피부타입 (별도 키워드로 이미 추출됨)
    "지성", "건성", "민감성", "복합성", "여드름성", "민감", "여드름", "복합",
    # 카테고리 (별도 키워드로 이미 추출됨)
    "크림", "토너", "세럼", "로션", "에센스", "앰플", "스킨", "젤",
    # 지시·기타
    "그거", "이거", "저거", "그게", "이게", "저게", "거기", "여기",
    # 시간·장소 (성분 아님)
    "어디", "지금", "오늘", "내일", "어제", "방금", "언제", "왜",
    "안녕", "헬로", "하이",
}

# 조사 분류 — STRICT는 항상 조사(명사 끝이 될 가능성 거의 없음), WEAK는 명사 일부일 수도
# 예: "녹차를" 의 "를" → STRICT (확실히 조사 → strip)
#     "알로에" 의 "에" → WEAK (명사 끝일 수도 → 4자 이상에서만 strip)
STRICT_PARTICLES = ["이라는", "라는", "이라고", "라고",
                    "에게", "에서", "에는", "에도", "으로",
                    "을", "를", "의", "과", "와", "랑",
                    "은", "는", "이", "가"]
WEAK_PARTICLES = ["에", "도", "만", "이라", "이라면"]
ALL_PARTICLES = STRICT_PARTICLES + WEAK_PARTICLES


def extract_ingredient_candidate(text):
    """메시지에서 성분명 후보 1개 추출 (백엔드 부분일치 검색용).

    동작:
    - 한글 2자 이상 / 영문 3자 이상 토큰 뽑음
    - 조사 제거: STRICT 조사는 항상 strip("녹차를"→"녹차"),
                 WEAK 조사는 4자 이상에서만 strip("알로에"→"알로에" 유지)
    - 조사 떼고 남는 stem이 불용어면 토큰 전체를 불용어로 간주
      (예: "피부에" → stem "피부"가 불용어 → 후보 제외)
    - 남은 후보 중 가장 긴 것 반환
    """
    tokens = re.findall(r'[가-힣]{2,}|[a-zA-Z]{3,}', text)

    candidates = []
    for token in tokens:
        cleaned = token
        for p in ALL_PARTICLES:
            if not cleaned.endswith(p):
                continue
            remaining = len(cleaned) - len(p)
            if remaining < 2:
                continue
            stem = cleaned[:-len(p)]
            # stem이 불용어면 이 토큰 전체를 불용어 처리
            # ("피부에" → "피부"가 불용어 → 후보 제외)
            if stem in INGREDIENT_STOPWORDS:
                cleaned = None
                break
            # STRICT 조사거나 4자 이상이면 strip
            # WEAK 조사 + 3자 이하는 명사 일부 가능성 → 유지
            if p in STRICT_PARTICLES or len(cleaned) >= 4:
                cleaned = stem
            break

        if cleaned is None:
            continue
        if cleaned in INGREDIENT_STOPWORDS or len(cleaned) < 2:
            continue
        candidates.append(cleaned)

    if not candidates:
        return None

    return max(candidates, key=len)


# -------------------------------
# 2. 더미 제품 DB
# -------------------------------
products_db = [
    {"name": "저자극 진정 크림", "skin_type": "민감성", "category": "크림",
     "description": "자극 가능 성분이 적고 진정 성분이 포함된 제품입니다.", "riskLevel": "LOW"},
    {"name": "약산성 수분 토너", "skin_type": "민감성", "category": "토너",
     "description": "피부 자극을 줄이고 수분 공급에 도움을 주는 토너입니다.", "riskLevel": "LOW"},
    {"name": "장벽 강화 세럼", "skin_type": "민감성", "category": "세럼",
     "description": "피부 장벽 관리와 진정 케어에 도움을 주는 세럼입니다.", "riskLevel": "LOW"},
    {"name": "수분 장벽 크림", "skin_type": "건성", "category": "크림",
     "description": "보습 성분이 풍부해 건성 피부에 적합한 제품입니다.", "riskLevel": "LOW"},
    {"name": "고보습 영양 크림", "skin_type": "건성", "category": "크림",
     "description": "건조한 피부에 영양과 수분 보충을 도와주는 크림입니다.", "riskLevel": "LOW"},
    {"name": "히알루론 수분 세럼", "skin_type": "건성", "category": "세럼",
     "description": "수분 공급과 보습 유지에 도움을 주는 세럼입니다.", "riskLevel": "LOW"},
    {"name": "피지 케어 토너", "skin_type": "지성", "category": "토너",
     "description": "피지 조절에 도움을 주는 산뜻한 토너입니다.", "riskLevel": "MEDIUM"},
    {"name": "BHA 케어 세럼", "skin_type": "지성", "category": "세럼",
     "description": "번들거림 완화와 모공 케어에 도움을 주는 세럼입니다.", "riskLevel": "MEDIUM"},
    {"name": "산뜻 수분 젤크림", "skin_type": "지성", "category": "크림",
     "description": "가볍게 흡수되어 지성 피부가 사용하기 좋은 젤 타입 크림입니다.", "riskLevel": "LOW"},
    {"name": "밸런스 수분 토너", "skin_type": "복합성", "category": "토너",
     "description": "유분과 수분 밸런스를 맞추는 데 도움을 주는 토너입니다.", "riskLevel": "LOW"},
    {"name": "복합성 밸런스 크림", "skin_type": "복합성", "category": "크림",
     "description": "건조한 부위와 번들거리는 부위를 함께 관리하는 크림입니다.", "riskLevel": "LOW"},
    {"name": "진정 시카 세럼", "skin_type": "여드름성", "category": "세럼",
     "description": "트러블 피부 진정과 피부 컨디션 관리에 도움을 주는 세럼입니다.", "riskLevel": "LOW"},
    {"name": "트러블 케어 토너", "skin_type": "여드름성", "category": "토너",
     "description": "트러블 피부의 피지와 각질 관리에 도움을 주는 토너입니다.", "riskLevel": "MEDIUM"},
    {"name": "진정 수분 크림", "skin_type": "여드름성", "category": "크림",
     "description": "무겁지 않은 사용감으로 트러블 피부의 부담을 줄인 크림입니다.", "riskLevel": "LOW"}
]

# -------------------------------
# 3. 더미 성분 DB
# -------------------------------
ingredients_db = {
    "페녹시에탄올": {"risk": "MEDIUM", "description": "보존제로 사용되며 민감성 피부에는 자극이 될 수 있습니다."},
    "알코올": {"risk": "HIGH", "description": "피부를 건조하게 만들 수 있어 민감성·건성 피부는 주의가 필요합니다."},
    "병풀추출물": {"risk": "LOW", "description": "피부 진정에 도움을 줄 수 있는 성분입니다."},
    "나이아신아마이드": {"risk": "LOW", "description": "피부 톤 개선과 장벽 관리에 도움을 줄 수 있는 성분입니다."},
    "히알루론산": {"risk": "LOW", "description": "수분 공급과 보습 유지에 도움을 줄 수 있는 성분입니다."},
    "세라마이드": {"risk": "LOW", "description": "피부 장벽 강화와 보습에 도움을 줄 수 있는 성분입니다."},
    "레티놀": {"risk": "MEDIUM", "description": "피부 탄력 관리에 사용되지만 민감성 피부에는 자극이 될 수 있습니다."},
    "이리스산": {"risk": "MEDIUM", "description": "각질과 피지 관리에 도움을 줄 수 있지만 과사용 시 자극이 생길 수 있습니다."},
    "향료": {"risk": "HIGH", "description": "향을 더하기 위해 사용되며 민감성 피부에는 자극 가능성이 있습니다."},
    "파라벤": {"risk": "MEDIUM", "description": "보존제로 사용되며 일부 사용자에게는 민감 반응이 나타날 수 있습니다."},
    "티트리오일": {"risk": "MEDIUM", "description": "트러블 관리에 사용되지만 고농도에서는 자극이 될 수 있습니다."},
    "글리세린": {"risk": "LOW", "description": "수분을 끌어당겨 피부 보습에 도움을 주는 성분입니다."},
    "PDRN": {"risk": "LOW", "description": "연어 DNA에서 추출한 성분으로 피부 재생과 진정에 도움을 줄 수 있습니다."},
    "마데카소사이드": {"risk": "LOW", "description": "병풀의 핵심 진정 성분으로 자극받은 피부 회복에 도움을 줄 수 있습니다."},
    "판테놀": {"risk": "LOW", "description": "비타민 B5 유도체로 보습과 피부 진정에 도움을 줄 수 있습니다."},
    "아데노신": {"risk": "LOW", "description": "주름 개선 기능성 고시 성분으로 피부 탄력 관리에 도움을 줄 수 있습니다."},
    "콜라겐": {"risk": "LOW", "description": "피부에 보습막을 형성해 탄력 유지에 도움을 줄 수 있는 성분입니다."},
    "트라넥삼산": {"risk": "LOW", "description": "피부 톤 개선과 진정에 도움을 줄 수 있는 성분입니다."},
    "갈락토미세스": {"risk": "LOW", "description": "효모 발효 추출물로 피부 결과 톤 관리에 도움을 줄 수 있습니다."},
    "AHA": {"risk": "MEDIUM", "description": "각질 제거에 도움을 주지만 자극과 광민감성을 유발할 수 있어 사용 시 자외선 차단이 필요합니다."},
    "PHA": {"risk": "LOW", "description": "AHA보다 자극이 적은 각질 관리 성분으로 보습 효과도 함께 기대할 수 있습니다."},
    "비타민C": {"risk": "MEDIUM", "description": "미백·항산화에 도움을 줄 수 있지만 고농도에서는 자극이 될 수 있습니다."},
    "펩타이드": {"risk": "LOW", "description": "피부 탄력과 결 관리에 도움을 줄 수 있는 단백질 성분입니다."},
    "스쿠알란": {"risk": "LOW", "description": "피부에 가깝게 흡수되는 보습 오일 성분으로 다양한 피부 타입에 활용됩니다."},
    "알란토인": {"risk": "LOW", "description": "피부 진정과 보호에 도움을 줄 수 있는 저자극 성분입니다."},
    "프로폴리스": {"risk": "LOW", "description": "꿀벌이 만드는 천연 성분으로 진정과 피부 장벽 관리에 도움을 줄 수 있습니다."},
    "시어버터": {"risk": "LOW", "description": "풍부한 보습감을 제공하는 식물성 버터 성분입니다."}
}

# -------------------------------
# 4. 유사 검색용 키워드 사전
# -------------------------------
ingredient_aliases = {
    "페녹시에탄올": ["페녹시에탄올", "phenoxyethanol", "페녹시"],
    "알코올": ["알코올", "에탄올", "ethanol", "alcohol"],
    "병풀추출물": ["병풀추출물", "병풀", "시카", "cica", "centella"],
    "나이아신아마이드": ["나이아신아마이드", "나이아신", "niacinamide"],
    "히알루론산": ["히알루론산", "히알루론", "hyaluronic", "hyaluronic acid"],
    "세라마이드": ["세라마이드", "ceramide"],
    "레티놀": ["레티놀", "retinol"],
    "이리스산": ["이리스산", "bha", "베타", "salicylic"],
    "향료": ["향료", "향", "fragrance", "퍼퓸", "parfum"],
    "파라벤": ["파라벤", "paraben"],
    "티트리오일": ["티트리오일", "티트리", "tea tree"],
    "글리세린": ["글리세린", "glycerin"],
    "PDRN": ["PDRN", "pdrn", "피디알엔", "폴리데옥시리보뉴클레오티드", "연어주사", "연어 추출물", "polydeoxyribonucleotide"],
    "마데카소사이드": ["마데카소사이드", "마데카소", "madecassoside", "마데카"],
    "판테놀": ["판테놀", "panthenol", "비타민B5", "비타민 B5", "프로비타민B5"],
    "아데노신": ["아데노신", "adenosine"],
    "콜라겐": ["콜라겐", "collagen"],
    "트라넥삼산": ["트라넥삼산", "tranexamic", "트라넥삼"],
    "갈락토미세스": ["갈락토미세스", "galactomyces", "갈락토"],
    "AHA": ["AHA", "aha", "글리콜산", "glycolic", "유산", "lactic acid"],
    "PHA": ["PHA", "pha", "글루코노락톤", "gluconolactone"],
    "비타민C": ["비타민C", "비타민 C", "vitamin c", "vitaminc", "아스코르빈산", "아스코르브산", "ascorbic"],
    "펩타이드": ["펩타이드", "peptide", "펩티드"],
    "스쿠알란": ["스쿠알란", "squalane", "스쿠알렌", "squalene"],
    "알란토인": ["알란토인", "allantoin"],
    "프로폴리스": ["프로폴리스", "propolis"],
    "시어버터": ["시어버터", "shea butter", "시어"]
}

# -------------------------------
# 마스터데이터 기반 8가지 피부타입 (유분 O/D × 민감 S/N × 수분 +/-)
# 사용자는 코드(OS+ 등)를 모를 수 있어 한글 이름으로 노출하고,
# "잘 모르겠어요" 선택 시 3단계 진단으로 타입을 찾아준다.
#   backend     : 백엔드(건성/지성/민감만 지원)로 변환할 키워드 (유분·민감 축)
#   comment_key : 성분 코멘트 사전(지성/건성/민감성) 재사용 키
# -------------------------------
SKIN_TYPES_8 = {
    "OS+": {"kr": "민지형",     "en": "Oily, Sensitive, Hydrated",       "desc": "유분은 많고 민감도도 높으며 수분은 비교적 유지되는 피부", "backend": "지성 민감", "comment_key": "민감성"},
    "OS-": {"kr": "수부민지형", "en": "Oily, Sensitive, Dehydrated",     "desc": "유분은 많지만 수분이 부족하고 민감도가 높은 피부",       "backend": "지성 민감", "comment_key": "민감성"},
    "ON+": {"kr": "건지형",     "en": "Oily, Non-Sensitive, Hydrated",   "desc": "지성이지만 비교적 건강하고 수분이 유지되는 피부",         "backend": "지성",     "comment_key": "지성"},
    "ON-": {"kr": "수부지형",   "en": "Oily, Non-Sensitive, Dehydrated", "desc": "지성이면서 수분이 부족한 피부",                          "backend": "지성",     "comment_key": "지성"},
    "DS+": {"kr": "민건형",     "en": "Dry, Sensitive, Hydrated",        "desc": "건성이며 민감도가 높은 피부",                            "backend": "건성 민감", "comment_key": "민감성"},
    "DS-": {"kr": "극건민감형", "en": "Dry, Sensitive, Dehydrated",      "desc": "매우 건조하고 민감한 피부",                              "backend": "건성 민감", "comment_key": "민감성"},
    "DN+": {"kr": "건강건성형", "en": "Dry, Non-Sensitive, Hydrated",    "desc": "건성이지만 피부 장벽이 비교적 건강한 피부",               "backend": "건성",     "comment_key": "건성"},
    "DN-": {"kr": "수부건형",   "en": "Dry, Non-Sensitive, Dehydrated",  "desc": "건성이며 수분 부족이 두드러지는 피부",                    "backend": "건성",     "comment_key": "건성"},
}

# 한글 이름 → 코드 역참조
SKIN_KR_TO_CODE = {info["kr"]: code for code, info in SKIN_TYPES_8.items()}

# 8타입 한글 이름 → 성분 코멘트 사전 키(지성/건성/민감성)
SKIN_COMMENT_KEY = {info["kr"]: info["comment_key"] for info in SKIN_TYPES_8.values()}

skin_aliases = {}
# 8타입을 먼저 등록 — '건강건성형'에 '건성'이 부분일치하는 등 충돌 방지(먼저 매칭되게)
for _code, _info in SKIN_TYPES_8.items():
    skin_aliases[_info["kr"]] = [_info["kr"], _code.lower()]
# 기존 5타입 — 자유 입력·기존 칩·FAQ 호환용 (의미 매칭/백엔드 패스스루로 유지)
skin_aliases.update({
    "민감성": ["민감성", "예민", "자극", "붉어", "발갛", "진정"],
    "건성": ["건성", "건조", "보습", "땡김", "트임"],
    "지성": ["지성", "피지", "번들", "기름", "유분", "모공"],
    "복합성": ["복합성", "티존", "t존", "부분건조", "부분유분"],
    "여드름성": ["여드름성", "여드름", "트러블", "뾰루지", "각질"],
})

# 피부타입 → 백엔드 지원 키워드(건성/지성/민감)로 매핑
# 백엔드는 네이버 데이터 한계로 건성·지성·민감만 가지고 있어,
# 8타입은 유분·민감 축으로 변환해서 보낸다(OS→지성 민감, ON→지성, DS→건성 민감, DN→건성).
SKIN_TYPE_BACKEND_MAP = {info["kr"]: info["backend"] for info in SKIN_TYPES_8.values()}
SKIN_TYPE_BACKEND_MAP.update({
    "복합성": "건성 지성",     # 복합성 = 건조 부위 + 유분 부위 혼합
    "여드름성": "지성 민감",   # 여드름 = 지성(피지) + 민감(자극·진정 필요)
})

# 카테고리 분리 — 백엔드가 에센스/앰플/세럼을 별도 카테고리로 가지고 있음
# (에센스: 묽음/수분, 세럼: 농축 성분, 앰플: 고농도 — 다른 제품군)
category_aliases = {
    "크림": ["크림", "수분크림", "젤크림", "보습크림"],
    "토너": ["토너", "스킨"],
    "에센스": ["에센스"],
    "세럼": ["세럼"],
    "앰플": ["앰플"],
    "로션": ["로션", "에멀젼"]
}

# 피부 타입별 성분 코멘트
skin_ingredient_comments = {
    "지성": {
        "글리세린": "지성 피부도 사용할 수 있지만, 너무 무거운 제형에서는 번들거림이 느껴질 수 있어 산뜻한 제형을 선택하는 것이 좋습니다.",
        "히알루론산": "지성 피부에도 비교적 잘 맞는 보습 성분이며, 가벼운 수분 세럼이나 젤 타입 제형에 들어간 경우 사용하기 좋습니다.",
        "세라마이드": "지성 피부도 장벽 관리가 필요하기 때문에 사용할 수 있지만, 유분감이 강한 크림보다는 가벼운 제형이 적합합니다.",
        "나이아신아마이드": "지성 피부의 피지 조절과 피부 결 관리에 도움을 줄 수 있어 비교적 잘 맞는 성분입니다.",
        "이리스산": "지성 피부의 피지와 각질 관리에 도움을 줄 수 있지만, 과하게 사용하면 자극이 생길 수 있습니다.",
        "티트리오일": "트러블 관리에 도움을 줄 수 있지만, 고농도 제형은 자극이 될 수 있어 주의가 필요합니다.",
        "알코올": "지성 피부에는 산뜻하게 느껴질 수 있지만, 장기적으로는 건조감이나 자극을 유발할 수 있어 주의가 필요합니다.",
        "향료": "지성 피부라도 향료는 자극 가능성이 있어 민감하게 반응한다면 피하는 것이 좋습니다.",
        "레티놀": "피부 결 관리에 도움을 줄 수 있지만, 처음 사용할 때는 낮은 농도부터 천천히 사용하는 것이 좋습니다.",
        "페녹시에탄올": "일반적으로 보존제로 쓰이지만, 피부가 예민하게 반응하면 사용을 줄이는 것이 좋습니다.",
        "파라벤": "보존제로 사용되며 대체로 안전 사용되지만, 민감 반응이 있다면 피하는 것이 좋습니다.",
        "병풀추출물": "진정 케어에 도움을 줄 수 있어 지성 피부의 트러블 진정에도 활용하기 좋습니다.",
        "PDRN": "지성 피부의 재생과 피부결 관리에 도움을 줄 수 있으며, 가벼운 세럼 제형으로 사용하면 부담이 적습니다.",
        "마데카소사이드": "지성 피부의 트러블 진정과 결 관리에 도움을 줄 수 있어 가벼운 제형으로 사용하기 좋습니다.",
        "판테놀": "지성 피부에도 부담 없이 사용할 수 있는 저자극 보습·진정 성분입니다.",
        "아데노신": "지성 피부의 탄력 관리에 사용할 수 있으며, 가벼운 세럼 제형이 적합합니다.",
        "콜라겐": "지성 피부에도 사용할 수 있지만 너무 무거운 제형은 번들거림이 생길 수 있어 가벼운 제품을 선택하는 것이 좋습니다.",
        "트라넥삼산": "지성 피부의 피부 톤 개선과 진정에 도움을 줄 수 있어 비교적 잘 맞습니다.",
        "갈락토미세스": "지성 피부의 피지 조절과 결 관리에 도움을 줄 수 있어 잘 맞는 편입니다.",
        "AHA": "지성 피부의 각질·피지 관리에 도움을 줄 수 있지만, 사용 시 자외선 차단을 함께 해주는 것이 좋습니다.",
        "PHA": "지성 피부의 각질·피지 관리에 도움을 줄 수 있으며 AHA보다 자극이 적어 부담이 덜합니다.",
        "비타민C": "지성 피부의 피부 톤 관리에 도움을 줄 수 있지만, 고농도에서는 자극이 될 수 있어 농도 조절이 필요합니다.",
        "펩타이드": "지성 피부에도 사용할 수 있으며 탄력 관리에 도움을 줄 수 있어 가벼운 제형이 적합합니다.",
        "스쿠알란": "지성 피부에는 산뜻한 제형의 스쿠알란이 적합하며 과한 양은 피하는 것이 좋습니다.",
        "알란토인": "지성 피부에도 부담 없이 사용할 수 있는 저자극 진정 성분입니다.",
        "프로폴리스": "지성 피부도 사용할 수 있지만 무거운 제형은 부담될 수 있어 가벼운 제품을 선택하는 것이 좋습니다.",
        "시어버터": "지성 피부에는 무거운 사용감으로 부담될 수 있어 부분 사용이 적합합니다."
    },
    "건성": {
        "글리세린": "건성 피부에는 수분을 끌어당기는 보습 성분으로 잘 맞는 편입니다.",
        "히알루론산": "건성 피부에 수분 공급에 도움을 줄 수 있어 잘 맞는 성분입니다.",
        "세라마이드": "건성 피부의 장벽 강화와 보습 유지에 도움을 줄 수 있어 특히 적합합니다.",
        "나이아신아마이드": "장벽 관리와 피부 톤 관리에 도움을 줄 수 있어 건성 피부도 사용할 수 있습니다.",
        "이리스산": "각질 관리에 도움을 줄 수 있지만 건성 피부에는 건조감이 심해질 수 있어 주의가 필요합니다.",
        "티트리오일": "건성 피부에는 자극이나 건조감을 줄 수 있어 신중하게 사용하는 것이 좋습니다.",
        "알코올": "건성 피부에는 건조함을 더 느끼게 할 수 있어 피하는 것이 좋습니다.",
        "향료": "건성 피부가 예민해진 상태라면 향료가 자극이 될 수 있어 주의가 필요합니다.",
        "레티놀": "탄력 관리에 도움을 줄 수 있지만 건조감이 생길 수 있어 보습제와 함께 사용하는 것이 좋습니다.",
        "페녹시에탄올": "보존제로 쓰이는 성분이며, 건조하고 예민한 피부라면 반응 여부를 확인하는 것이 좋습니다.",
        "파라벤": "일반적인 보존 성분이지만, 피부가 예민하게 반응한다면 피하는 것이 좋습니다.",
        "병풀추출물": "건조로 인해 예민해진 피부 진정에 도움을 줄 수 있습니다.",
        "PDRN": "건성 피부의 재생과 보습 유지에 도움을 줄 수 있어 잘 맞는 편이며, 수분 보습제와 함께 사용하면 더 좋습니다.",
        "마데카소사이드": "건조로 예민해진 피부 진정에 도움을 줄 수 있으며 보습제와 함께 사용하면 좋습니다.",
        "판테놀": "건성 피부의 보습 유지와 진정에 도움을 줄 수 있어 잘 맞는 편입니다.",
        "아데노신": "건성 피부의 주름·탄력 관리에 도움을 줄 수 있어 잘 맞는 편입니다.",
        "콜라겐": "건성 피부의 보습과 탄력 유지에 도움을 줄 수 있어 잘 맞는 편입니다.",
        "트라넥삼산": "건성 피부의 톤 관리와 진정에 도움을 줄 수 있어 잘 맞는 편입니다.",
        "갈락토미세스": "건성 피부의 보습 유지와 결 관리에 도움을 줄 수 있는 성분입니다.",
        "AHA": "건성 피부에는 건조감을 심하게 만들 수 있어 신중하게 사용하고 보습을 함께 해야 합니다.",
        "PHA": "AHA보다 자극이 적어 건성 피부도 비교적 부담 없이 사용할 수 있습니다.",
        "비타민C": "건성 피부에도 사용할 수 있지만 고농도에서는 자극이 될 수 있어 보습제와 함께 사용하는 것이 좋습니다.",
        "펩타이드": "건성 피부의 탄력·보습 관리에 도움을 줄 수 있어 잘 맞는 편입니다.",
        "스쿠알란": "건성 피부의 보습과 유분 보충에 도움을 줄 수 있어 잘 맞는 편입니다.",
        "알란토인": "건성 피부의 진정과 보호에 도움을 줄 수 있어 잘 맞는 편입니다.",
        "프로폴리스": "건성 피부의 진정과 장벽 관리에 도움을 줄 수 있어 잘 맞는 편입니다.",
        "시어버터": "건성 피부의 강력한 보습에 도움을 줄 수 있어 잘 맞는 편입니다."
    },
    "민감성": {
        "글리세린": "민감성 피부도 비교적 부담 없이 사용할 수 있는 보습 성분이지만, 제품 전체 성분 구성을 함께 확인하는 것이 좋습니다.",
        "히알루론산": "민감성 피부에도 비교적 잘 맞는 보습 성분이지만, 처음에는 소량 테스트 후 사용하는 것이 좋습니다.",
        "세라마이드": "민감성 피부의 장벽 관리에 도움을 줄 수 있어 잘 맞는 편입니다.",
        "나이아신아마이드": "피부 장벽 관리에 도움을 줄 수 있지만, 고농도에서는 발갛음을 띌 수 있어 주의가 필요합니다.",
        "이리스산": "민감성 피부에는 자극이 될 수 있어 낮은 농도부터 신중하게 사용하는 것이 좋습니다.",
        "티트리오일": "민감성 피부에는 자극이 될 수 있으므로 고농도 제형은 피하는 것이 좋습니다.",
        "알코올": "민감성 피부에는 자극과 건조감을 유발할 수 있어 피하는 것이 좋습니다.",
        "향료": "민감성 피부에는 자극 가능성이 높아 피하는 것이 좋습니다.",
        "레티놀": "민감성 피부에는 자극이 강할 수 있어 처음부터 자주 사용하는 것은 권장하지 않습니다.",
        "페녹시에탄올": "보존제로 사용되지만 민감성 피부에는 자극이 될 수 있어 반응을 확인하는 것이 좋습니다.",
        "파라벤": "보존제로 사용되며 일부 민감성 피부에는 반응이 나타날 수 있어 주의가 필요합니다.",
        "병풀추출물": "피부 진정에 도움을 줄 수 있어 민감성 피부에 비교적 잘 맞는 성분입니다.",
        "PDRN": "민감성 피부의 진정과 장벽 회복에 도움을 줄 수 있는 편이지만, 처음에는 소량 테스트 후 사용하는 것이 좋습니다.",
        "마데카소사이드": "민감성 피부 진정에 잘 맞는 편이며 발갛게 올라온 상태 회복에 도움을 줄 수 있습니다.",
        "판테놀": "민감성 피부에도 비교적 잘 맞는 저자극 보습 성분입니다.",
        "아데노신": "민감성 피부도 사용할 수 있지만 처음에는 소량 테스트 후 사용하는 것이 좋습니다.",
        "콜라겐": "민감성 피부도 비교적 부담 없이 사용할 수 있는 보습 성분입니다.",
        "트라넥삼산": "민감성 피부에도 비교적 잘 맞는 진정·미백 성분입니다.",
        "갈락토미세스": "민감성 피부는 처음에 소량 테스트 후 사용하는 것이 좋습니다.",
        "AHA": "민감성 피부에는 자극이 강할 수 있어 낮은 농도부터 시작하거나 피하는 것이 좋습니다.",
        "PHA": "민감성 피부도 비교적 잘 맞는 저자극 각질 관리 성분입니다.",
        "비타민C": "민감성 피부에는 자극이 될 수 있어 저농도 유도체부터 시작하는 것이 좋습니다.",
        "펩타이드": "민감성 피부도 비교적 부담 없이 사용할 수 있는 저자극 성분입니다.",
        "스쿠알란": "민감성 피부에도 비교적 잘 맞는 저자극 보습 성분입니다.",
        "알란토인": "민감성 피부에 매우 잘 맞는 진정 성분으로 발갛음 완화에 도움을 줄 수 있습니다.",
        "프로폴리스": "민감성 피부는 벌 관련 성분에 알레르기 반응이 있을 수 있어 패치 테스트가 필요합니다.",
        "시어버터": "민감성 피부도 사용할 수 있는 저자극 보습 성분이지만 무거운 사용감은 주의가 필요합니다."
    },
    "여드름성": {
        "글리세린": "여드름성 피부도 사용할 수 있지만, 제품 제형이 너무 무겁다면 모공 부담이 느껴질 수 있습니다.",
        "히알루론산": "여드름성 피부에도 가볍게 수분을 공급하는 데 도움을 줄 수 있어 잘 맞는 편입니다.",
        "세라마이드": "장벽 회복에 도움을 줄 수 있어 여드름성 피부에도 도움이 될 수 있습니다.",
        "나이아신아마이드": "피지 조절과 피부 장벽 관리에 도움을 줄 수 있어 여드름성 피부에 비교적 잘 맞습니다.",
        "이리스산": "피지와 각질 관리에 도움을 줄 수 있어 여드름성 피부에 자주 사용되지만, 과사용은 피해야 합니다.",
        "티트리오일": "트러블 진정에 도움을 줄 수 있지만, 고농도에서는 자극이 될 수 있어 주의가 필요합니다.",
        "알코올": "일시적으로 산뜻하게 느껴질 수 있지만 피부 장벽을 건조하게 만들 수 있어 주의가 필요합니다.",
        "향료": "트러블 피부에는 자극 요인이 될 수 있어 피하는 것이 좋습니다.",
        "레티놀": "피부 결 관리에 도움을 줄 수 있지만, 여드름성 피부가 예민한 상태라면 천천히 적응해야 합니다.",
        "페녹시에탄올": "보존제로 사용되며 대체로 안전이지만, 트러블이 심한 경우 피부 반응을 확인하는 것이 좋습니다.",
        "파라벤": "보존제로 쓰이며 일부 사용자에게 민감 반응이 있을 수 있어 확인이 필요합니다.",
        "병풀추출물": "트러블로 예민해진 피부 진정에 도움을 줄 수 있습니다.",
        "PDRN": "여드름 흉터 케어와 피부 재생에 도움을 줄 수 있어 트러블 진정 후 피부 결 관리에 활용하기 좋습니다.",
        "마데카소사이드": "트러블 진정과 흉터 관리에 도움을 줄 수 있어 여드름성 피부에 자주 활용되는 성분입니다.",
        "판테놀": "트러블로 예민해진 피부 진정에 도움을 줄 수 있어 여드름성 피부에 활용하기 좋습니다.",
        "아데노신": "트러블이 진정된 후 탄력·결 관리에 활용하기 좋은 성분입니다.",
        "콜라겐": "트러블 피부에도 활용할 수 있으나 가벼운 제형을 선택하는 것이 좋습니다.",
        "트라넥삼산": "여드름 자국 관리에 도움을 줄 수 있어 트러블 피부에 활용하기 좋습니다.",
        "갈락토미세스": "트러블 피부의 결 관리에 도움을 줄 수 있으나 상태에 따라 반응을 확인하는 것이 좋습니다.",
        "AHA": "여드름성 피부의 각질·트러블 관리에 도움을 줄 수 있지만 과사용 시 자극이 생길 수 있습니다.",
        "PHA": "트러블 피부의 부드러운 각질 관리에 활용하기 좋은 저자극 성분입니다.",
        "비타민C": "여드름 자국 관리에 도움을 줄 수 있지만 농도가 높으면 자극이 될 수 있어 주의가 필요합니다.",
        "펩타이드": "트러블이 진정된 후 결·탄력 관리에 활용하기 좋은 성분입니다.",
        "스쿠알란": "여드름성 피부도 사용할 수 있지만 무겁지 않은 제형을 선택하는 것이 좋습니다.",
        "알란토인": "트러블 피부 진정에 도움을 줄 수 있어 여드름성 피부에 활용하기 좋습니다.",
        "프로폴리스": "트러블 진정에 도움을 줄 수 있지만 알레르기 반응을 확인한 후 사용하는 것이 좋습니다.",
        "시어버터": "여드름성 피부에는 모공 부담이 될 수 있어 신중하게 사용해야 합니다."
    },
    "복합성": {
        "글리세린": "복합성 피부도 사용할 수 있으며, 건조한 부위에도 보습에 도움이 될 수 있습니다.",
        "히알루론산": "복합성 피부의 수분 밸런스 관리에 도움을 줄 수 있어 무난하게 사용할 수 있습니다.",
        "세라마이드": "피부 장벽 관리에 도움을 줄 수 있어 복합성 피부도 사용할 수 있습니다.",
        "나이아신아마이드": "피지와 피부 결 관리에 도움을 줄 수 있어 복합성 피부에 잘 맞는 편입니다.",
        "이리스산": "유분이 많은 부위의 각질과 피지 관리에 도움을 줄 수 있으나 건조한 부위에는 주의가 필요합니다.",
        "티트리오일": "트러블 부위에는 도움이 될 수 있지만 전체 얼굴에 고농도로 사용하는 것은 주의가 필요합니다.",
        "알코올": "유분 부위에는 산뜻하게 느껴질 수 있지만 건조한 부위에는 자극이 될 수 있습니다.",
        "향료": "피부가 예민한 부위에는 자극이 될 수 있어 주의가 필요합니다.",
        "레티놀": "피부 결 관리에 도움을 줄 수 있지만 건조한 부위에는 보습을 함께 해주는 것이 좋습니다.",
        "페녹시에탄올": "보존제로 사용되며 대체로 안전이지만, 예민한 부위에는 반응을 확인하는 것이 좋습니다.",
        "파라벤": "보존제로 사용되며 민감 반응이 있다면 피하는 것이 좋습니다.",
        "병풀추출물": "예민하거나 붉어진 부위 진정에 도움을 줄 수 있습니다.",
        "PDRN": "복합성 피부의 재생과 피부 결 관리에 무난하게 사용할 수 있는 성분입니다.",
        "마데카소사이드": "예민하거나 붉어진 부위 진정에 무난하게 사용할 수 있는 성분입니다.",
        "판테놀": "건조한 부위와 예민한 부위 모두 사용할 수 있는 무난한 보습·진정 성분입니다.",
        "아데노신": "복합성 피부의 탄력 관리에 무난하게 사용할 수 있는 성분입니다.",
        "콜라겐": "복합성 피부의 보습 균형 잡기에 무난하게 사용할 수 있는 성분입니다.",
        "트라넥삼산": "복합성 피부의 톤 개선과 진정에 무난하게 사용할 수 있는 성분입니다.",
        "갈락토미세스": "복합성 피부의 결과 톤 관리에 무난하게 사용할 수 있는 성분입니다.",
        "AHA": "유분 부위에는 사용할 수 있으나 건조한 부위에는 자극이 될 수 있어 부분 사용이 좋습니다.",
        "PHA": "복합성 피부의 각질 관리에 무난하게 사용할 수 있는 저자극 성분입니다.",
        "비타민C": "복합성 피부도 사용할 수 있으나 예민한 부위에는 반응을 확인하며 사용해야 합니다.",
        "펩타이드": "복합성 피부의 결과 탄력 관리에 무난하게 사용할 수 있는 성분입니다.",
        "스쿠알란": "복합성 피부의 보습 균형 잡기에 무난하게 사용할 수 있는 성분입니다.",
        "알란토인": "복합성 피부의 예민한 부위 진정에 무난하게 사용할 수 있는 성분입니다.",
        "프로폴리스": "복합성 피부의 진정과 장벽 관리에 무난하게 사용할 수 있는 성분입니다.",
        "시어버터": "건조한 부위에 부분적으로 사용하면 좋으며 유분 부위는 피하는 것이 좋습니다."
    }
}

# -------------------------------
# 5. 검색 함수
# -------------------------------
def normalize_text(text):
    return text.replace(" ", "").lower()


def find_ingredient(text):
    normalized = normalize_text(text)

    for ingredient, aliases in ingredient_aliases.items():
        for alias in aliases:
            if normalize_text(alias) in normalized:
                return ingredient

    for ingredient in ingredients_db.keys():
        if normalize_text(ingredient) in normalized:
            return ingredient

    return None


# (정규화된 별칭, 피부타입) 쌍을 별칭 길이 내림차순으로 정렬 — 1회 계산.
# "민지형"이 "수부민지형"의 부분문자열이라, 긴 별칭을 먼저 매칭해야 오인식을 막는다.
_SKIN_ALIAS_PAIRS = sorted(
    ((normalize_text(alias), skin_type)
     for skin_type, aliases in skin_aliases.items()
     for alias in aliases),
    key=lambda pair: len(pair[0]),
    reverse=True,
)


def find_skin_type_alias(text):
    """1단계: 사전(별칭) 기반 빠른 매칭. 긴 별칭부터 검사."""
    normalized = normalize_text(text)

    for alias, skin_type in _SKIN_ALIAS_PAIRS:
        if alias in normalized:
            return skin_type

    return None


def find_skin_type_semantic(text, embedding=None):
    """2단계: 임베딩 의미 매칭 (사전에 없는 표현도 잡음).
    짧은 입력은 임베딩이 불안정해 false positive 발생 → 4글자 미만은 건너뜀."""
    if len(text.strip()) < 4:
        return None

    if embedding is None:
        embedding = model.encode([text])

    best_skin_type = None
    best_score = -1.0

    for skin_type, embeddings in skin_type_phrase_embeddings.items():
        scores = cosine_similarity(embedding, embeddings)
        max_score = float(np.max(scores))
        if max_score > best_score:
            best_score = max_score
            best_skin_type = skin_type

    if best_score < SKIN_TYPE_SEMANTIC_THRESHOLD:
        return None

    return best_skin_type


def find_skin_type(text, embedding=None):
    """하이브리드: 사전 매칭 우선, 실패하면 임베딩 폴백."""
    alias_match = find_skin_type_alias(text)
    if alias_match:
        return alias_match
    return find_skin_type_semantic(text, embedding=embedding)


def find_category(text):
    normalized = normalize_text(text)

    for category, aliases in category_aliases.items():
        for alias in aliases:
            if normalize_text(alias) in normalized:
                return category

    return None


GREETING_PATTERNS = {
    "안녕", "안녕하세요", "안녕하십니까", "하이", "헬로", "헬로우",
    "hi", "hello", "hey", "반가워", "반갑습니다",
}


def is_greeting(text):
    """짧은 단독 인사말이면 True. '안녕하세요 토너 추천' 같이 인사+요청 섞이면 False."""
    cleaned = re.sub(r"[!?.,~\s]+", "", text).lower()
    if not cleaned:
        return False
    return cleaned in {p.replace(" ", "").lower() for p in GREETING_PATTERNS}


def build_greeting_response():
    """간단한 인사 + 무엇을 할 수 있는지 짧게 안내."""
    return {
        "intent": "GREETING",
        "score": 1.0,
        "keywords": {},
        "message": (
            "안녕하세요! 더마렌즈 챗봇이에요 ✨\n"
            "성분 분석·제품 추천·피부 진단을 도와드릴 수 있어요. "
            "아래 버튼을 누르거나 자유롭게 물어보세요!"
        ),
        "components": [],
        "quickReplies": ["제품 추천", "성분 분석", "피부 진단", "메뉴"]
    }


# -------------------------------
# 전체 메뉴 — 첫 화면 "메뉴" 버튼 클릭 시 모든 페이지 바로가기를 칩으로 펼침
# 칩 라벨(보기 좋은 이름) → detect_page_request가 인식하는 키워드 매핑은 프론트 quickMap에서 처리
# -------------------------------
MENU_KEYWORDS = {"메뉴", "전체 메뉴", "전체 기능", "기능 보기", "메뉴 보기", "메뉴 보여줘", "전체보기"}

# 메뉴에 노출할 페이지 칩 라벨 (PAGES 키 순서대로)
MENU_ITEMS = [
    "제품 페이지", "바르는 루틴", "성분 사진 분석(OCR)", "피부 타입 진단",
    "리뷰 페이지", "알레르기 관리", "제품 등록", "문의하기",
    "제품 신고", "앱 만족도 평가",
]


def is_menu_request(text):
    """'메뉴', '전체 기능' 등 전체 메뉴를 요청하는 입력이면 True."""
    normalized = normalize_text(text)
    return any(normalize_text(k) in normalized for k in MENU_KEYWORDS)


def build_menu_response():
    """전체 기능(페이지) 바로가기 메뉴 응답."""
    return {
        "intent": "MENU",
        "score": 1.0,
        "keywords": {},
        "message": "DermaLens에서 이용할 수 있는 기능이에요. 원하는 곳으로 안내해드릴게요! 🧭",
        "components": [],
        "quickReplies": MENU_ITEMS,
    }


def detect_page_request(text):
    """사용자 입력에 특정 페이지 키워드가 있으면 (page_id, page_info) 반환.
    예: '리뷰 페이지 보여줘' → ('review', {...})"""
    normalized = normalize_text(text)
    for page_id, info in PAGES.items():
        for kw in info["keywords"]:
            if normalize_text(kw) in normalized:
                return page_id, info
    return None, None


def build_page_response(page_info):
    """페이지 안내 응답 생성 — 링크 카드 포함."""
    return {
        "intent": "PAGE_LINK",
        "score": 1.0,
        "keywords": {},
        "message": f"{page_info['title']} 페이지로 안내해드릴게요. 아래 버튼을 눌러주세요.",
        "components": [
            {
                "type": "link",
                "label": page_info["label"],
                "url": page_info["url"]
            }
        ],
        "quickReplies": ["제품 추천", "성분 분석", "피부 진단"]
    }


def classify_intent(user_input, embedding=None):
    if embedding is None:
        embedding = model.encode([user_input])

    best_intent = None
    best_score = -1.0

    for intent, embeddings in intent_embeddings.items():
        scores = cosine_similarity(embedding, embeddings)
        max_score = float(np.max(scores))

        if max_score > best_score:
            best_score = max_score
            best_intent = intent

    if best_score < 0.4:
        return "UNKNOWN", best_score

    return best_intent, best_score


def extract_keywords(text, embedding=None):
    result = {}

    ingredient = find_ingredient(text)
    skin_type = find_skin_type(text, embedding=embedding)
    category = find_category(text)

    if ingredient:
        result["ingredient"] = ingredient
    if skin_type:
        result["skin_type"] = skin_type
    if category:
        result["category"] = category

    return result


def recommend_products(keywords):
    results = []
    # 로컬 더미 제품은 5타입 태그라 8타입은 코멘트 키(지성/건성/민감성)로 변환해 매칭
    target = keywords.get("skin_type")
    if target:
        target = SKIN_COMMENT_KEY.get(target, target)
    for product in products_db:
        skin_match = target is None or product["skin_type"] == target
        category_match = "category" not in keywords or product["category"] == keywords["category"]
        if skin_match and category_match:
            results.append(product)
    return results


def get_skin_ingredient_comment(skin_type, ingredient):
    if not skin_type:
        return ""

    # 8타입(민지형 등)은 코멘트 사전 키(지성/건성/민감성)로 변환해서 조회
    comment_key = SKIN_COMMENT_KEY.get(skin_type, skin_type)
    comment = skin_ingredient_comments.get(comment_key, {}).get(ingredient)
    if comment:
        return " " + comment

    return f" {skin_type} 피부라면 처음 사용할 때 소량 테스트 후 사용하는 것이 좋습니다."


def build_flag_description(card):
    """백엔드 카드의 플래그(보습/진정/자극 등)로 한글 설명 자동 생성.
    백엔드 description이 영문 학명만 있을 때 보강용.
    예: moisturizing_flag + soothing_flag → "이 성분은 보습, 진정 효과가 있어요."
    """
    positive = []
    warning = []
    if card.get("moisturizing_flag"):
        positive.append("보습")
    if card.get("soothing_flag"):
        positive.append("진정")
    if card.get("allergy_flag"):
        warning.append("알레르기")
    if card.get("irritant_flag"):
        warning.append("자극")
    if card.get("acne_caution_flag"):
        warning.append("여드름")

    parts = []
    if positive:
        parts.append(f"{', '.join(positive)} 효과가 있어요")
    if warning:
        parts.append(f"{', '.join(warning)} 가능성이 있어 주의가 필요해요")

    if parts:
        return "이 성분은 " + ", ".join(parts) + "."
    return None


# 카테고리별 식별 키워드 (제품명에서 카테고리 판별용)
# 백엔드가 가끔 다른 카테고리 제품을 섞어 보내는 경우 필터링에 사용
CATEGORY_KEYWORDS = {
    "토너": ["토너", "스킨"],
    "에센스": ["에센스"],
    "세럼": ["세럼"],
    "앰플": ["앰플"],
    "크림": ["크림"],
    "로션": ["로션", "에멀젼"],
}


def filter_cards_by_category(cards, requested_category):
    """제품명에 다른 카테고리 키워드가 명시된 카드 제외.
    예: 토너 요청인데 제품명에 "에센스"/"앰플" 들어있으면 제외.
    필터 후 0개면 원본 그대로 반환 (빈 화면 방지)."""
    if not requested_category or not cards:
        return cards

    # 요청 카테고리 외의 다른 카테고리 키워드들 수집
    other_keywords = set()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if cat == requested_category:
            continue
        other_keywords.update(kws)

    filtered = []
    for card in cards:
        title = card.get("title", "")
        # 제품명에 다른 카테고리 키워드 명시되어 있으면 제외
        has_other = any(kw in title for kw in other_keywords)
        if not has_other:
            filtered.append(card)

    # 필터 결과 0개면 원본 유지 (사용자가 빈 화면 보는 것보단 나음)
    return filtered if filtered else cards


def build_recommend_message(keywords):
    # skin_type / category 유무에 따라 자연스러운 메시지 생성
    skin = keywords.get("skin_type")
    category = keywords.get("category")

    if skin and category:
        return f"{skin} 피부에 맞는 {category} 제품을 추천해드릴게요."
    if skin:
        return f"{skin} 피부에 맞는 제품을 추천해드릴게요."
    if category:
        return f"추천드릴 만한 {category} 제품을 찾아봤어요."
    return "조건에 맞는 제품을 추천해드릴게요."


def generate_response(intent, keywords, user_input=""):
    if intent == "PRODUCT_RECOMMEND":
        # 피부 타입을 모르면 먼저 물어봄 (KT 챗봇 스타일 단계별 안내)
        if "skin_type" not in keywords:
            if "category" in keywords:
                ask_msg = f"어떤 피부 타입에 맞는 {keywords['category']}을(를) 추천해드릴까요?"
            else:
                ask_msg = "어떤 피부 타입에 맞는 제품을 추천해드릴까요? 아래에서 선택해주세요."
            return {
                "intent": intent,
                "message": ask_msg,
                "components": [],
                "quickReplies": [
                    "민지형(OS+)", "수부민지형(OS-)", "건지형(ON+)", "수부지형(ON-)",
                    "민건형(DS+)", "극건민감형(DS-)", "건강건성형(DN+)", "수부건형(DN-)",
                    "잘 모르겠어요"
                ]
            }

        products = recommend_products(keywords)

        if not products:
            return {
                "intent": intent,
                "message": "조건에 맞는 추천 제품을 찾지 못했습니다. 피부 타입이나 제품 종류를 조금 더 구체적으로 입력해주세요.",
                "components": [],
                "quickReplies": ["민감성 크림 추천", "건성 크림 추천", "지성 토너 추천"]
            }

        return {
            "intent": intent,
            "message": build_recommend_message(keywords),
            "components": [
                {
                    "type": "card",
                    "title": p["name"],
                    "description": p["description"],
                    "riskLevel": p["riskLevel"],
                    "buttonText": "제품 상세보기"
                }
                for p in products
            ],
            "quickReplies": ["성분 분석", "피부 진단", "여드름 제품 추천"]
        }

    if intent == "INGREDIENT_RISK":
        ingredient = keywords.get("ingredient")
        skin_type = keywords.get("skin_type")

        if not ingredient:
            return {
                "intent": intent,
                "message": "확인하고 싶은 성분명을 입력해주세요. 예를 들어 '히알루론산 알려줘'처럼 입력하면 됩니다.",
                "components": [],
                "quickReplies": ["히알루론산 알려줘", "PDRN 알려줘", "마데카소사이드 어때?", "비타민C 위험해?"]
            }

        data = ingredients_db.get(ingredient)

        if not data:
            # DB에 없는 성분 — 알려진 성분 중 비슷한 것 제안
            known = sorted(ingredients_db.keys())[:6]
            return {
                "intent": intent,
                "message": f"'{ingredient}' 성분 정보가 아직 등록되어 있지 않아요. 아래 등록된 성분 중에서 골라보시거나, 정확한 성분명으로 다시 입력해주세요.",
                "components": [],
                "quickReplies": known + ["다른 성분"]
            }

        skin_comment = get_skin_ingredient_comment(skin_type, ingredient)
        risk_ko = RISK_LABEL_KO.get(data["risk"], data["risk"])

        if skin_type:
            message = f"{ingredient}은/는 {skin_type} 피부 기준으로 위험도는 {risk_ko}입니다. {data['description']}{skin_comment}"
        else:
            message = f"{ingredient}의 위험도는 {risk_ko}입니다. {data['description']} 특정 피부 타입이 있다면 함께 입력하면 더 자세히 안내할 수 있습니다."

        # 성분 정보도 카드로 함께 반환 (PDF 응답 일관성)
        return {
            "intent": intent,
            "message": message,
            "components": [
                {
                    "type": "card",
                    "title": ingredient,
                    "description": data["description"],
                    "riskLevel": data["risk"],
                    "buttonText": "성분 자세히 보기"
                }
            ],
            "quickReplies": ["제품 추천", "다른 성분 확인", "피부 진단"]
        }

    if intent == "INGREDIENT_ANALYSIS":
        text = user_input or ""

        # 이미지/OCR 요청
        if any(kw in text for kw in ["사진", "이미지", "OCR", "ocr"]):
            return {
                "intent": intent,
                "message": "성분표 이미지 업로드 기능은 곧 추가될 예정입니다. 지금은 성분명을 직접 입력하거나 아래 버튼에서 선택해주세요.",
                "components": [],
                "quickReplies": ["글리세린", "히알루론산", "알코올", "다시 처음으로"]
            }

        # 다른 성분 보여달라는 요청 (페이지 2)
        if "다른 성분" in text or "다른성분" in text:
            return {
                "intent": intent,
                "message": "다른 성분들도 확인해보세요.",
                "components": [],
                "quickReplies": ["마데카소사이드", "판테놀", "아데노신", "콜라겐", "트라넥삼산", "AHA", "PHA", "비타민C", "펩타이드", "스쿠알란", "프로폴리스", "다시 처음으로"]
            }

        # 기본: 어떤 성분 묻기 — DB에 있는 인기 성분 위주로
        return {
            "intent": intent,
            "message": "어떤 성분이 궁금하세요? 아래에서 선택하거나 직접 입력해주세요.",
            "components": [],
            "quickReplies": ["히알루론산", "나이아신아마이드", "PDRN", "마데카소사이드", "판테놀", "비타민C", "레티놀", "다른 성분", "이미지로 분석"]
        }

    if intent == "SKIN_TYPE_TEST":
        return {
            "intent": intent,
            "message": "피부 타입 진단은 진단 페이지에서 진행해드릴게요. 아래 버튼을 눌러 시작해보세요.",
            "components": [
                {
                    "type": "link",
                    "label": "피부 진단 시작하기",
                    "url": DIAGNOSE_URL
                }
            ],
            "quickReplies": ["제품 추천", "성분 분석"]
        }

    if intent == "REVIEW_SUMMARY":
        return {
            "intent": intent,
            "message": "제품 리뷰 데이터를 바탕으로 장점과 단점을 요약해드릴 수 있습니다. 현재 시연 버전에서는 리뷰 DB 연결 전 단계입니다.",
            "components": [],
            "quickReplies": ["제품 추천", "민감성 크림 추천"]
        }

    if intent == "ANALYSIS_HISTORY":
        return {
            "intent": intent,
            "message": "마이페이지에서 이전 분석 기록을 확인할 수 있습니다. 현재 시연 버전에서는 분석 이력 DB 연결 전 단계입니다.",
            "components": [],
            "quickReplies": ["최근 분석 보기", "성분 분석"]
        }

    # UNKNOWN 처리 — KT 챗봇 스타일 친절 응답 + FAQ 분기
    text = user_input or ""

    if any(kw in text for kw in ["자주", "FAQ", "faq", "도움말", "물어볼"]):
        return {
            "intent": "FAQ",
            "message": "고객님들이 자주 묻는 질문과 답변을 모아봤어요.\n궁금하신 내용을 구체적으로 질문해 주시면 더 정확한 답을 찾아볼게요!",
            "components": [],
            "quickReplies": [
                "민감성 크림 추천",
                "글리세린 위험해?",
                "페녹시에탄올 위험해?",
                "피부 타입 진단"
            ]
        }

    return {
        "intent": "UNKNOWN",
        "message": "제가 잘 이해하지 못했어요.\nDermaLens는 피부 분석, 성분 위험도, 제품 추천을 도와드릴 수 있어요.\n\n▶ 아래 [자주 묻는 질문]을 통해 어떤 질문을 할 수 있는지 확인해 보세요.",
        "components": [],
        "quickReplies": ["자주 묻는 질문", "제품 추천", "성분 분석", "피부 진단"]
    }


def build_suggestion_pool():
    """자동완성에 사용할 모든 가능한 사용자 질문 풀.
    성분/피부타입/카테고리 조합 + FAQ 200개 + 페이지 키워드.
    챗봇 시작 시 1회 계산."""
    pool = set()

    # 성분별 질문 (각 성분 × 기본 문형 3종)
    for ing in ingredients_db.keys():
        pool.add(f"{ing} 위험해?")
        pool.add(f"{ing} 알려줘")
        pool.add(f"{ing} 어때?")

    # 피부타입 + 카테고리 추천 조합
    # 5개 카테고리(토너/에센스/세럼/앰플/로션) + 크림 모두 커버
    # 마스터데이터 8타입(민지형 등) + 기존 5타입 모두 자동완성에 포함
    skin_names = ["민감성", "건성", "지성", "복합성", "여드름성"] + \
        [info["kr"] for info in SKIN_TYPES_8.values()]
    for skin in skin_names:
        pool.add(f"{skin} 피부 추천")
        for cat in ["크림", "토너", "에센스", "세럼", "앰플", "로션"]:
            pool.add(f"{skin} {cat} 추천")
            pool.add(f"{skin}피부인데 {cat} 추천좀")  # 회화체

    # 카테고리 단독 추천
    for cat in ["크림", "토너", "에센스", "세럼", "앰플", "로션"]:
        pool.add(f"{cat} 추천해줘")

    # FAQ 질문 200개
    for item in faq_bank:
        pool.add(item["question"])

    # 페이지 키워드 + 타이틀
    for info in PAGES.values():
        pool.add(info["title"])
        for kw in info["keywords"]:
            pool.add(kw)

    # 의도 분류 예시도 일부 포함 (추천 다양성)
    for examples in intent_examples.values():
        for ex in examples:
            pool.add(ex)

    return sorted(pool)


SUGGESTION_POOL = build_suggestion_pool()
print(f"[자동완성] 추천 후보 {len(SUGGESTION_POOL)}개 로드 완료")


def call_backend(user_id, message, intent, keywords):
    """백엔드 분석 API 호출. 성공 시 응답 dict, 실패/미설정 시 None.
    실패 시 호출부가 더미 데이터로 자동 폴백."""
    if not BACKEND_URL:
        return None
    try:
        resp = requests.post(
            BACKEND_URL,
            json={
                "user_id": user_id,
                "message": message,
                "intent": intent,
                "keywords": keywords,
            },
            timeout=BACKEND_TIMEOUT,
        )
        if resp.ok:
            return resp.json()
        print(f"[Backend] HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"[Backend] 호출 실패, 더미 데이터로 폴백: {e}")
    return None


@app.route("/chat", methods=["POST"])
def chat():
    # FIX: 요청 본문 방어
    data = request.get_json(silent=True) or {}
    user_input = (data.get("message") or "").strip()
    user_id = data.get("user_id")  # 선택 (백엔드 로그용)

    if not user_input:
        return jsonify({
            "intent": "UNKNOWN",
            "score": 0.0,
            "keywords": {},
            "message": "메시지를 입력해주세요.",
            "components": [],
            "quickReplies": ["제품 추천", "성분 분석", "피부 진단"]
        })

    # 자모만 있거나 기호만 있는 무의미 입력은 분류기·백엔드 모두 건너뛰고 UNKNOWN
    if not is_meaningful_text(user_input):
        response = generate_response("UNKNOWN", {}, user_input=user_input)
        response["score"] = 0.0
        response["keywords"] = {}
        return jsonify(response)

    # 인사말 감지 — 의도분류·백엔드 안 거치고 바로 응답
    # 예: "안녕", "안녕하세요", "hi", "hello"
    if is_greeting(user_input):
        return jsonify(build_greeting_response())

    # 전체 메뉴 요청 — 페이지 감지보다 먼저(모든 페이지 바로가기 칩 노출)
    if is_menu_request(user_input):
        return jsonify(build_menu_response())

    # 페이지 요청 감지 — 특정 페이지 키워드("리뷰 페이지", "OCR", "알레르기 등록" 등)
    # 가장 먼저 처리해서 의도분류나 백엔드 안 거치고 바로 링크 안내
    page_id, page_info = detect_page_request(user_input)
    if page_id:
        return jsonify(build_page_response(page_info))

    # 임베딩 1회만 계산해서 FAQ + 의도분류 + 피부타입 의미 매칭에 재사용
    user_embedding = model.encode([user_input])

    keywords = extract_keywords(user_input, embedding=user_embedding)

    # 성분명 오타 보정 — alias 매칭이 실패했을 때만 fuzzy 시도
    # 예: "글리세인 위험해?" → "글리세린"으로 보정
    typo_note = ""
    if "ingredient" not in keywords:
        correction = fuzzy_correct_ingredient(user_input)
        if correction:
            keywords["ingredient"] = correction["corrected"]
            typo_note = f"('{correction['original']}'을(를) '{correction['corrected']}'(으)로 인식했어요) "

    # 사용법·상식 질문 패턴이면 FAQ 우선 시도
    # 예: "로션은 언제 발라야해?", "토너 어떻게 써?", "비타민C 얼마나 자주?"
    # 키워드(로션/토너 등)에 휘둘리지 않게 FAQ 먼저 매칭
    if is_usage_question(user_input):
        faq_match = find_faq_match(user_input, embedding=user_embedding)
        if faq_match:
            return jsonify({
                "intent": "FAQ",
                "score": round(faq_match["score"], 3),
                "keywords": {"category": faq_match["category"]},
                "message": faq_match["answer"],
                "components": [],
                "quickReplies": ["제품 추천", "성분 분석", "피부 진단"]
            })

    # FIX: 분류기 결과를 우선 신뢰하고, 신뢰도가 낮을 때만 키워드 폴백
    intent, score = classify_intent(user_input, embedding=user_embedding)

    if intent == "UNKNOWN" or score < 0.5:
        if "ingredient" in keywords:
            intent = "INGREDIENT_RISK"
            score = max(score, 0.9)
        elif "skin_type" in keywords or "category" in keywords:
            intent = "PRODUCT_RECOMMEND"
            score = max(score, 0.9)

    # 추천 신호("추천")가 있고 피부타입/카테고리 키워드가 잡히면 PRODUCT_RECOMMEND로 보정.
    # "민건형 피부 제품 추천해줘"처럼 생소한 타입명이 SKIN_TYPE_TEST로 오분류되는 것 방지
    # (점수가 0.5보다 높아 위 폴백이 안 걸리는 경우까지 커버)
    if "추천" in user_input and ("skin_type" in keywords or "category" in keywords):
        intent = "PRODUCT_RECOMMEND"
        score = max(score, 0.9)

    # 키워드도 없고 신뢰도도 애매하면(0.7 미만) → 명사 후보 추출 시도
    # "알로에", "녹차" 같이 짧은 성분명 단독 입력도 백엔드로 보냄
    # 추출 실패하면 UNKNOWN ("뭐야", "넌", 빈 토큰은 후보 없음)
    if not keywords and score < 0.7:
        candidate = extract_ingredient_candidate(user_input)
        if candidate:
            keywords["ingredient"] = candidate
            intent = "INGREDIENT_RISK"
            score = max(score, 0.5)
        else:
            intent = "UNKNOWN"

    # 백엔드 호출 대상 — DB 조회가 필요한 의도만
    # SKIN_TYPE_TEST·INGREDIENT_ANALYSIS·REVIEW_SUMMARY·ANALYSIS_HISTORY는
    # 챗봇 UX 흐름(링크 버튼·성분 선택·안내)이라 백엔드 안 거침
    BACKEND_INTENTS = {"PRODUCT_RECOMMEND", "INGREDIENT_RISK"}

    # 제품 추천인데 피부타입 모르면 백엔드 호출 전에 먼저 물어봄
    # ("제품 추천해줘"만 입력 시 → "어떤 피부 타입?" 선택 칩)
    if intent == "PRODUCT_RECOMMEND" and "skin_type" not in keywords:
        response = generate_response(intent, keywords, user_input=user_input)
        response["score"] = round(float(score), 3)
        response["keywords"] = keywords
        if typo_note and "message" in response:
            response["message"] = typo_note + response["message"]
        return jsonify(response)

    # 성분 위험도인데 챗봇 사전(27개)에서 성분 못 찾았으면
    # 메시지에서 명사 후보 추출해서 백엔드(21,810개 DB)로 넘김
    # 백엔드가 부분일치 검색해서 처리 (예: "알로에" → "알로에베라잎추출물")
    if intent == "INGREDIENT_RISK" and "ingredient" not in keywords:
        candidate = extract_ingredient_candidate(user_input)
        if candidate:
            # 후보 찾음 → 키워드 설정하고 백엔드 호출 흐름으로 계속 진행
            keywords["ingredient"] = candidate
        else:
            # 후보도 못 찾으면 "어떤 성분?" 물어봄
            response = generate_response(intent, keywords, user_input=user_input)
            response["score"] = round(float(score), 3)
            response["keywords"] = keywords
            if typo_note and "message" in response:
                response["message"] = typo_note + response["message"]
            return jsonify(response)

    if intent in BACKEND_INTENTS:
        # 백엔드 호출용 키워드 변환 — skin_type만 매핑(복합성→"건성 지성" 등)
        # 사용자 응답에는 원본 keywords 유지 (혼란 방지)
        backend_keywords = dict(keywords)
        if "skin_type" in backend_keywords:
            backend_keywords["skin_type"] = SKIN_TYPE_BACKEND_MAP.get(
                backend_keywords["skin_type"], backend_keywords["skin_type"]
            )

        backend_resp = call_backend(user_id, user_input, intent, backend_keywords)
        if backend_resp is not None:
            ingredient = keywords.get("ingredient")

            # [0-A] 제품 추천 — 카테고리 필터링 (다른 카테고리 제품 섞임 방지)
            # 예: 토너 요청인데 "스킨푸드 ... 에센스" 같은 제품 섞여 옴 → 제외
            if intent == "PRODUCT_RECOMMEND" and keywords.get("category"):
                backend_resp["components"] = filter_cards_by_category(
                    backend_resp.get("components", []),
                    keywords["category"]
                )

            # [0-B] 제품 추천인데 빈 결과 반환 (해당 카테고리 데이터 없음)
            # 예: 토너·세럼은 백엔드 DB에 미등록 → 사용자한테 명확히 안내
            if (intent == "PRODUCT_RECOMMEND"
                    and not backend_resp.get("components")):
                cat = keywords.get("category", "해당 카테고리")
                skin = keywords.get("skin_type", "")
                prefix = f"{skin} 피부용 " if skin else ""
                return jsonify({
                    "intent": intent,
                    "score": round(float(score), 3),
                    "keywords": keywords,
                    "message": (
                        f"{prefix}{cat} 제품은 아직 DermaLens DB에 등록 중이에요. "
                        f"현재는 크림·로션 카테고리를 우선 지원합니다."
                    ),
                    "components": [],
                    "quickReplies": [
                        f"{skin} 크림 추천" if skin else "민감성 크림 추천",
                        f"{skin} 로션 추천" if skin else "건성 로션 추천",
                        "성분 분석", "피부 진단"
                    ]
                })

            # [1] 백엔드 not_found + 챗봇 27개에 있음 → 챗봇 답변으로 폴백
            if (backend_resp.get("not_found")
                    and intent == "INGREDIENT_RISK"
                    and ingredient in ingredients_db):
                response = generate_response(intent, keywords, user_input=user_input)
                response["score"] = round(float(score), 3)
                response["keywords"] = keywords
                if typo_note and "message" in response:
                    response["message"] = typo_note + response["message"]
                return jsonify(response)

            # [2] 안전망: 백엔드가 일부 필드 빠뜨려도 챗봇 NLP 결과로 채움
            backend_resp.setdefault("intent", intent)
            backend_resp.setdefault("score", round(float(score), 3))
            backend_resp.setdefault("keywords", keywords)

            if backend_resp.get("not_found"):
                # 둘 다 모르는 성분 — UX 보강
                backend_resp.setdefault("components", [])
                if not backend_resp.get("quickReplies"):
                    popular = ["히알루론산", "나이아신아마이드", "PDRN",
                               "마데카소사이드", "비타민C", "레티놀"]
                    backend_resp["quickReplies"] = popular + ["다른 성분", "성분 분석"]
            else:
                # [3] 백엔드 정상 응답 → 카드·메시지 보강
                # 성분 카드 description 보강은 INGREDIENT_RISK만 (제품 카드는 그대로)
                # 우선순위: 백엔드 한글 > 챗봇 27개 한글 > 플래그 자동 > placeholder
                if intent == "INGREDIENT_RISK" and backend_resp.get("components"):
                    for card in backend_resp["components"]:
                        if card.get("type") != "card":
                            continue
                        desc = (card.get("description") or "").strip()
                        has_korean = bool(KOREAN_SYLLABLE_RE.search(desc))

                        if has_korean:
                            # 백엔드가 한글 설명 줌 — 그대로 사용 ✓
                            pass
                        elif ingredient in ingredients_db:
                            # 백엔드 비어있음 → 챗봇 27개에서 가져옴
                            chatbot_data = ingredients_db[ingredient]
                            card["description"] = chatbot_data["description"]
                            # 위험도도 챗봇 게 더 정확 (백엔드는 거의 다 LOW)
                            card["riskLevel"] = chatbot_data["risk"]
                        else:
                            # 챗봇도 모름 → 플래그로 한 줄 자동 생성
                            korean_desc = build_flag_description(card)
                            if korean_desc:
                                card["description"] = korean_desc
                            else:
                                # 백엔드 description: 461/21,810만 채워짐 (2.1%)
                                # 마이너 성분은 이름만 있고 효능 데이터 비어 있음
                                name_en = (card.get("name_en") or "").strip()
                                if name_en:
                                    card["description"] = (
                                        f"국제명: {name_en}\n"
                                        "상세 효능 정보는 준비 중이에요. "
                                        "사용 전 패치 테스트를 권장드려요."
                                    )
                                else:
                                    card["description"] = (
                                        "상세 효능 정보는 준비 중이에요. "
                                        "사용 전 패치 테스트를 권장드려요."
                                    )

                # 메시지 보강 — 의도별로 다르게 구성
                if (intent == "INGREDIENT_RISK"
                        and keywords.get("skin_type")
                        and ingredient in ingredients_db):
                    # 성분 + 피부타입 + 챗봇 27개 → 피부타입 코멘트 메시지
                    ing = keywords["ingredient"]
                    skin = keywords["skin_type"]
                    data = ingredients_db[ing]
                    risk_ko = RISK_LABEL_KO.get(data["risk"], data["risk"])
                    skin_comment = get_skin_ingredient_comment(skin, ing)
                    backend_resp["message"] = (
                        f"{ing}은/는 {skin} 피부 기준으로 위험도는 {risk_ko}입니다."
                        f"{skin_comment}"
                    )
                else:
                    # 백엔드 메시지가 echo이거나 한글 없으면 의도별로 재구성
                    msg = backend_resp.get("message", "")
                    if msg == user_input or not KOREAN_SYLLABLE_RE.search(msg):
                        if intent == "PRODUCT_RECOMMEND":
                            # 제품 추천 — 피부타입·카테고리 기반 메시지
                            backend_resp["message"] = build_recommend_message(keywords)
                        elif backend_resp.get("components"):
                            # 성분 카드 — title 활용
                            title = backend_resp["components"][0].get("title", "")
                            if title:
                                backend_resp["message"] = (
                                    f"{title}에 대한 정보를 안내해드릴게요."
                                )

            # 오타 보정 알림을 메시지 앞에 안내
            if typo_note and "message" in backend_resp:
                backend_resp["message"] = typo_note + backend_resp["message"]
            return jsonify(backend_resp)

    # FAQ 폴백: UNKNOWN이거나, 의도 신뢰도가 낮고 키워드도 없는 경우
    # ("기초 화장품 바르는 순서"처럼 표현이 살짝 달라도 FAQ에 잡히게)
    # 단, 강한 의도 매칭(score > 0.7 또는 키워드 있음)은 FAQ에 뺏기지 않음
    should_try_faq = (intent == "UNKNOWN") or (score < 0.7 and not keywords)
    if should_try_faq:
        faq_match = find_faq_match(user_input, embedding=user_embedding)
        if faq_match:
            return jsonify({
                "intent": "FAQ",
                "score": round(faq_match["score"], 3),
                "keywords": {"category": faq_match["category"]},
                "message": faq_match["answer"],
                "components": [],
                "quickReplies": ["제품 추천", "성분 분석", "피부 진단"]
            })

    # 최종 폴백: 백엔드 실패 / FAQ도 매칭 안 됨 → 챗봇의 안내 응답
    response = generate_response(intent, keywords, user_input=user_input)
    response["score"] = round(float(score), 3)
    response["keywords"] = keywords
    if typo_note and "message" in response:
        response["message"] = typo_note + response["message"]

    return jsonify(response)


# 성분명처럼 보이는 입력에 자동으로 붙여줄 질문 템플릿
# 챗봇 27개 풀에 없는 성분(예: "알로에", "꿀")이라도 이걸로 즉석 제안 가능
# 피부타입 조합("X 민감성 피부에 어때?")은 빼둠 — 백엔드 21,810개 중
# 코멘트가 있는 건 27개뿐이라 일관된 답변이 어려움
SUGGESTION_TEMPLATES = [
    "{name} 알려줘",
    "{name} 위험해?",
    "{name} 어때?",
]


def is_ingredient_like_query(q):
    """입력이 성분명 같은 형태인지 판단.
    - 한글 2자 이상 또는 영문 단일 토큰
    - 불용어 아님 (성분, 피부, 추천 등)
    - 공백·구두점 없음 (조합 질문 아님)
    """
    if not q or len(q) < 2:
        return False
    if not re.match(r'^[가-힣a-zA-Z0-9]+$', q):
        return False
    if q in INGREDIENT_STOPWORDS:
        return False
    return True


@lru_cache(maxsize=512)
def _fetch_backend_ingredient_names_cached(query, limit):
    """같은 (query, limit) 두 번째 호출부터 백엔드 안 거치고 캐시 즉시 반환.
    튜플로 캐싱 (리스트는 hashable 아님)."""
    if not INGREDIENT_SEARCH_URL or not query:
        return ()
    try:
        resp = requests.get(
            INGREDIENT_SEARCH_URL,
            params={"q": query, "limit": limit},
            timeout=INGREDIENT_SEARCH_TIMEOUT,
        )
        if resp.ok:
            data = resp.json()
            names = data.get("ingredients", [])
            return tuple(n for n in names if isinstance(n, str) and n.strip())
    except Exception as e:
        print(f"[Suggest] 백엔드 검색 실패: {e}")
    return ()


def fetch_backend_ingredient_names(query, limit=5):
    """백엔드 성분 검색 API 호출 → 매칭되는 성분명 배열 반환.
    실패하면 빈 리스트 (자동완성 흐름 막지 않게).
    같은 검색어는 LRU 캐시로 즉시 반환 (Railway RTT 200~800ms 회피).

    예: query="정제", limit=5
        → ["정제수", "정제염", "정제벌꿀", ...]
    """
    return list(_fetch_backend_ingredient_names_cached(query, limit))


@app.route("/suggest", methods=["GET"])
def suggest():
    """자동완성 추천 — KB 챗봇 스타일.

    검색 우선순위:
    1) 챗봇 로컬 풀 (성분 27개 × 3 + FAQ + 페이지) — 빠름
    2) 백엔드 DB 검색 (21,810개 부분일치) — 진짜 등록된 성분명
    3) 사용자 입력 그대로 템플릿 (백엔드 실패 시 폴백)

    예: q=정제
        → 백엔드 검색 → "정제수", "정제염"
        → "정제수 알려줘", "정제염 알려줘"로 자동완성
    """
    raw = (request.args.get("q") or "").strip()
    if not raw:
        return jsonify({"suggestions": []})
    query = raw.lower()

    # 1단계: 로컬 풀 매칭 (FAQ·페이지·챗봇 27개 성분 등)
    prefix_matches = []
    contains_matches = []
    for item in SUGGESTION_POOL:
        item_lower = item.lower()
        if item_lower.startswith(query):
            prefix_matches.append(item)
        elif query in item_lower:
            contains_matches.append(item)

    suggestions = prefix_matches + contains_matches

    # 2단계: 백엔드 DB에서 실제 성분명 검색 → "이름 알려줘" 형태로 추가
    # 21,810개 모두 커버. 짧고 깔끔한 이름 우선.
    if len(suggestions) < 8 and is_ingredient_like_query(raw):
        backend_names = fetch_backend_ingredient_names(raw, limit=10)
        # 짧은 이름 우선 정렬 (긴 화학 복합명은 뒤로) + prefix 매칭 우선
        backend_names.sort(key=lambda n: (
            0 if n.startswith(raw) else 1,  # prefix 매칭 우선
            len(n)                            # 짧은 이름 우선
        ))
        # 너무 긴 이름(>30자 복합 INCI)은 자동완성에 부담 → 제외
        backend_names = [n for n in backend_names if len(n) <= 30]

        for name in backend_names:
            if len(suggestions) >= 8:
                break
            generated = f"{name} 알려줘"
            if generated not in suggestions:
                suggestions.append(generated)

    # 3단계: 그래도 부족하면 사용자 입력 그대로 템플릿 (백엔드 결과 없을 때 폴백)
    if len(suggestions) < 8 and is_ingredient_like_query(raw):
        for tpl in SUGGESTION_TEMPLATES:
            if len(suggestions) >= 8:
                break
            generated = tpl.format(name=raw)
            if generated not in suggestions:
                suggestions.append(generated)

    return jsonify({"suggestions": suggestions[:8]})


@app.route("/")
def home():
    return render_template_string("""
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>DermaLens Chatbot</title>
<style>
    * { box-sizing: border-box; }
    body {
        margin: 0;
        font-family: Arial, sans-serif;
        background: #f3f6fb;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }
    .phone {
        width: 390px;
        height: 720px;
        background: #ffffff;
        border-radius: 28px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        overflow: hidden;
        display: flex;
        flex-direction: column;
    }
    .header {
        background: linear-gradient(135deg, #8ab4ff, #b9e3ff);
        color: white;
        padding: 22px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
    }
    .header h2 { margin: 0; font-size: 22px; }
    .header p { margin: 6px 0 0; font-size: 13px; opacity: 0.9; }
    .home-btn {
        flex-shrink: 0;
        background: rgba(255, 255, 255, 0.25);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 16px;
        padding: 7px 12px;
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        white-space: nowrap;
        transition: background 0.15s;
    }
    .home-btn:hover { background: rgba(255, 255, 255, 0.4); }
    .chat {
        flex: 1;
        padding: 16px;
        overflow-y: auto;
        background: #f7f9fc;
    }
    .msg {
        max-width: 78%;
        padding: 11px 13px;
        margin: 8px 0;
        border-radius: 16px;
        line-height: 1.45;
        font-size: 14px;
        white-space: pre-wrap;
    }
    .bot {
        background: #ffffff;
        color: #222;
        border-top-left-radius: 4px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .user {
        background: #8ab4ff;
        color: white;
        margin-left: auto;
        border-top-right-radius: 4px;
    }
    /* 가로 스크롤 카드 줄 (KT 챗봇 스타일) */
    .cards-row {
        display: flex;
        overflow-x: auto;
        gap: 10px;
        padding: 4px 2px 8px;
        margin: 8px 0;
        -webkit-overflow-scrolling: touch;
    }
    .cards-row::-webkit-scrollbar { height: 4px; }
    .cards-row::-webkit-scrollbar-thumb { background: #cdd6e3; border-radius: 4px; }

    .card {
        flex: 0 0 200px;
        background: white;
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        display: flex;
        flex-direction: column;
    }
    .card-header {
        background: linear-gradient(135deg, #b9e3ff, #8ab4ff);
        color: #1c3a66;
        padding: 14px 14px 14px;
        min-height: 92px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .card-header.diagnose {
        background: linear-gradient(135deg, #ffd58c, #ff9a5c);
        color: #5a2e00;
    }
    .card-header .card-emoji { font-size: 22px; }
    .card-title { font-weight: bold; font-size: 14px; line-height: 1.3; }
    .card-header-sub { font-size: 11px; opacity: 0.8; margin-top: 4px; }

    .card-body {
        padding: 12px;
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .card-desc {
        font-size: 12px;
        color: #444;
        line-height: 1.45;
        margin: 0 0 12px;
    }
    .card-action-btn {
        display: block;
        background: white;
        border: 1px solid #8ab4ff;
        color: #3366aa;
        padding: 8px 10px;
        border-radius: 9px;
        text-align: center;
        text-decoration: none;
        font-size: 12px;
        font-weight: bold;
        cursor: pointer;
    }
    .card-action-btn.primary {
        background: #ff9a5c;
        border-color: #ff9a5c;
        color: white;
    }

    .risk-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 8px;
        font-size: 10px;
        font-weight: bold;
        margin-bottom: 6px;
    }
    .risk-LOW { background: #e9f7ef; color: #247a3d; }
    .risk-MEDIUM { background: #fff3e0; color: #b76a00; }
    .risk-HIGH { background: #fdecea; color: #b3261e; }
    /* 백엔드 플래그 뱃지 — 보습·진정(긍정) / 알레르기·자극성·여드름(주의) */
    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        margin-bottom: 6px;
    }
    .flag-badge {
        display: inline-block;
        padding: 3px 7px;
        border-radius: 8px;
        font-size: 10px;
        font-weight: bold;
        white-space: nowrap;
    }
    .flag-good { background: #e3f2fd; color: #1565c0; }
    .flag-warn { background: #fff8e1; color: #b76a00; }
    /* 영문 학명 (name_en) — 한글명 아래 작게 표시 */
    .card-name-en {
        font-size: 10px;
        color: #4a6789;
        margin-top: 3px;
        font-style: italic;
        opacity: 0.85;
    }
    /* 기능 태그 (functions) — 미백·보습·진정 등 */
    .function-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 3px;
        margin-bottom: 6px;
    }
    .fn-tag {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 6px;
        font-size: 10px;
        font-weight: 600;
        background: #f0f4f8;
        color: #3d556e;
    }
    .quick {
        display: flex;
        flex-wrap: wrap;
        gap: 7px;
        padding: 10px 12px 4px;
        background: white;
    }
    .quick button {
        border: none;
        background: #eef4ff;
        color: #3366aa;
        padding: 8px 11px;
        border-radius: 20px;
        cursor: pointer;
        font-size: 12px;
    }
    .quick button.home-chip {
        background: #fff;
        color: #7a8aa3;
        border: 1px solid #d7e0ee;
    }
    .quick button.home-chip:hover { background: #f2f6fc; }
    .suggestions {
        display: none;
        flex-wrap: nowrap;
        gap: 6px;
        padding: 10px 12px;
        background: #f7f9fc;
        border-top: 1px solid #e3eaf3;
        overflow-x: auto;
        max-height: 80px;
        -webkit-overflow-scrolling: touch;
    }
    .suggestions::-webkit-scrollbar { height: 3px; }
    .suggestions::-webkit-scrollbar-thumb { background: #cdd6e3; border-radius: 3px; }
    .suggestions button {
        flex: 0 0 auto;
        border: 1px solid #d6e1f0;
        background: white;
        color: #3366aa;
        padding: 6px 11px;
        border-radius: 14px;
        cursor: pointer;
        font-size: 11px;
        white-space: nowrap;
    }
    .suggestions button:hover {
        background: #eef4ff;
    }
    .suggestions button .match {
        color: #d63384;
        font-weight: bold;
    }
    .suggestions .searching {
        flex: 0 0 auto;
        color: #8a96a8;
        font-size: 11px;
        padding: 6px 11px;
        font-style: italic;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .suggestions .searching::before {
        content: "";
        width: 10px;
        height: 10px;
        border: 2px solid #cdd6e3;
        border-top-color: #3366aa;
        border-radius: 50%;
        animation: spin 0.7s linear infinite;
    }
    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    .input-area {
        display: flex;
        gap: 8px;
        padding: 12px;
        background: white;
        border-top: 1px solid #eee;
    }
    input {
        flex: 1;
        border: 1px solid #ddd;
        border-radius: 18px;
        padding: 11px 13px;
        outline: none;
    }
    .send {
        border: none;
        background: #8ab4ff;
        color: white;
        border-radius: 18px;
        padding: 0 16px;
        cursor: pointer;
        font-weight: bold;
    }
    .debug {
        font-size: 11px;
        color: #888;
        margin-top: 5px;
    }
</style>
</head>
<body>
<div class="phone">
    <div class="header">
        <div class="header-title">
            <h2>DermaLens</h2>
            <p>피부·성분 기반 AI 뷰티 챗봇</p>
        </div>
        <button class="home-btn" onclick="resetChat()">🏠 처음으로</button>
    </div>

    <div id="chat" class="chat">
        <div class="msg bot">안녕하세요. DermaLens 챗봇입니다. 피부 타입이나 성분을 입력해보세요.</div>
    </div>

    <div id="quick" class="quick">
        <button onclick="quickSend('제품 추천')">제품 추천</button>
        <button onclick="quickSend('성분 분석')">성분 분석</button>
        <button onclick="quickSend('피부 진단')">피부 진단</button>
        <button onclick="quickSend('메뉴')">메뉴</button>
    </div>

    <div id="suggestions" class="suggestions"></div>

    <div class="input-area">
        <input id="message" placeholder="질문을 입력해주세요" onkeydown="enterSend(event)" autocomplete="off">
        <button class="send" onclick="sendMessage()">전송</button>
    </div>
</div>

<script>
const chat = document.getElementById("chat");
const quick = document.getElementById("quick");

const RISK_LABEL = { LOW: "낮음", MEDIUM: "중간", HIGH: "높음" };

const quickMap = {
    "제품 추천": "제품 추천해줘",
    "성분 분석": "성분 분석해줘",
    "피부 진단": "피부 타입 알려줘",
    "민감성": "민감성 피부 제품 추천해줘",
    "건성": "건성 피부 제품 추천해줘",
    "지성": "지성 피부 제품 추천해줘",
    "복합성": "복합성 피부 제품 추천해줘",
    "여드름성": "여드름성 피부 제품 추천해줘",
    // 마스터데이터 8타입 직접 선택 칩
    "민지형(OS+)": "민지형 피부 제품 추천해줘",
    "수부민지형(OS-)": "수부민지형 피부 제품 추천해줘",
    "건지형(ON+)": "건지형 피부 제품 추천해줘",
    "수부지형(ON-)": "수부지형 피부 제품 추천해줘",
    "민건형(DS+)": "민건형 피부 제품 추천해줘",
    "극건민감형(DS-)": "극건민감형 피부 제품 추천해줘",
    "건강건성형(DN+)": "건강건성형 피부 제품 추천해줘",
    "수부건형(DN-)": "수부건형 피부 제품 추천해줘",
    // 잘 모르겠어요 → 피부 진단 페이지로 안내
    "잘 모르겠어요": "피부 타입 알려줘",
    // 전체 메뉴 칩 라벨 → 페이지 인식 키워드
    "성분 사진 분석(OCR)": "성분 사진",
    "피부 타입 진단": "피부 진단 페이지",
    "알레르기 관리": "알레르기 등록",
    "앱 만족도 평가": "앱 평가",
    "여드름 제품 추천": "여드름 피부 제품 추천해줘",
    "민감성 크림 추천": "민감성 크림 추천",
    "건성 크림 추천": "건성 크림 추천",
    "지성 토너 추천": "지성 토너 추천",
    // 5개 카테고리 칩 (그림과 동일) — 클릭 시 카테고리 단독 추천
    "토너": "토너 추천해줘",
    "에센스": "에센스 추천해줘",
    "세럼": "세럼 추천해줘",
    "앰플": "앰플 추천해줘",
    "로션": "로션 추천해줘",
    "다른 성분 확인": "히알루론산 알려줘",
    "히알루론산": "히알루론산",
    "페녹시에탄올": "페녹시에탄올",
    "알코올": "알코올",
    "병풀추출물": "병풀추출물",
    "글리세린": "글리세린",
    "나이아신아마이드": "나이아신아마이드",
    "레티놀": "레티놀",
    "세라마이드": "세라마이드",
    "이리스산": "이리스산",
    "향료": "향료",
    "파라벤": "파라벤",
    "티트리오일": "티트리오일",
    "PDRN": "PDRN 알려줘",
    "마데카소사이드": "마데카소사이드 알려줘",
    "판테놀": "판테놀 알려줘",
    "아데노신": "아데노신 알려줘",
    "콜라겐": "콜라겐 알려줘",
    "트라넥삼산": "트라넥삼산 알려줘",
    "갈락토미세스": "갈락토미세스 알려줘",
    "AHA": "AHA 알려줘",
    "PHA": "PHA 알려줘",
    "비타민C": "비타민C 알려줘",
    "펩타이드": "펩타이드 알려줘",
    "스쿠알란": "스쿠알란 알려줘",
    "알란토인": "알란토인 알려줘",
    "프로폴리스": "프로폴리스 알려줘",
    "시어버터": "시어버터 알려줘",
    "다른 성분": "다른 성분 보여줘",
    
    "이미지로 분석": "성분 사진",
    "다시 처음으로": "성분 분석해줘",
    "자주 묻는 질문": "자주 묻는 질문 보여줘",
    "글리세린 위험해?": "글리세린 위험해?",
    "페녹시에탄올 위험해?": "페녹시에탄올 위험해?",
    "예민한 편": "민감성 크림 추천",
    "건조한 편": "건성 크림 추천",
    "기름진 편": "지성 토너 추천",
    "트러블이 많음": "여드름 피부 제품 추천해줘"
};

function addMessage(text, type) {
    const div = document.createElement("div");
    div.className = "msg " + type;
    div.innerText = text;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}

// 백엔드 플래그 → 화면 뱃지 매핑
const FLAG_BADGES = [
    { key: "moisturizing_flag",  label: "💧 보습",        cls: "flag-good" },
    { key: "soothing_flag",      label: "🌿 진정",        cls: "flag-good" },
    { key: "allergy_flag",       label: "⚠ 알레르기",     cls: "flag-warn" },
    { key: "irritant_flag",      label: "⚠ 자극성",       cls: "flag-warn" },
    { key: "acne_caution_flag",  label: "⚠ 여드름 주의",  cls: "flag-warn" }
];

function renderFlagBadges(card) {
    return FLAG_BADGES
        .filter(f => card[f.key] === true)
        .map(f => `<span class="flag-badge ${f.cls}">${f.label}</span>`)
        .join("");
}

function makeProductCard(card) {
    const div = document.createElement("div");
    div.className = "card";
    const riskKo = RISK_LABEL[card.riskLevel] || card.riskLevel;
    const flagsHtml = renderFlagBadges(card);

    // name_en — 한글명 아래 영문 학명 (있을 때만)
    const subtitleHtml = card.name_en
        ? `<div class="card-name-en">${card.name_en}</div>`
        : `<div class="card-header-sub">DermaLens 추천</div>`;

    // functions — 미백·보습·진정 같은 기능 태그 (배열 비어있지 않을 때만)
    const fns = Array.isArray(card.functions) ? card.functions.filter(Boolean) : [];
    const functionsHtml = fns.length
        ? `<div class="function-tags">${fns.map(f => `<span class="fn-tag">${f}</span>`).join("")}</div>`
        : "";

    div.innerHTML = `
        <div class="card-header">
            <div class="card-emoji">🧴</div>
            <div>
                <div class="card-title">${card.title}</div>
                ${subtitleHtml}
            </div>
        </div>
        <div class="card-body">
            <div>
                <div class="badge-row">
                    <span class="risk-badge risk-${card.riskLevel}">위험도 ${riskKo}</span>
                    ${flagsHtml}
                </div>
                ${functionsHtml}
                <div class="card-desc">${card.description}</div>
            </div>
            <a class="card-action-btn">${card.buttonText || "자세히 보기"}</a>
        </div>
    `;
    return div;
}

function makeLinkCard(link) {
    const div = document.createElement("div");
    div.className = "card";
    div.style.cursor = "pointer";
    div.onclick = () => { window.location.href = link.url; };
    div.innerHTML = `
        <div class="card-header diagnose">
            <div class="card-emoji">🔍</div>
            <div>
                <div class="card-title">피부 진단</div>
                <div class="card-header-sub">맞춤형 진단을 받아보세요</div>
            </div>
        </div>
        <div class="card-body">
            <div class="card-desc">간단한 질문 몇 가지로 내 피부 타입을 정확하게 알아봐요.</div>
            <a class="card-action-btn primary" href="${link.url}">${link.label || "진단 시작하기"}</a>
        </div>
    `;
    return div;
}

function renderComponents(components) {
    if (!components || !components.length) return;
    const row = document.createElement("div");
    row.className = "cards-row";
    components.forEach(item => {
        if (item.type === "card") row.appendChild(makeProductCard(item));
        else if (item.type === "link") row.appendChild(makeLinkCard(item));
    });
    if (row.children.length) {
        chat.appendChild(row);
        chat.scrollTop = chat.scrollHeight;
    }
}

function renderQuickReplies(list) {
    quick.innerHTML = "";
    (list || []).forEach(text => {
        const btn = document.createElement("button");
        const isHome = (text === "처음으로" || text === "🏠 처음으로");
        if (isHome) {
            btn.innerText = "🏠 처음으로";
            btn.className = "home-chip";
            btn.onclick = resetChat;
        } else {
            btn.innerText = text;
            btn.onclick = () => quickSend(text);
        }
        quick.appendChild(btn);
    });
}

async function sendMessage() {
    const input = document.getElementById("message");
    const message = input.value.trim();
    if (!message) return;

    addMessage(message, "user");
    input.value = "";

    const res = await fetch("/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({message, user_id: 1})
    });

    const data = await res.json();

    addMessage(data.message, "bot");

    renderComponents(data.components);

    const debug = `intent: ${data.intent} / score: ${data.score}`;
    const debugDiv = document.createElement("div");
    debugDiv.className = "debug";
    debugDiv.innerText = debug;
    chat.appendChild(debugDiv);

    // 봇 응답 하단 퀵리플라이 끝에 '처음으로' 칩을 항상 붙여 흐름 중 홈 복귀 제공
    const replies = (data.quickReplies || []).slice();
    if (!replies.includes("처음으로") && !replies.includes("🏠 처음으로")) {
        replies.push("처음으로");
    }
    renderQuickReplies(replies);

    chat.scrollTop = chat.scrollHeight;
}

function quickSend(text) {
    const realMessage = quickMap[text] || text;
    document.getElementById("message").value = realMessage;
    sendMessage();
}

// 처음으로 — 대화 내용을 비우고 첫 인사 화면으로 초기화 (서버 호출 없음)
function resetChat() {
    chat.innerHTML = '<div class="msg bot">안녕하세요. DermaLens 챗봇입니다. 피부 타입이나 성분을 입력해보세요.</div>';
    renderQuickReplies(["제품 추천", "성분 분석", "피부 진단", "메뉴"]);
    document.getElementById("message").value = "";
    hideSuggestions();
    chat.scrollTop = 0;
}

function enterSend(event) {
    // 한글 IME 조합 중(Enter로 글자 확정 중)에는 전송하지 않음 — 마지막 글자가 입력창에 남는 문제 방지
    if (event.isComposing || event.keyCode === 229) return;
    if (event.key === "Enter") {
        document.getElementById("suggestions").style.display = "none";
        sendMessage();
    }
}

// 자동완성 (KB 챗봇 스타일)
const suggestionsEl = document.getElementById("suggestions");
const messageInput = document.getElementById("message");
const quickEl = document.getElementById("quick");
let suggestDebounce = null;

function escapeHtml(s) {
    return s.replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
}

function highlightMatch(text, query) {
    if (!query) return escapeHtml(text);
    const lower = text.toLowerCase();
    const idx = lower.indexOf(query.toLowerCase());
    if (idx === -1) return escapeHtml(text);
    const before = escapeHtml(text.slice(0, idx));
    const match = escapeHtml(text.slice(idx, idx + query.length));
    const after = escapeHtml(text.slice(idx + query.length));
    return `${before}<span class="match">${match}</span>${after}`;
}

function hideSuggestions() {
    suggestionsEl.style.display = "none";
    quickEl.style.display = "flex";  // 빠른답장 다시 표시
}

function showSearching() {
    suggestionsEl.innerHTML = '<span class="searching">DB 검색 중...</span>';
    suggestionsEl.style.display = "flex";
    quickEl.style.display = "none";
}

function renderSuggestions(list, query) {
    suggestionsEl.innerHTML = "";
    if (!list || !list.length) {
        hideSuggestions();
        return;
    }
    list.forEach(text => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.innerHTML = highlightMatch(text, query);
        // mousedown 우선 사용 — input blur보다 먼저 발화해 클릭 누락 방지
        btn.addEventListener("mousedown", (e) => {
            e.preventDefault();  // input focus 유지
            messageInput.value = text;
            hideSuggestions();
            sendMessage();
        });
        suggestionsEl.appendChild(btn);
    });
    suggestionsEl.style.display = "flex";
    quickEl.style.display = "none";  // 자동완성 뜨면 빠른답장 숨김 (겹침 방지)
}

// race condition 방지 — 빨리 타이핑하면 늦은 응답이 새 응답 덮어쓸 수 있음
let suggestReqId = 0;

messageInput.addEventListener("input", () => {
    clearTimeout(suggestDebounce);
    const q = messageInput.value.trim();
    if (!q) {
        hideSuggestions();
        return;
    }
    suggestDebounce = setTimeout(async () => {
        const myReqId = ++suggestReqId;
        showSearching();  // 즉시 "검색 중..." 표시 (멈춘 느낌 제거)
        try {
            const res = await fetch(`/suggest?q=${encodeURIComponent(q)}`);
            const data = await res.json();
            if (myReqId !== suggestReqId) return;  // 더 새 요청 있으면 무시
            renderSuggestions(data.suggestions, q);
        } catch (e) {
            if (myReqId !== suggestReqId) return;
            hideSuggestions();
        }
    }, 120);
});

// 입력칸 밖 클릭하면 자동완성 닫기
document.addEventListener("click", (e) => {
    if (e.target !== messageInput && !suggestionsEl.contains(e.target)) {
        hideSuggestions();
    }
});
</script>
</body>
</html>
""")


if __name__ == "__main__":
    # FIX: 환경변수로만 debug 활성화 (기본은 OFF)
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)

