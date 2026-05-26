from flask import Flask, request, jsonify, render_template_string
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import re

# 한글 완성형 음절(가-힣) / 영문 단어 패턴
KOREAN_SYLLABLE_RE = re.compile(r'[가-힣]')
ENGLISH_WORD_RE = re.compile(r'[a-zA-Z]{2,}')


def is_meaningful_text(text):
    """한글 음절(가-힣)이나 2글자 이상 영문이 있어야 의미 있는 입력으로 간주.
    'ㄱㄴㄷ', 'ㅏㅓㅗ', '...' 같은 자모/기호만 있는 입력은 거름."""
    if KOREAN_SYLLABLE_RE.search(text):
        return True
    if ENGLISH_WORD_RE.search(text):
        return True
    return False

app = Flask(__name__)

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
        "민감성 피부에 써도 돼?",
        "알코올 들어가면 안 좋아?",
        "이 성분 안전해?",
        "성분 위험도 알려줘",
        "이 성분 주의해야 해?",
        "히알루론산 알려줘",
        "나이아신아마이드 뭐야",
        "성분 설명해줘",
        "글리세린 지성피부에 어때?",
        "히알루론산 건성 피부에 좋아?",
        "알코올 민감성 피부에 써도 돼?",
        "레티놀 여드름 피부에 괜찮아?"
    ],
    "PRODUCT_RECOMMEND": [
        "추천해줘",
        "제품 추천해줘",
        "크림 추천",
        "토너 추천",
        "세럼 추천",
        "수분크림 추천해줘",
        "민감성 피부 추천",
        "민감성 피부에 좋은 제품 알려줘",
        "건성 피부에 맞는 화장품 추천해줘",
        "지성 피부 토너 추천해줘",
        "복합성 피부에 맞는 제품 있어?",
        "여드름 피부에 맞는 제품 있어?",
        "피부가 예민한데 뭘 써야 해?",
        "보습에 좋은 제품 추천해줘",
        "진정에 좋은 화장품 알려줘"
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
    "글리세린": {"risk": "LOW", "description": "수분을 끌어당겨 피부 보습에 도움을 주는 성분입니다."}
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
    "글리세린": ["글리세린", "glycerin"]
}

skin_aliases = {
    "민감성": ["민감성", "예민", "자극", "붉어", "발갛", "진정"],
    "건성": ["건성", "건조", "보습", "땡김", "트임"],
    "지성": ["지성", "피지", "번들", "기름", "유분", "모공"],
    "복합성": ["복합성", "티존", "t존", "부분건조", "부분유분"],
    "여드름성": ["여드름성", "여드름", "트러블", "뾰루지", "각질"]
}

# FIX: "토너" 중복 제거
category_aliases = {
    "크림": ["크림", "수분크림", "젤크림", "보습크림"],
    "토너": ["토너", "스킨"],
    "세럼": ["세럼", "앰플", "에센스"],
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
        "병풀추출물": "진정 케어에 도움을 줄 수 있어 지성 피부의 트러블 진정에도 활용하기 좋습니다."
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
        "병풀추출물": "건조로 인해 예민해진 피부 진정에 도움을 줄 수 있습니다."
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
        "병풀추출물": "피부 진정에 도움을 줄 수 있어 민감성 피부에 비교적 잘 맞는 성분입니다."
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
        "병풀추출물": "트러블로 예민해진 피부 진정에 도움을 줄 수 있습니다."
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
        "병풀추출물": "예민하거나 붉어진 부위 진정에 도움을 줄 수 있습니다."
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


def find_skin_type_alias(text):
    """1단계: 사전(별칭) 기반 빠른 매칭."""
    normalized = normalize_text(text)

    for skin_type, aliases in skin_aliases.items():
        for alias in aliases:
            if normalize_text(alias) in normalized:
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
    for product in products_db:
        skin_match = "skin_type" not in keywords or product["skin_type"] == keywords["skin_type"]
        category_match = "category" not in keywords or product["category"] == keywords["category"]
        if skin_match and category_match:
            results.append(product)
    return results


def get_skin_ingredient_comment(skin_type, ingredient):
    if not skin_type:
        return ""

    comment = skin_ingredient_comments.get(skin_type, {}).get(ingredient)
    if comment:
        return " " + comment

    return f" {skin_type} 피부라면 처음 사용할 때 소량 테스트 후 사용하는 것이 좋습니다."


def build_recommend_message(keywords):
    # FIX: skin_type / category 유무에 따라 자연스러운 메시지 생성
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
                "quickReplies": ["민감성", "건성", "지성", "복합성", "여드름성", "잘 모르겠어요"]
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
                "message": "확인하고 싶은 성분명을 입력해주세요. 예를 들어 '글리세린 지성피부에 어때?'처럼 입력하면 됩니다.",
                "components": [],
                "quickReplies": ["글리세린 지성피부에 어때?", "히알루론산 건성 피부에 좋아?", "알코올 민감성 피부에 써도 돼?"]
            }

        data = ingredients_db.get(ingredient)

        if not data:
            return {
                "intent": intent,
                "message": f"{ingredient} 성분은 현재 DB에 없습니다.",
                "components": [],
                "quickReplies": ["다른 성분 확인", "성분 분석하기"]
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
                "quickReplies": ["병풀추출물", "세라마이드", "이리스산", "향료", "파라벤", "티트리오일", "다시 처음으로"]
            }

        # 기본: 어떤 성분 묻기
        return {
            "intent": intent,
            "message": "어떤 성분이 궁금하세요? 아래에서 선택하거나 직접 입력해주세요.",
            "components": [],
            "quickReplies": ["글리세린", "히알루론산", "나이아신아마이드", "레티놀", "알코올", "페녹시에탄올", "다른 성분", "이미지로 분석"]
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


@app.route("/chat", methods=["POST"])
def chat():
    # FIX: 요청 본문 방어
    data = request.get_json(silent=True) or {}
    user_input = (data.get("message") or "").strip()

    if not user_input:
        return jsonify({
            "intent": "UNKNOWN",
            "score": 0.0,
            "keywords": {},
            "message": "메시지를 입력해주세요.",
            "components": [],
            "quickReplies": ["제품 추천", "성분 분석", "피부 진단"]
        })

    # 자모만 있거나 기호만 있는 무의미 입력은 분류기 돌리지 않고 바로 UNKNOWN
    if not is_meaningful_text(user_input):
        response = generate_response("UNKNOWN", {}, user_input=user_input)
        response["score"] = 0.0
        response["keywords"] = {}
        return jsonify(response)

    # 임베딩 1회만 계산해서 의도 분류 + 피부타입 의미 매칭에 재사용
    user_embedding = model.encode([user_input])

    keywords = extract_keywords(user_input, embedding=user_embedding)

    # FIX: 분류기 결과를 우선 신뢰하고, 신뢰도가 낮을 때만 키워드 폴백
    intent, score = classify_intent(user_input, embedding=user_embedding)

    if intent == "UNKNOWN" or score < 0.5:
        if "ingredient" in keywords:
            intent = "INGREDIENT_RISK"
            score = max(score, 0.9)
        elif "skin_type" in keywords or "category" in keywords:
            intent = "PRODUCT_RECOMMEND"
            score = max(score, 0.9)

    # 키워드도 없고 신뢰도도 애매하면(0.7 미만) UNKNOWN으로 강제
    # 짧은 의미없는 한국어("뭐야", "넌", "니" 등)가 약한 매칭으로 통과되는 것 방지
    if not keywords and score < 0.7:
        intent = "UNKNOWN"

    response = generate_response(intent, keywords, user_input=user_input)
    response["score"] = round(float(score), 3)
    response["keywords"] = keywords

    return jsonify(response)


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
    }
    .header h2 { margin: 0; font-size: 22px; }
    .header p { margin: 6px 0 0; font-size: 13px; opacity: 0.9; }
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
        <h2>DermaLens</h2>
        <p>피부·성분 기반 AI 뷰티 챗봇</p>
    </div>

    <div id="chat" class="chat">
        <div class="msg bot">안녕하세요. DermaLens 챗봇입니다. 피부 타입이나 성분을 입력해보세요.</div>
    </div>

    <div id="quick" class="quick">
        <button onclick="quickSend('제품 추천')">제품 추천</button>
        <button onclick="quickSend('성분 분석')">성분 분석</button>
        <button onclick="quickSend('피부 진단')">피부 진단</button>
    </div>

    <div class="input-area">
        <input id="message" placeholder="질문을 입력해주세요" onkeydown="enterSend(event)">
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
    "잘 모르겠어요": "내 피부 타입이 뭔지 잘 모르겠어",
    "여드름 제품 추천": "여드름 피부 제품 추천해줘",
    "민감성 크림 추천": "민감성 크림 추천",
    "건성 크림 추천": "건성 크림 추천",
    "지성 토너 추천": "지성 토너 추천",
    "다른 성분 확인": "히알루론산 건성 피부에 좋아?",
    "글리세린 지성피부에 어때?": "글리세린 지성피부에 어때?",
    "히알루론산 건성 피부에 좋아?": "히알루론산 건성 피부에 좋아?",
    "알코올 민감성 피부에 써도 돼?": "알코올 민감성 피부에 써도 돼?",
    "페녹시에탄올 민감성 피부에 어때?": "페녹시에탄올 민감성 피부에 어때?",
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
    "다른 성분": "다른 성분 보여줘",
    "이미지로 분석": "성분표 이미지로 분석해줘",
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

function makeProductCard(card) {
    const div = document.createElement("div");
    div.className = "card";
    const riskKo = RISK_LABEL[card.riskLevel] || card.riskLevel;
    div.innerHTML = `
        <div class="card-header">
            <div class="card-emoji">🧴</div>
            <div>
                <div class="card-title">${card.title}</div>
                <div class="card-header-sub">DermaLens 추천</div>
            </div>
        </div>
        <div class="card-body">
            <div>
                <span class="risk-badge risk-${card.riskLevel}">위험도 ${riskKo}</span>
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
    list.forEach(text => {
        const btn = document.createElement("button");
        btn.innerText = text;
        btn.onclick = () => quickSend(text);
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
        body: JSON.stringify({message})
    });

    const data = await res.json();

    addMessage(data.message, "bot");

    renderComponents(data.components);

    const debug = `intent: ${data.intent} / score: ${data.score}`;
    const debugDiv = document.createElement("div");
    debugDiv.className = "debug";
    debugDiv.innerText = debug;
    chat.appendChild(debugDiv);

    if (data.quickReplies) {
        renderQuickReplies(data.quickReplies);
    }

    chat.scrollTop = chat.scrollHeight;
}

function quickSend(text) {
    const realMessage = quickMap[text] || text;
    document.getElementById("message").value = realMessage;
    sendMessage();
}

function enterSend(event) {
    if (event.key === "Enter") sendMessage();
}
</script>
</body>
</html>
""")


if __name__ == "__main__":
    # FIX: 환경변수로만 debug 활성화 (기본은 OFF)
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)
