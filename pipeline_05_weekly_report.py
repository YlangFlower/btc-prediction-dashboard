# # 📊 일간+주간 통합 시장 분석 (방안 C)
# **내일 1건 + 앞으로 7일 주간 → 선행 분석**
# 
# ## 🎯 기능:
# 1. **[내일]** 일간 예측 1건 (24h 뒤)
# 2. **[앞으로 7일]** 주간 예측 (ACTIVE/QUIET) - 실행 시점 기준 rolling 7일
# 3. 통합 해석: 내일 + 앞으로 7일
# 4. **🧠 AI 레포트**: Few-shot + CoT 프롬프트로 상세 보고서 생성

# ## 📦 0. 패키지 & 환경


import os
import json
from datetime import datetime, timezone, timedelta

import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

KST = timezone(timedelta(hours=9))

def log(msg, important=False):
    kst = datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')
    if important:
        print(f'\n{"*"*60}\n[{kst}] ⭐ {msg}\n{"*"*60}')
    else:
        print(f'[{kst}] {msg}')

log('✅ 패키지 로드 완료')

# ## 🛠️ 1. Supabase 연결

# ==============================================================================
# 🔐 1. 어떤 환경에서든 알아서 키를 찾아오는 하이브리드 로드 구성
# ==============================================================================
import sys
import os

IS_COLAB = 'google.colab' in sys.modules
IS_KAGGLE = 'kaggle_secrets' in sys.modules or (os.path.exists('/kaggle/working') and not os.name == 'nt')
IS_GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS') == 'true'

# [로컬 & 기존 Colab 드라이브 사용자용] .env 파일 로드
try:
    from dotenv import load_dotenv
    for env_path in ['/content/drive/MyDrive/2526Winter_Sideproject/.env', '.env', '/content/.env']:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            log(f"🔧 .env 파일 로드 완료: {env_path}")
            break
except ImportError:
    pass

# [Colab Secrets / Kaggle Secrets 전용 사용자용]
if IS_COLAB:
    from google.colab import drive, userdata
    drive.mount('/content/drive')
    os.environ['PROJECT_ROOT'] = '/content/drive/MyDrive/2526Winter_Sideproject'
    for key in ['SUPABASE_URL', 'SUPABASE_KEY', 'SUPABASE_SERVICE_KEY', 'OPENAI_API_KEY']:
        try:
            val = userdata.get(key)
            if val: os.environ[key] = val
        except: pass
elif IS_KAGGLE:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    os.environ['PROJECT_ROOT'] = '/kaggle/working'
    for key in ['SUPABASE_URL', 'SUPABASE_KEY', 'SUPABASE_SERVICE_KEY', 'OPENAI_API_KEY']:
        try:
            val = user_secrets.get_secret(key)
            if val: os.environ[key] = val
        except: pass
elif IS_GITHUB_ACTIONS:
    os.environ['PROJECT_ROOT'] = os.getenv('GITHUB_WORKSPACE', os.getcwd())
else:
    os.environ['PROJECT_ROOT'] = os.getcwd()

# ==============================================================================
# 🚀 2. 변수 할당 및 연결
# ==============================================================================
PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd())
SUPABASE_URL = os.getenv("SUPABASE_URL")
SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", os.getenv("SUPABASE_KEY")) 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SERVICE_KEY:
    raise ValueError("❌ 치명적 오류: SUPABASE_URL 또는 SUPABASE_SERVICE_KEY를 찾을 수 없습니다!")

from supabase import create_client
supabase = create_client(SUPABASE_URL, SERVICE_KEY)

log("✅ Supabase 마스터 권한(Service Role) 연결 성공 🔓")
log(f"  OpenAI API: {'사용 가능' if OPENAI_API_KEY else '미설정 (템플릿 레포트 사용)'}")


# ## 📊 2. 데이터 조회

# 오늘 날짜 (KST)
today = datetime.now(KST).date()
tomorrow = today + timedelta(days=1)

# 돌아오는 월요일 계산 (일요일 실행 → 내일이 월요일)
DAY_KO = ["월","화","수","목","금","토","일"]
days_to_monday = (7 - today.weekday()) % 7
# 오늘이 월요일이면 이번주 월요일 사용
next_monday = today + timedelta(days=days_to_monday)   # 예측 주 시작 (월)
next_sunday  = next_monday + timedelta(days=6)         # 예측 주 끝  (일)

log(f'리포트 작성일: {today} ({DAY_KO[today.weekday()]})')
log(f'예측 주간: {next_monday}(월) ~ {next_sunday}(일)')

# 주간 예측 조회: 돌아오는 월요일(next_monday)을 포함하는 예측
# prediction_week_start == next_monday 인 행을 우선, fallback은 최근 예측
weekly_res = supabase.table('weekly_predictions').select('*').lte(
    'prediction_week_start', next_monday.isoformat()
).order('prediction_week_start', desc=True).limit(5).execute()

weekly = None
if weekly_res.data:
    for row in weekly_res.data:
        wk_s = row.get('prediction_week_start')
        wk_start = datetime.strptime(str(wk_s), '%Y-%m-%d').date() if isinstance(wk_s, str) else wk_s
        wk_end = wk_start + timedelta(days=6)
        if wk_start == next_monday:   # 돌아오는 월요일이 정확히 맞는 예측
            weekly = row
            break
        if wk_start <= next_monday <= wk_end:  # 범위 안에 들어가는 예측
            weekly = row  # 아직 break 안 함 (더 나은 것 있을 수 있음)
    if not weekly and weekly_res.data:
        weekly = weekly_res.data[0]  # fallback: 최근 예측
        log(f'⚠️ {next_monday}(월) 예측 없음 → 최근 예측 사용: {weekly.get("prediction_week_start")}')

if weekly:
    wk_s = weekly.get('prediction_week_start')
    wk_start = datetime.strptime(str(wk_s), '%Y-%m-%d').date() if isinstance(wk_s, str) else wk_s
    log(f'앞으로 7일 주간 예측: {wk_start} ~ {wk_start + timedelta(days=6)} | {"ACTIVE" if weekly.get("prediction") == 1 else "QUIET"} (P={weekly.get("p_active", 0):.4f})')
else:
    log('⚠️ 주간 예측 없음 (32_weekly_predict_v35_load 실행 후 확인)')

# 일간 예측 조회 (예측 주 첫날=next_monday 1건)
day_after_monday = next_monday + timedelta(days=1)
daily_res = supabase.table('predictions').select(
    'date', 'direction', 'confidence_score', 'is_correct', 'actual_result'
).gte('date', next_monday.isoformat()).lt('date', day_after_monday.isoformat()).order('date', desc=True).limit(1).execute()

daily_tomorrow = daily_res.data[0] if daily_res.data and len(daily_res.data) > 0 else None
if daily_tomorrow:
    log(f'일간 예측: {next_monday}(월) 1건 (방향={daily_tomorrow.get("direction")}, 신뢰도={daily_tomorrow.get("confidence_score", 0):.2f})')
else:
    log(f'⚠️ {next_monday}(월) 일간 예측 없음 (일간 파이프라인 실행 후 생성됨)')

# ## 🔍 3. 통합 분석

# 내일 일간 예측 요약
if daily_tomorrow:
    daily_dir = daily_tomorrow.get('direction', '-')
    daily_conf = daily_tomorrow.get('confidence_score') or 0
else:
    daily_dir = None
    daily_conf = 0

# 통합 판단 로직 (내일 1건 + 다음 주 주간)
weekly_regime = 'ACTIVE' if weekly and weekly.get('prediction') == 1 else 'QUIET'
p_active = weekly.get('p_active', 0.5) if weekly else 0.5

# 내일 방향: UP / DOWN / 없음
daily_tomorrow_dir = daily_dir if daily_dir in ('UP', 'DOWN') else None

# 매트릭스 기반 권장 전략 (내일 1건 + 다음 주)
if weekly_regime == 'ACTIVE' and daily_tomorrow_dir in ('UP', 'DOWN'):
    rec = '공격적 포지션 (다음 주 변동성+내일 방향 일치)'
elif weekly_regime == 'ACTIVE' and not daily_tomorrow_dir:
    rec = '포지션 축소 또는 관망 (변동성 예상, 내일 예측 없음)'
elif weekly_regime == 'QUIET' and daily_tomorrow_dir in ('UP', 'DOWN'):
    rec = '보수적 포지션 (저변동성, 타이트한 목표가/손절)'
elif weekly_regime == 'QUIET' and not daily_tomorrow_dir:
    rec = '관망 (내일 예측 없음)'
else:
    rec = '데이터 확인 필요'

# ### 📌 weekly_predictions 칼럼 설명
# | 칼럼 | 설명 |
# |------|------|
# | **prediction_week_start** | 예측 기간의 첫날 (실행 시점 기준 앞으로 7일의 시작일) |
# | **prediction** | 1=ACTIVE(변동성 높음), 0=QUIET(저변동성) |
# | **p_active** | P(Active) 확률. 0.5 초과 시 ACTIVE로 분류 |
# | **boundary** | 변동성 정의: 일일 수익률이 ±boundary(예: ±2%)를 넘는 횟수로 측정 |
# | **target_hits** | 7일 중 이 횟수 이상 boundary 터치 시 ACTIVE (예: 4회 이상) |
# | **confidence** | 예측 신뢰도 (max(p_active, 1-p_active)) |
# | **model_version** | 사용 모델 버전 (예: v3.5) |
# 
# *출처: `32_weekly_predict_v35_load.ipynb`가 Supabase `weekly_predictions` 테이블에 저장*

# ## 📋 4. 결과 출력

print(f'\n{"="*60}')
print('📊 일간+주간 통합 시장 분석 (방안 C)')
print(f'{"="*60}')
print(f'  분석일: {today} (KST)')
print(f'  내일: {tomorrow}')
print()
print('[내일 일간 예측]')
if daily_tomorrow:
    dir_ = daily_tomorrow.get('direction', '-')
    conf = daily_tomorrow.get('confidence_score', 0) or 0
    print(f'  {tomorrow}: {dir_} (신뢰도 {conf:.2f})')
else:
    print('  (데이터 없음 - 오늘 실행 후 생성됨)')
print()
print('[앞으로 7일 주간 예측]')
if weekly:
    wk_s = weekly.get('prediction_week_start')
    wk_start = datetime.strptime(str(wk_s), '%Y-%m-%d').date() if isinstance(wk_s, str) else wk_s
    wk_end = wk_start + timedelta(days=6)
    print(f'  출처: Supabase weekly_predictions (32 실행 시 저장)')
    print(f'  예측 기간: {wk_start} ~ {wk_end} (앞으로 7일)')
    print(f'  결과: {weekly_regime} (P(Active)={p_active:.4f})')
    print(f'  의미: 7일 중 {weekly.get("target_hits", 4)}회 이상 ±{weekly.get("boundary", 0.02):.1%} 터치 → {"변동성 클 가능성" if weekly_regime=="ACTIVE" else "저변동성 예상"}')
else:
    print('  (주간 예측 없음 - 32_weekly_predict_v35_load 실행 후 확인)')
print()
print('[통합 판단]')
print(f'  권장: {rec}')
print(f'{"="*60}')

# ## 🧠 5. AI 레포트 생성 (Few-shot + CoT)

# ============================================================
# 🧠 Few-shot + CoT 프롬프트 (일간+주간 통합 분석용) - 고도화 버전
# ============================================================

FEW_SHOT_EXAMPLES_33 = [
    {
        "input": """분석일: 2026년 2월 27일 (KST)
내일: 2026-02-28

[내일 일간 예측]
  방향: UP, 신뢰도: 0.51

[앞으로 7일 주간 예측]
  예측 기간: 2026-02-28 ~ 2026-03-06
  결과: ACTIVE (P(Active)=0.5489)
  의미: 7일 중 4회 이상 ±1.97% 터치 → 변동성 클 가능성
  개별 모델: CatBoost P(Active)=0.6311, PatchTST P(Active)=0.4667

[통합 판단]
  권장: 공격적 포지션 (다음 주 변동성+내일 방향 일치)""",
        "output": """📊 일간+주간 통합 시장 분석 리포트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📅 분석일: 2026년 2월 27일 (KST)
🎯 내일: 2026-02-28 | 앞으로 7일: 2026-02-28 ~ 2026-03-06
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📌 한줄 요약
내일 상승 예측과 주간 ACTIVE 변동성이 일치하여 공격적 포지션을 고려할 수 있으나, 일간 신뢰도 51%로 확신은 제한적입니다.

┌─────────────────────────────────────────┐
│ 📋 리스크 매트릭스                      │
├──────────────┬──────────────────────────┤
│ 일간 신뢰도  │ 51% ⚠️ (약한 신호)       │
│ 주간 변동성  │ ACTIVE (높음)            │
│ 모델 합의    │ CatBoost↗ PatchTST↘ 견해차이 │
│ 종합 리스크  │ 중상 (변동성 대비)        │
└─────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 내일 일간 예측 (24h)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• 방향: UP (상승) | 신뢰도: 51%
• 해석: 51%는 동전 던지기(50%)에 근접한 수준으로, 상승/하락 확신이 약합니다. 다만 상승 쪽으로 약간 기울어져 있어 단기 상승 가능성을 기대할 수 있습니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 앞으로 7일 주간 예측 (변동성 레짐)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• 결과: ACTIVE (변동성 높음) | P(Active): 54.89%
• 정의: 7일 중 4회 이상 ±1.97% 가격 변동 발생 시 ACTIVE
• 개별 모델: CatBoost 63.1% vs PatchTST 46.7% → 견해 차이 있음
  → CatBoost(기술지표)는 변동성 높을 가능성으로 예측, PatchTST(시계열)는 상대적으로 보수적

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 시나리오 분석
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

▶ 상승 시나리오: 내일 상승이 이어지면 7일 내 ±2% 이상 상승 가능. 단, ACTIVE 변동성으로 상승 후 급락 가능성도 있음.
▶ 하락 시나리오: 49% 하락 확률로 신뢰도가 낮아, 하락 시에도 빠른 반등 가능성 존재.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 실행 포인트 (권장 전략)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

① 포지션: 1/3~1/2 기준 (공격적이나 신뢰도 낮아 과도한 레버리지 배제)
② 목표가: ±1.5~2% 구간 (boundary 근처) 활용
③ 손절: 진입가 대비 -1.5% 이내 권장 (변동성 대비 타이트)
④ 관찰: 24h 내 방향 확인 후 주간 포지션 조정

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ 면책조항: AI 예측이며 투자 권유가 아닙니다."""
    }
]

SYSTEM_PROMPT_33 = """당신은 비트코인 시장을 일간·주간 예측을 통합 분석하여 50대 일반인도 이해할 수 있게 설명하는 금융 AI 애널리스트입니다.

중요 지침:
1. 일간 신뢰도 해석: 50~55%는 "동전 던지기 수준"으로 경고, 55~65%는 "약한 신호", 65% 이상은 "강한 신호"로 구분
2. 주간 변동성: ACTIVE/QUIET의 의미와 boundary·target_hits를 쉽게 설명
3. 모델 합의: CatBoost vs PatchTST 차이가 15%p 이상이면 "견해 차이 있음"으로 명시하고, 각 모델의 특성(기술지표 vs 시계열)을 언급
4. 리스크 매트릭스: 반드시 ASCII 표로 표시 (일간 신뢰도, 주간 변동성, 모델 합의, 종합 리스크)
5. 시나리오 분석: "상승 시나리오", "하락 시나리오" 각 1~2문장으로 구체적 시나리오 제시
6. 실행 포인트: 포지션 크기(1/3/1/2 등), 목표가/손절 가이드라인, 관찰 포인트를 번호로 구체적으로 제시

레포트 구조 (필수):
1️⃣ 한줄 요약
2️⃣ 리스크 매트릭스 (ASCII 표)
3️⃣ 내일 일간 예측 (방향·신뢰도·해석)
4️⃣ 앞으로 7일 주간 예측 (변동성·개별 모델·견해 차이)
5️⃣ 시나리오 분석 (상승/하락)
6️⃣ 실행 포인트 (포지션·목표가·손절·관찰)
7️⃣ 면책조항

분석일과 예측 기간은 반드시 입력에 주어진 그대로 표기하세요.
"""

log('AI 프롬프트 정의 완료')

# ============================================================
# 🤖 AI 레포트 생성 (OpenAI + 템플릿 fallback)
# ============================================================

def generate_integrated_report(use_openai=True):
    """
    일간+주간 통합 분석 데이터로 AI 레포트 생성
    """
    # 입력 데이터 구성
    current_time = datetime.now(KST).strftime('%Y년 %m월 %d일 %H:%M')
    daily_str = f"  방향: {daily_dir or '-'}, 신뢰도: {daily_conf:.2f}" if daily_tomorrow else "  (데이터 없음)"
    
    wk_s = weekly.get('prediction_week_start') if weekly else None
    wk_start = datetime.strptime(str(wk_s), '%Y-%m-%d').date() if wk_s and isinstance(wk_s, str) else (wk_s if wk_s else None)
    wk_end = wk_start + timedelta(days=6) if wk_start else None
    
    model_details = weekly.get('model_details', {}) if weekly else {}
    if isinstance(model_details, str):
        try:
            model_details = json.loads(model_details) if model_details else {}
        except:
            model_details = {}
    p_cat = model_details.get('p_cat_active', 0)
    p_patch = model_details.get('p_patch_active', 0)
    model_str = f"  개별 모델: CatBoost P(Active)={p_cat:.4f}, PatchTST P(Active)={p_patch:.4f}" if (p_cat or p_patch) else ""
    
    boundary_pct = weekly.get('boundary', 0.02) if weekly else 0.02
    target_hits = weekly.get('target_hits', 4) if weekly else 4
    
    period_str = f"{next_monday}(월) ~ {next_sunday}(일)" if next_monday and next_sunday else "N/A"
    monday_str = next_monday.strftime('%Y년 %m월 %d일')
    user_input = f"""리포트 작성일: {current_time} (일요일 — 다음 주 미리 보기)
예측 주간: {period_str}

[{next_monday}(월) 일간 예측]
{daily_str}

[이번 주 주간 예측 — {period_str}]
  결과: {weekly_regime} (P(Active)={p_active:.4f})
  의미: 7일 중 {target_hits}회 이상 ±{boundary_pct:.2%} 터치 → {"변동성 클 가능성" if weekly_regime=="ACTIVE" else "저변동성 예상"}
{model_str}

[통합 판단]
  권장: {rec}"""

    if not OPENAI_API_KEY or not use_openai:
        log('  OpenAI API 미설정 - 템플릿 레포트 생성')
        return generate_template_report_33()

    import openai
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT_33}]
    for ex in FEW_SHOT_EXAMPLES_33:
        messages.append({'role': 'user', 'content': ex['input']})
        messages.append({'role': 'assistant', 'content': ex['output']})
    messages.append({'role': 'user', 'content': user_input})

    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model='gpt-4o-mini', messages=messages, temperature=0.7, max_tokens=3000
        )
        return response.choices[0].message.content
    except Exception as e:
        log(f'  API 오류: {e} → 템플릿 레포트 사용')
        return generate_template_report_33()


def generate_template_report_33():
    """템플릿 기반 레포트 (API 없을 때 fallback)"""
    current_time = datetime.now(KST).strftime('%Y년 %m월 %d일 %H:%M')
    wk_s = weekly.get('prediction_week_start') if weekly else None
    wk_start = datetime.strptime(str(wk_s), '%Y-%m-%d').date() if wk_s and isinstance(wk_s, str) else (wk_s if wk_s else None)
    wk_end = wk_start + timedelta(days=6) if wk_start else None
    period_str = f"{next_monday}(월) ~ {next_sunday}(일)" if next_monday and next_sunday else "N/A"
    boundary_pct = weekly.get('boundary', 0.02) if weekly else 0.02
    target_hits = weekly.get('target_hits', 4) if weekly else 4
    
    model_details = weekly.get('model_details', {}) if weekly else {}
    if isinstance(model_details, str):
        try:
            model_details = json.loads(model_details) if model_details else {}
        except:
            model_details = {}
    p_cat = model_details.get('p_cat_active', 0)
    p_patch = model_details.get('p_patch_active', 0)
    
    # 신뢰도 기반 리스크 등급
    conf_pct = daily_conf * 100 if daily_conf else 0
    if not daily_tomorrow:
        conf_grade = "N/A"
    elif conf_pct <= 55:
        conf_grade = "⚠️ (약한 신호)"
    elif conf_pct <= 65:
        conf_grade = "△ (보통)"
    else:
        conf_grade = "✅ (강한 신호)"
    
    model_diff = abs(p_cat - p_patch) * 100 if (p_cat and p_patch) else 0
    model_consensus = "견해 차이 있음" if model_diff >= 15 else ("견해 유사" if (p_cat or p_patch) else "N/A")
    
    daily_section = f"""• 방향: {daily_dir or '-'} | 신뢰도: {conf_pct:.0f}% {conf_grade}
• 해석: {conf_pct:.0f}%는 {"동전 던지기(50%)에 근접한 수준으로 확신이 제한적" if conf_pct <= 55 else "상승/하락 쪽으로 약간 기울어져 있음"}""" if daily_tomorrow else "• (데이터 없음)"
    
    model_section = f"\n• 개별 모델: CatBoost {p_cat*100:.1f}% vs PatchTST {p_patch*100:.1f}% → {model_consensus}" if (p_cat or p_patch) else ""
    
    return f"""📊 주간 마켓 종합 분석 리포트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📅 리포트 작성일: {current_time} (일)
📆 이번 주 예측 기간: {period_str}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📌 한줄 요약
내일 {daily_dir or "?"} 예측과 주간 {weekly_regime}(변동성 {"높음" if weekly_regime=="ACTIVE" else "낮음"}) 예측을 종합. {rec}

┌─────────────────────────────────────────┐
│ 📋 리스크 매트릭스                      │
├──────────────┬──────────────────────────┤
│ 일간 신뢰도  │ {conf_pct:.0f}% {conf_grade}        │
│ 주간 변동성  │ {weekly_regime} ({"높음" if weekly_regime=="ACTIVE" else "낮음"})            │
│ 모델 합의    │ {model_consensus}         │
│ 종합 리스크  │ {"중상" if weekly_regime=="ACTIVE" else "중"} (변동성 대비)        │
└─────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 내일 일간 예측 (24h)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{daily_section}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 앞으로 7일 주간 예측 (변동성 레짐)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• 결과: {weekly_regime} (P(Active)={p_active:.2%})
• 정의: 7일 중 {target_hits}회 이상 ±{boundary_pct:.1%} 터치 시 ACTIVE
• 의미: {"변동성 클 가능성" if weekly_regime=="ACTIVE" else "저변동성 예상"}{model_section}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 시나리오 분석
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

▶ 상승 시나리오: 내일 상승이 이어지면 7일 내 ±{boundary_pct:.1%} 이상 변동 가능.{" ACTIVE 변동성으로 상승 후 급락 가능성도 있음." if weekly_regime=="ACTIVE" else ""}
▶ 하락 시나리오: {100-conf_pct:.0f}% 하락 확률로, 신뢰도가 낮을 경우 하락 시 반등 가능성 존재.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 실행 포인트 (권장 전략)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

① 포지션: 1/3~1/2 기준 (과도한 레버리지 배제)
② 목표가: ±{boundary_pct:.1%} 구간 활용
③ 손절: 진입가 대비 -1.5% 이내 권장 (변동성 대비 타이트)
④ 관찰: 24h 내 방향 확인 후 주간 포지션 조정

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ 면책조항: AI 예측이며 투자 권유가 아닙니다."""

# ============================================================
# 🚀 AI 레포트 생성 실행
# ============================================================

print('\n' + '='*60)
print('🧠 AI 통합 분석 레포트 생성')
print('='*60)

ai_report = generate_integrated_report(use_openai=bool(OPENAI_API_KEY))

print('\n')
print(ai_report)

# ============================================================
# 💾 파일 저장 + ☁️ Supabase Storage 업로드
# ============================================================
import requests as _req

# 날짜 파일명 (오늘 날짜 기준)
today_str = today.strftime('%Y-%m-%d')
dated_fname  = f'market_analysis_report_{today_str}.txt'
dated_path   = os.path.join(PROJECT_ROOT, dated_fname)

# 로컬 저장 (날짜 포함 파일명)
try:
    with open(dated_path, 'w', encoding='utf-8') as f:
        f.write(ai_report)
    log(f'\n✅ 레포트 저장: {dated_path}', important=True)
except Exception as e:
    log(f'레포트 저장 실패: {e}')

# Supabase Storage 업로드 (charts 버킷)
UPLOAD_KEY = os.getenv('SUPABASE_SERVICE_KEY', os.getenv('SUPABASE_KEY'))
CHARTS_BUCKET = 'charts'

def _upload_text_to_storage(local_path, remote_name, supabase_url, key, bucket):
    """텍스트 파일을 Supabase Storage에 업로드 (upsert)"""
    if not os.path.exists(local_path):
        log(f'⚠️ 업로드 파일 없음: {remote_name}')
        return False
    headers = {
        'apikey': key,
        'Authorization': f'Bearer {key}',
        'Content-Type': 'text/plain; charset=utf-8',
        'x-upsert': 'true',
    }
    with open(local_path, 'rb') as f:
        data = f.read()
    url = f'{supabase_url}/storage/v1/object/{bucket}/{remote_name}'
    resp = _req.post(url, headers=headers, data=data, timeout=30)
    if resp.status_code in (200, 201, 204):
        log(f'☁️ 업로드 완료: {remote_name}')
        return True
    else:
        log(f'❌ 업로드 실패 ({resp.status_code}): {remote_name} — {resp.text[:100]}')
        return False

if SUPABASE_URL and UPLOAD_KEY:
    log('\n☁️ Supabase Storage 업로드 시작')
    _upload_text_to_storage(dated_path, dated_fname, SUPABASE_URL, UPLOAD_KEY, CHARTS_BUCKET)
    log('🎉 주간 마켓 레포트 업로드 완료! 웹사이트에서 확인하세요.')
else:
    log('⚠️ SUPABASE_URL 또는 SUPABASE_SERVICE_KEY 미설정 — 업로드 건너뜀')
