import streamlit as st
import pandas as pd
import os
import json
import glob
import textwrap
import re
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, timezone
from supabase import create_client
from dotenv import load_dotenv

def clean_html(html_str):
    # Strip all leading whitespaces from every line to prevent Streamlit from wrapping in code blocks
    return re.sub(r'^\s+', '', html_str, flags=re.MULTILINE)

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="BTC AI 퀀트 대시보드",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS: 커스텀 스타일 ---
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .gradient-text {
        background: linear-gradient(135deg, #f97316 0%, #f59e0b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        padding-bottom: 10px;
    }
    div[data-testid="stColumn"] > div {
        height: 100%;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        height: 100%;
        border-radius: 16px !important;
        background: rgba(30, 41, 59, 0.7) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        padding: 0.5rem !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
        border-color: rgba(96, 165, 250, 0.5) !important;
    }
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        padding: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
        border-color: rgba(96, 165, 250, 0.5);
    }
    .badge {
        display: inline-block; padding: 4px 12px; border-radius: 16px; font-size: 13px; font-weight: 700; margin-bottom: 8px; margin-right: 8px;
        box-shadow: inset 0 1px 1px rgba(255,255,255,0.15), 0 2px 4px rgba(0,0,0,0.2);
    }
    .badge.bull { background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(34,197,94,0.1)); color: #4ade80; border: 1px solid rgba(34,197,94,0.4); }
    .badge.bear { background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(239,68,68,0.1)); color: #f87171; border: 1px solid rgba(239,68,68,0.4); }
    .badge.neutral { background: linear-gradient(135deg, rgba(148,163,184,0.2), rgba(148,163,184,0.1)); color: #94a3b8; border: 1px solid rgba(148,163,184,0.4); }
    .highlight-val { font-size: 2rem; font-weight: 800; margin: 0; padding: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.5); }
    .pred-up { color: #4ade80; }
    .pred-down { color: #f87171; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 800 !important; text-shadow: 0 2px 4px rgba(0,0,0,0.3); }
</style>
""", unsafe_allow_html=True)

# --- DB 설정 (Streamlit Cloud Secrets 우선, 로컬 .env 폴백) ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    # 로컬 개발 환경에서는 .env 파일 탐색
    for env_path in ["c:\\25WinterProject\\.env", ".env"]:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            break
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Supabase Storage 버킷명
CHARTS_BUCKET = "charts"

@st.cache_resource
def init_supabase():
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None
    except: return None

supabase = init_supabase()

# --- 데이터 로딩 (캐싱) ---
@st.cache_data(ttl=60)
def fetch_all_data():
    out = {
        "features": {}, "market": [], "sentiment_7d": [],
        "prediction": {}, "acc_30d": {"correct": 0, "total": 0},
        "weekly_prediction": {}, "features_30d": []
    }
    if not supabase: return out

    try:
        # 1. Market Data
        res_m = supabase.table('market_realtime').select('*').order('timestamp', desc=True).limit(2).execute()
        out["market"] = res_m.data if res_m.data else []

        # 2. Features Data (latest + 30 days for charts)
        res_f = supabase.table('features_master').select('*').order('date', desc=True).limit(1).execute()
        out["features"] = res_f.data[0] if res_f.data else {}

        thirty_days_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        res_f30 = supabase.table('features_master').select(
            'date, RSI_14, MACD, MACD_signal, BB_position, fng_value, close'
        ).gte('date', thirty_days_ago).order('date', desc=False).execute()
        out["features_30d"] = res_f30.data if res_f30.data else []

        # 3. Sentiment Data
        fourteen_days_ago = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
        res_s = supabase.table('raw_sentiment').select('*').gte('date', fourteen_days_ago).order('date', desc=True).execute()
        out["sentiment_7d"] = res_s.data if res_s.data else []

        # 4. Daily Prediction (Latest)
        res_p = supabase.table('predictions').select('*').order('date', desc=True).limit(1).execute()
        if res_p.data:
            out["prediction"] = res_p.data[0]

        # 5. 30d Accuracy & History
        res_acc = supabase.table('predictions').select('date, direction, is_correct, confidence_score').gte('date', thirty_days_ago).not_.is_('is_correct', 'null').order('date', desc=True).execute()
        if res_acc.data:
            out["acc_30d"]["total"] = len(res_acc.data)
            out["acc_30d"]["correct"] = sum(1 for r in res_acc.data if r.get('is_correct'))
            out["acc_30d"]["history"] = res_acc.data

        # 6. Weekly Prediction (Latest)
        res_w = supabase.table('weekly_predictions').select('*').order('prediction_week_start', desc=True).limit(1).execute()
        if res_w.data:
            out["weekly_prediction"] = res_w.data[0]

    except Exception as e:
        print("Data fetch error:", e)
    return out

data = fetch_all_data()

# --- 마켓/예측 리포트 로더 (캐시 5분) ---
# @st.cache_data 제거: 전역변수(SUPABASE_URL/KEY)가 캐시 키에 반영 안 되는 버그 방지
def _load_text_from_storage(supabase_url, supabase_key, prefix: str):
    """Supabase Storage REST API 직접 호출 - 인자로 URL/KEY 명시 전달"""
    if not supabase_url or not supabase_key:
        return None, None
    try:
        import requests as _req
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json"
        }
        list_url = f"{supabase_url}/storage/v1/object/list/{CHARTS_BUCKET}"
        resp = _req.post(list_url, headers=headers,
                         json={"prefix": prefix, "sortBy": {"column": "name", "order": "desc"}},
                         timeout=10)
        if resp.ok:
            files = resp.json()
            if files and isinstance(files, list):
                latest_name = files[0].get('name', '')
                if latest_name:
                    file_url = f"{supabase_url}/storage/v1/object/public/{CHARTS_BUCKET}/{latest_name}"
                    file_resp = _req.get(file_url, timeout=15)
                    if file_resp.ok:
                        return file_resp.text, latest_name
    except Exception:
        pass
    return None, None

# 캐시 제거: Streamlit Cloud 서버측 캐시 지속 문제 방지
def load_market_report():
    # 1. Supabase table
    try:
        if supabase:
            res = supabase.table('market_reports').select('content,filename').order('created_at', desc=True).limit(1).execute()
            if res.data:
                return res.data[0]['content'], res.data[0].get('filename', 'market_report.txt')
    except Exception:
        pass
    # 2. Supabase Storage (URL/KEY 명시 전달)
    content, fname = _load_text_from_storage(SUPABASE_URL, SUPABASE_KEY, 'market_analysis_report_')
    if content:
        return content, fname
    # 3. 로컬 파일 폴백
    for d in [r"c:\25WinterProject", r"c:\25WinterProject\models\production\v7E_production"]:
        if os.path.exists(d):
            files = glob.glob(os.path.join(d, "market_analysis_report_*.txt"))
            if files:
                latest = sorted(files)[-1]
                with open(latest, "r", encoding="utf-8") as f:
                    return f.read(), os.path.basename(latest)
    return None, None

def load_daily_report():
    # 1. Supabase table
    try:
        if supabase:
            res = supabase.table('reports').select('content,filename').order('created_at', desc=True).limit(1).execute()
            if res.data:
                return res.data[0]['content'], res.data[0].get('filename', 'prediction_report.txt')
    except Exception:
        pass
    # 2. Supabase Storage (URL/KEY 명시 전달)
    content, fname = _load_text_from_storage(SUPABASE_URL, SUPABASE_KEY, 'prediction_report_20')
    if content:
        return content, fname
    # 3. 로컬 파일 폴백
    for d in [r"c:\25WinterProject", r"c:\25WinterProject\models\production\v7E_production"]:
        if os.path.exists(d):
            files = glob.glob(os.path.join(d, "prediction_report_*.txt"))
            if files:
                latest = sorted(files)[-1]
                with open(latest, "r", encoding="utf-8") as f:
                    return f.read(), os.path.basename(latest)
    return None, None

# --- 예측 이미지 탐색 함수 (Supabase Storage 우선, 로컬 폴백) ---
@st.cache_data(ttl=300)
def get_chart_url(chart_name: str) -> str | None:
    """
    Supabase Storage에서 차트 URL 반환.
    - 날짜 suffix 파일(chart_price_v7e_2026-03-01.png) → prefix 기반으로 최신 탐색
    - 고정 이름 파일(backtest_v7e.png) → 직접 URL 반환
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        import requests as _rq
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        }
        # 버킷 전체 목록 조회
        list_url = f"{SUPABASE_URL}/storage/v1/object/list/{CHARTS_BUCKET}"
        r = _rq.post(list_url, headers=headers,
                     json={"prefix": "", "sortBy": {"column": "name", "order": "desc"}},
                     timeout=8)
        if not r.ok:
            return None
        all_files = r.json()
        if not isinstance(all_files, list):
            return None

        # 파일명에서 확장자 제거한 stem(예: chart_price_v7e)으로 prefix 매칭
        stem = os.path.splitext(chart_name)[0]  # e.g. "chart_price_v7e"
        matching = [f['name'] for f in all_files
                    if isinstance(f, dict) and f.get('name', '').startswith(stem)]
        if matching:
            latest = sorted(matching)[-1]  # 날짜 내림차순 → 마지막 = 최신
            return f"{SUPABASE_URL}/storage/v1/object/public/{CHARTS_BUCKET}/{latest}"
    except Exception:
        pass
    return None

def find_pred_image(names):
    """Supabase Storage URL 우선 반환, 없으면 로컬 파일 경로 탐색."""
    for name in names:
        url = get_chart_url(name)
        if url:
            return url
    search_dirs = [
        "c:\\25WinterProject",
        "c:\\25WinterProject\\models\\production\\v7E_production",
        "c:\\25WinterProject\\models\\production\\v7E_production_highAccuracy_dynH"
    ]
    for d in search_dirs:
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return p
    return None


# 포맷팅 유틸
def format_krw(val):
    if not val: return "N/A"
    return f"{val/100000000:.2f}억원" if val > 100000000 else f"{val:,.0f}원"

# --- 상단 헤더 ---
st.markdown("<div style='text-align: center; margin-bottom: 1rem;'>"
            "<div style='font-size: 3.5rem; line-height: 1.2;'>₿</div>"
            "<div class='gradient-text'>BTC AI 종합 대시보드</div>"
            "<p style='color: #8b949e; font-size: 1.1rem; margin-top: -10px;'>End-to-End 예측 & 실시간 마켓 애널리틱스</p>"
            "</div>", unsafe_allow_html=True)

# --- 탭 구성 ---
tab_main, tab_news, tab_charts, tab_report = st.tabs(["🎯 최신 AI 예측 & 시황", "📰 AI 뉴스 감성 분석", "📊 기술적 차트 및 구조", "📝 일간/주간 마켓 리포트"])

# ==============================================================================
# 탭 1: 대시보드 메인
# ==============================================================================
with tab_main:

    # ── ① 예측 차트 이미지 상단 배치 ──────────────────────────────────────────
    # chart_models_v7e.png (AI 모델별 예측 막대 차트) 를 메인 탭 최상단에 배치.
    # 나중에 별도 Instagram용 카드 이미지를 동일한 위치/크기로 교체하려면
    # find_pred_image()에서 찾는 파일 목록 첫 번째 항목만 바꾸면 됩니다.
    _pred_img = find_pred_image(["chart_models_v7e.png", "chart_price_v7e.png"])
    if _pred_img:
        with st.container(border=True):
            st.image(_pred_img, caption="📊 AI 모델별 예측 현황 (최신 실행 결과)", use_container_width=True)
    else:
        st.info("🖼️ 예측 차트 이미지를 찾을 수 없습니다. `32FA_daily_predict_report_v7E.ipynb`를 실행하면 생성됩니다.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 상단 4단 요약 카드 ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.1])

    with c1:
        with st.container(border=True, key="c1_box"):
            st.markdown("#### 🎯 1d 메인 모델 (dynH)")
            pred_data = data["prediction"]

            if pred_data:
                pred_dir = pred_data.get('direction', '하락')
                is_up = pred_dir in ['상승', 'UP', 1]
                pred_label = "UP (상승)" if is_up else "DOWN (하락)"

                conf = pred_data.get('confidence_score', 0.522) * 100

                mb_str = pred_data.get('model_breakdown', '{}')
                try:
                    mb = json.loads(mb_str) if isinstance(mb_str, str) else mb_str
                except:
                    mb = {}

                regime = mb.get('regime', 'bear')
                expected_acc = mb.get('predicted_accuracy_pct', 66.8)

                clr_cls = "pred-up" if is_up else "pred-down"
                badge = "bull" if is_up else "bear"
                icon = "🚀 매수 신호" if is_up else "🛡️ 관망 권장"

                st.markdown(f"<span class='badge {badge}'>{icon}</span><span class='badge neutral'>Regime: {regime}</span>", unsafe_allow_html=True)
                st.markdown(f"<p class='highlight-val {clr_cls}'>{pred_label}</p>", unsafe_allow_html=True)

                st.markdown(
                    f"<div style='margin-bottom: 3px; font-size: 1.1rem;'><b>AI 신뢰도:</b> {conf:.1f}% <span style='color: #8b949e; font-size: 0.9rem;'>(최종 예측 확률)</span></div>"
                    f"<div style='margin-bottom: 12px; font-size: 1.1rem; color: #4ade80;'><b>예상 정확도:</b> {expected_acc:.1f}% <span style='color: #8b949e; font-size: 0.9rem;'>(유사 구간 과거 백테스트 승률)</span></div>",
                    unsafe_allow_html=True
                )

                st.progress(conf/100.0)

                with st.expander("🤖 개별 모델 확률 보기"):
                    indiv = mb.get("individual_predictions", {})
                    if indiv:
                        st.write("**개별 앙상블 모델 예측:**")
                        for m_name, p in indiv.items():
                            st.write(f"- {m_name}: {p:.4f}")
                        st.write("---")
                    stk = mb.get("meta_stacking_probability", "N/A")
                    dyn = mb.get("regime_probability", "N/A")
                    fin = mb.get("final_probability", "N/A")
                    st.write(f"**스태킹(Stacking):** {stk:.4f}" if isinstance(stk, float) else f"**스태킹:** {stk}")
                    st.write(f"**레짐 동적 앙상블:** {dyn:.4f}" if isinstance(dyn, float) else f"**레짐 동적 앙상블:** {dyn}")
                    st.write(f"**최종 융합:** {fin:.4f}" if isinstance(fin, float) else f"**최종 융합:** {fin}")
                    st.caption(f"예측 기준일: {pred_data.get('date', 'N/A')[:16].replace('T', ' ')}")
            else:
                st.warning("예측 데이터를 불러오고 있습니다...")

    # ── ④ +7d 변동성 전망 — weekly_predictions 테이블 기반 ─────────────────────
    with c2:
        with st.container(border=True, key="c2_box"):
            st.markdown("#### 🌪️ +7d 시장 변동성 전망")
            wp = data.get("weekly_prediction", {})

            if wp:
                w_pred = wp.get("prediction", 1)
                w_conf = wp.get("confidence", 0.55)
                w_boundary = wp.get("boundary", 0.019)
                w_target_hits = wp.get("target_hits", 0)
                w_week_start = wp.get("prediction_week_start", "")
                w_p_active = wp.get("p_active", w_conf)
                w_model = wp.get("model_version", "")

                # prediction=1 → ACTIVE(고변동성), prediction=0 → QUIET(저변동성)
                is_active = int(w_pred) == 1
                boundary_pct = w_boundary * 100

                if is_active:
                    w_label = "🔥 고변동성 주간 (ACTIVE)"
                    w_badge = "bear"   # 빨간색 — 위험 강조
                else:
                    w_label = "💤 저변동성 주간 (QUIET)"
                    w_badge = "neutral"  # 회색 — 조용함

                st.markdown(f"<span class='badge {w_badge}'>{w_label}</span>", unsafe_allow_html=True)
                st.write(f"P(Active): **{w_p_active*100:.1f}%** / 신뢰도: **{w_conf*100:.1f}%**")
                st.write(f"변동 기준선: **±{boundary_pct:.2f}%** | 터치 예상: **{w_target_hits}회** / 주")
                st.caption(f"* 7일 중 ±{boundary_pct:.2f}% 초과 움직임이 {w_target_hits}회 이상 → ACTIVE로 분류")
                st.caption(f"모델: {w_model} | 예측 주간 시작: {w_week_start[:10] if w_week_start else ''}")

                if is_active:
                    st.warning(
                        "⚠️ **전략 가이드 (고변동성 주간)**\n"
                        f"이번 주는 일간 변동폭 ±{boundary_pct:.2f}%를 {w_target_hits}회 이상 돌파하는 **고변동성** 주간으로 예측됩니다."
                        f" (AI 신뢰도 {w_conf*100:.1f}%)\n\n"
                        "• 포지션 규모를 평소보다 **축소**하여 리스크를 관리하세요.\n"
                        "• 1d 모델의 신호가 발생해도 **빠른 이익 실현 / 분할 매도** 전략이 유효합니다.\n"
                        "• 예상치 못한 급등락에 대비해 손절 라인을 반드시 설정하세요."
                    )
                else:
                    st.info(
                        "💡 **전략 가이드 (저변동성 주간)**\n"
                        f"이번 주는 일간 변동폭 ±{boundary_pct:.2f}% 이내에서 움직이는 **저변동성** 주간으로 예측됩니다."
                        f" (AI 신뢰도 {w_conf*100:.1f}%)\n\n"
                        "• 잦은 단타보다 1d 모델의 **고신뢰도(65%+)** 신호에만 집중하세요.\n"
                        "• 큰 방향 전환보다는 좁은 박스권 내 움직임이 예상됩니다.\n"
                        "• 무리한 추격 매수/매도를 피하고 신호 대기 위주로 대응하세요."
                    )

            else:
                st.markdown("<span class='badge neutral'>주간 예측 데이터 없음</span>", unsafe_allow_html=True)
                st.caption("weekly_predictions 테이블에 데이터가 없거나 연결 실패")

    with c3:
        with st.container(border=True, key="c3_box"):
            st.markdown("#### 💹 핵심 마켓 데이터")
            m_data = data["market"][0] if data["market"] else {}
            if m_data:
                krw = m_data.get('btc_krw_price', 0)
                usd = m_data.get('btc_usd_price', 0)
                kimchi = m_data.get('kimchi_premium', 0)
                ex_rate = m_data.get('usd_krw_rate', 0)

                st.metric("BTC (KRW)", format_krw(krw), delta=f"김치프리미엄 {kimchi:.2f}%", delta_color="inverse" if kimchi > 2 else "normal")
                st.write(f"**BTC (USD)**: ${usd:,.2f}")
                st.write(f"**원/달러 환율**: {ex_rate:,.1f}원")
                st.caption(f"Update: {m_data.get('timestamp', '')[:16]}")
            else:
                st.write("마켓 데이터 수신 대기 중...")

    with c4:
        with st.container(border=True, key="c4_box"):
            st.markdown("#### 🎯 30일 타율 & 최근 기록")
            
            acc = data["acc_30d"]
            acc_pct = (acc["correct"]/acc["total"]*100) if acc["total"] > 0 else 0
            
            st.markdown(f"""
            <div style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 12px; text-align: center; margin-bottom: 1rem; border: 1px solid rgba(255,255,255,0.05); box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);">
                <div style="font-size: 0.9rem; color: #94a3b8; font-weight: 600;">최근 30일 AI 예측 타율</div>
                <div style="font-size: 2.5rem; font-weight: 900; color: #4ade80; text-shadow: 0 0 10px rgba(74,222,128,0.4);">{acc_pct:.1f}%</div>
                <div style="font-size: 0.85rem; color: #64748b;">✅ {acc['correct']} 성공 / {acc['total']} 전체</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.8rem; padding-bottom: 0.3rem; border-bottom: 1px solid rgba(255,255,255,0.1);'>🗓️ 최근 예측 vs 실제 결과</div>", unsafe_allow_html=True)
            
            history = acc.get("history", [])
            if history:
                feed_html = "<div style='max-height: 220px; overflow-y: auto; padding-right: 5px; display: flex; flex-direction: column; gap: 8px;' class='custom-scrollbar'>"
                for row in history[:15]: # Show up to 15 recent predictions
                    dt = row.get("date", "")[:10]
                    dir_val = row.get("direction", "")
                    is_correct = row.get("is_correct")
                    conf = row.get("confidence_score", 0) * 100
                    
                    if is_correct is True:
                        badge = "<span style='color: #22c55e; background: rgba(34,197,94,0.15); padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight:bold; border: 1px solid rgba(34,197,94,0.4); box-shadow: 0 0 5px rgba(34,197,94,0.2);'>적중 ✅</span>"
                    elif is_correct is False:
                        badge = "<span style='color: #ef4444; background: rgba(239,68,68,0.15); padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight:bold; border: 1px solid rgba(239,68,68,0.4); box-shadow: 0 0 5px rgba(239,68,68,0.2);'>실패 ❌</span>"
                    else:
                        badge = "<span style='color: #94a3b8; background: rgba(255,255,255,0.05); padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; border: 1px solid rgba(255,255,255,0.1);'>결과 대기</span>"
                        
                    dir_icon = "📈" if dir_val in ["상승", "UP", "상승장", 1] else "📉" if dir_val in ["하락", "DOWN", "하락장", 0] else "➖"
                    dir_color = "#4ade80" if dir_icon == "📈" else "#f87171" if dir_icon == "📉" else "#94a3b8"
                    
                    feed_html += f"""
<div style="background: rgba(255,255,255,0.02); padding: 0.75rem 1rem; border-radius: 8px; display: flex; justify-content: space-between; align-items: center; border: 1px solid rgba(255,255,255,0.05); transition: background 0.2s;" onmouseover="this.style.background='rgba(255,255,255,0.06)'" onmouseout="this.style.background='rgba(255,255,255,0.02)'">
    <div>
        <div style="font-size: 0.7rem; color: #64748b; margin-bottom: 2px;">{dt} 예측</div>
        <div style="font-size: 0.95rem; font-weight: 800; color: {dir_color};">{dir_val} {dir_icon}</div>
    </div>
    <div style="text-align: right; display:flex; flex-direction:column; align-items:flex-end; gap:4px;">
        {badge}
        <div style="font-size: 0.7rem; color: #64748b;">신뢰도 {conf:.1f}%</div>
    </div>
</div>
"""
                feed_html += "</div>"
                
                # Custom scrollbar style inline
                feed_html += """
                <style>
                .custom-scrollbar::-webkit-scrollbar { width: 6px; }
                .custom-scrollbar::-webkit-scrollbar-track { background: rgba(0,0,0,0.1); border-radius: 4px; }
                .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 4px; }
                .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.25); }
                </style>
                """
                st.markdown(clean_html(feed_html), unsafe_allow_html=True)
            else:
                st.caption("최근 기록이 아직 수집되지 않았습니다.")

    # ── ②③ 종합 기술적 분석 — 인터랙티브 지표 선택 + 30일 그래프 ──────────────
    st.markdown("#### 📊 종합 기술적 분석")
    t1, t2, t3, t4 = st.columns(4)
    f_data = data["features"]

    if f_data:
        rsi = f_data.get("RSI_14", 50)
        macd = f_data.get("MACD", 0)
        macd_sig = f_data.get("MACD_signal", 0)
        bb_pos = f_data.get("BB_position", 0.5)
        # ③ F&G 컬럼명 수정: fng_value
        fng = f_data.get("fng_value", None)
        fng_display = f"{int(fng)}" if fng is not None else "N/A"

        with t1:
            st.metric("RSI (14일)", f"{rsi:.1f}", "과매수 🔴" if rsi > 70 else "과매도 🟢" if rsi < 30 else "중립 ⚪", delta_color="off")
            st.caption("황도 70 이상이면 과매수(과열), 30 이하면 과매도(반등) 신호. 50 기준으로 상승세/하락세 판단.")
        with t2:
            st.metric("MACD 지표", f"{macd:.1f}", "골든크로스 🟢" if macd > macd_sig else "데드크로스 🔴", delta_color="off")
            st.caption("단기 이동평균 - 장기 이동평균. 시그널선을 위로 돌파(골든크로스)하면 상승 신호.")
        with t3:
            st.metric("볼린저 밴드", f"{bb_pos*100:.0f}%", "상단 돌파 위험 🔴" if bb_pos > 0.8 else "하단 반등 기대 🟢" if bb_pos < 0.2 else "밴드 내 ⚪", delta_color="off")
            st.caption("검락도(BB) 범위 안에서 현재가의 위치. 80%+ 면 상단에 근접, 20%- 면 하단 근접.")
        with t4:
            fng_delta = "탐욕" if fng is not None and fng > 60 else "공포" if fng is not None and fng < 40 else "중립"
            st.metric("공포/탐욕 지수", fng_display, fng_delta, delta_color="off")
            st.caption("0(최고 공포)~100(최고 탐욕). 75 이상은 과열 경보, 25 이하는 분할 매수 기회 신호.")



    # ── ② 지표 선택 → 30일 시계열 그래프 ──────────────────────────────────────
    st.markdown("---")
    st.markdown("##### 📈 지표별 30일 시계열 그래프")
    st.caption("아래에서 보고 싶은 지표를 선택하면 최근 30일 추이를 확인할 수 있습니다.")

    indicator_choice = st.radio(
        "지표 선택",
        ["RSI (14일)", "MACD", "볼린저 밴드 위치", "공포/탐욕 지수"],
        horizontal=True,
        label_visibility="collapsed"
    )

    features_30d = data.get("features_30d", [])
    if features_30d:
        df30 = pd.DataFrame(features_30d)
        df30['date'] = pd.to_datetime(df30['date'])
        df30 = df30.sort_values('date')

        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(22,27,34,0.8)',
            font=dict(color='#c9d1d9'),
            height=320,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickformat='%m/%d'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )

        if indicator_choice == "RSI (14일)":
            fig.add_trace(go.Scatter(x=df30['date'], y=df30['RSI_14'], name='RSI', line=dict(color='#f59e0b', width=2)))
            fig.add_hline(y=70, line_dash="dot", line_color="#f87171", annotation_text="과매수 70")
            fig.add_hline(y=30, line_dash="dot", line_color="#4ade80", annotation_text="과매도 30")
            fig.update_layout(yaxis_title="RSI", yaxis_range=[0, 100])

        elif indicator_choice == "MACD":
            df30['MACD_hist'] = df30['MACD'] - df30['MACD_signal']
            colors = ['#4ade80' if v >= 0 else '#f87171' for v in df30['MACD_hist']]
            fig.add_trace(go.Bar(x=df30['date'], y=df30['MACD_hist'], name='MACD 히스토그램', marker_color=colors, opacity=0.7))
            fig.add_trace(go.Scatter(x=df30['date'], y=df30['MACD'], name='MACD', line=dict(color='#58a6ff', width=2)))
            fig.add_trace(go.Scatter(x=df30['date'], y=df30['MACD_signal'], name='Signal', line=dict(color='#f97316', width=1.5, dash='dot')))
            fig.update_layout(yaxis_title="MACD")

        elif indicator_choice == "볼린저 밴드 위치":
            fig.add_trace(go.Scatter(x=df30['date'], y=df30['BB_position']*100, name='BB 위치 %', line=dict(color='#a78bfa', width=2), fill='tozeroy', fillcolor='rgba(167,139,250,0.1)'))
            fig.add_hline(y=80, line_dash="dot", line_color="#f87171", annotation_text="상단 80%")
            fig.add_hline(y=20, line_dash="dot", line_color="#4ade80", annotation_text="하단 20%")
            fig.update_layout(yaxis_title="볼린저 밴드 위치 (%)", yaxis_range=[0, 100])

        elif indicator_choice == "공포/탐욕 지수":
            if 'fng_value' in df30.columns:
                df_fng = df30.dropna(subset=['fng_value'])
                colors_fng = ['#4ade80' if v > 60 else '#f87171' if v < 40 else '#f59e0b' for v in df_fng['fng_value']]
                fig.add_trace(go.Bar(x=df_fng['date'], y=df_fng['fng_value'], name='F&G 지수', marker_color=colors_fng, opacity=0.85))
                fig.add_hline(y=60, line_dash="dot", line_color="#4ade80", annotation_text="탐욕 60")
                fig.add_hline(y=40, line_dash="dot", line_color="#f87171", annotation_text="공포 40")
                fig.update_layout(yaxis_title="공포/탐욕 지수", yaxis_range=[0, 100])
            else:
                st.info("30일 F&G 데이터가 없습니다.")

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("30일 기술적 지표 데이터를 불러오는 중입니다...")

# ==============================================================================
# 탭 2: 뉴스 감성 심층 분석
# ==============================================================================
with tab_news:
    st.markdown("### 🗞️ 글로벌 모멘텀 & 뉴스 감성 분석")

    st.markdown("""
        본 시스템은 매일 최신 금융/암호화폐 뉴스를 수집하고, GPT-4o-mini를 활용하여
        **거시경제 맥락(Impact)**과 **감성(Sentiment)** 스코어를 딥러닝 피처로 변환합니다.
    """)

    sent_list = data["sentiment_7d"]
    if sent_list:
        df_news = pd.DataFrame(sent_list)
        df_news['date'] = pd.to_datetime(df_news['date'])

        recent_7d = df_news.head(7)
        avg_score = recent_7d['sentiment_score'].mean()

        # 상단 게이지 박스 헤더 (가로형 애니메이션 프로그레스 바 포함)
        # score = -1.0 ~ 1.0 -> percent = 0% ~ 100% (where 0 is 50%)
        # percent = (avg_score + 1.0) / 2.0 * 100
        gauge_percent = max(0, min(100, (avg_score + 1.0) / 2.0 * 100))
        gauge_color = "#4ade80" if avg_score > 0.1 else "#f87171" if avg_score < -0.1 else "#fcd34d"

        st.markdown(clean_html(f"""
        <style>
            @keyframes fillBar {{
                from {{ width: 50%; opacity: 0; }}
                to {{ width: {gauge_percent}%; opacity: 1; }}
            }}
            .sentiment-bar {{
                height: 100%;
                border-radius: 8px;
                background: linear-gradient(90deg, rgba(239,68,68,0.8) 0%, rgba(252,211,77,0.8) 50%, rgba(34,197,94,0.8) 100%);
                width: {gauge_percent}%;
                box-shadow: 0 0 10px {gauge_color};
                animation: fillBar 1.5s ease-out forwards;
                position: relative;
                overflow: hidden;
            }}
            /* 물결 효과(출렁거림) 추가 */
            .sentiment-bar::after {{
                content: '';
                position: absolute;
                top: 0; left: 0; right: 0; bottom: 0;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                animation: shimmer 2s infinite;
            }}
            @keyframes shimmer {{
                0% {{ transform: translateX(-100%); }}
                100% {{ transform: translateX(100%); }}
            }}
        </style>
        <div class="glass-card" style="margin-bottom: 2rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 1rem; margin-bottom: 1rem;">
                <div style="font-size: 1.25rem; font-weight: 800; color: #e2e8f0;">📊 주간 평균 감성 지표 (Sentiment)</div>
                <div style="font-size: 1.5rem; font-weight: 900; color: {gauge_color};">{avg_score:.2f}</div>
            </div>
            
            <!-- 가로 막대 그래프 (게이지바) -->
            <div style="margin-bottom: 1.5rem;">
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.5rem; font-weight: bold;">
                    <span>-1.0 (극단적 공포/거시 악재)</span>
                    <span>0.0 (중립)</span>
                    <span>+1.0 (극단적 탐욕/거시 호재)</span>
                </div>
                <div style="width: 100%; background: rgba(0,0,0,0.4); height: 24px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); position: relative;">
                    <!-- 중앙 0점 마커 -->
                    <div style="position: absolute; left: 50%; top: -4px; bottom: -4px; width: 2px; background: rgba(255,255,255,0.3); z-index: 10;"></div>
                    <!-- 실제 게이지 -->
                    <div class="sentiment-bar"></div>
                </div>
            </div>
            
            <div style="color: #cbd5e1; font-size: 1.05rem; line-height: 1.6;">
                {'<span style="color:#4ade80;">🟢 <strong>주간 모멘텀 긍정적:</strong></span> 기관 매수세, 호재성 뉴스가 가격 하락을 강하게 방어하고 있습니다.' if avg_score > 0.3 else '<span style="color:#f87171;">🔴 <strong>주간 모멘텀 부정적:</strong></span> 거시적 불안감 혹은 악재가 하방 압력을 높이고 있습니다.' if avg_score < -0.3 else '<span style="color:#94a3b8;">⚪ <strong>주간 모멘텀 중립적:</strong></span> 뚜렷한 재료 없이 기술적 지표에 의해 방향이 결정될 확률이 높습니다.'}
            </div>
        </div>
        """), unsafe_allow_html=True)

        st.markdown("#### 🕒 최근 14일 헤드라인 분석 피드")

        html_feed = "<div style='display:flex; flex-direction:column; gap:1.25rem;'>"
        for idx, row in df_news.iterrows():
            date_str = row['date'].strftime("%Y-%m-%d")
            score = row.get('sentiment_score', 0)
            imp = row.get('impact_score', 0)
            head = row.get('headline_summary', '(API 로딩 실패 또는 빈 헤드라인)')

            s_badge_color = "rgba(34, 197, 94, 0.15)" if score > 0.3 else ("rgba(239, 68, 68, 0.15)" if score < -0.3 else "rgba(148, 163, 184, 0.15)")
            s_text_color = "#4ade80" if score > 0.3 else ("#f87171" if score < -0.3 else "#94a3b8")
            s_border = "rgba(34,197,94,0.3)" if score > 0.3 else ("rgba(239,68,68,0.3)" if score < -0.3 else "rgba(148,163,184,0.3)")
            s_icon = "🟢" if score > 0.3 else ("🔴" if score < -0.3 else "⚪")

            if imp >= 0.8:
                imp_style = "background: rgba(34, 211, 238, 0.2); color: #22d3ee; border: 1px solid rgba(34,211,238,0.5); box-shadow: 0 0 10px rgba(34,211,238,0.3);"
                imp_icon = "🔥"
            else:
                imp_style = "background: rgba(148, 163, 184, 0.1); color: #94a3b8; border: 1px solid rgba(148,163,184,0.2);"
                imp_icon = "⚡"

            html_feed += f"""
            <div class="glass-card" style="padding: 1.5rem; margin-bottom: 0.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem; flex-wrap: wrap; gap: 10px;">
                    <div style="display: flex; gap: 8px; align-items: center;">
                        <span style="background: {s_badge_color}; color: {s_text_color}; border: 1px solid {s_border}; padding: 4px 12px; border-radius: 16px; font-size: 13px; font-weight: 700;">{s_icon} 감성 {score:.2f}</span>
                        <span style="{imp_style} padding: 4px 12px; border-radius: 16px; font-size: 13px; font-weight: 800;">{imp_icon} 임팩트 {imp:.2f}</span>
                    </div>
                    <span style="color: #64748b; font-size: 13px; font-weight: 500;">{date_str}</span>
                </div>
                <div style="color: #e2e8f0; font-size: 1.1rem; font-weight: 500; line-height: 1.5;">{head}</div>
            </div>
            """
        html_feed += "</div>"
        st.markdown(clean_html(html_feed), unsafe_allow_html=True)
    else:
        st.info("최근 뉴스 감성 데이터가 존재하지 않습니다.")

# ==============================================================================
# 탭 3: 퀀트 모델 차트 (리디자인)
# ==============================================================================
with tab_charts:
    st.markdown("### 🔬 퀀트 모델 검증 및 차트 브리핑")
    st.markdown("""
    <p style='color: #8b949e; font-size: 0.95rem; margin-bottom: 1.5rem;'>
    AI 파이프라인이 매일 자동 생성하는 분석 차트입니다. 각 그래프는 모델 예측 근거 및 성과 검증에 사용됩니다.
    </p>
    """, unsafe_allow_html=True)

    # 차트 메타데이터 (이름 → 제목, 설명, 배지색)
    CHART_META = [
        {
            "name": "chart_price_v7e.png",
            "title": "📈 가격 추이 & 24H AI 예측",
            "badge": "Price Forecast",
            "badge_color": "rgba(34,197,94,0.15)",
            "badge_text": "#4ade80",
            "border": "rgba(34,197,94,0.4)",
            "desc": "최근 7일간 BTC 가격 흐름과 AI가 예측한 24시간 후 목표가 범위. 초록 음영은 AI 예측 상승 구간(±1σ)입니다."
        },
        {
            "name": "chart_models_v7e.png",
            "title": "🤖 AI 모델별 예측 현황",
            "badge": "Ensemble",
            "badge_color": "rgba(249,115,22,0.15)",
            "badge_text": "#f97316",
            "border": "rgba(249,115,22,0.4)",
            "desc": "PatchTST(트랜스포머) · CNN-LSTM(딥러닝) · CatBoost(기술적) 3개 모델의 개별 확률과 최종 앙상블 결과 비교."
        },
        {
            "name": "chart_band_v7e.png",
            "title": "📊 신뢰구간 예측 밴드",
            "badge": "95% CI Band",
            "badge_color": "rgba(59,130,246,0.15)",
            "badge_text": "#60a5fa",
            "border": "rgba(59,130,246,0.4)",
            "desc": "Monte Carlo 시뮬레이션으로 계산한 95%/68% 신뢰구간. 음영 폭이 넓을수록 예측 불확실성이 높습니다."
        },
        {
            "name": "backtest_v7e.png",
            "title": "💹 백테스트 수익률 검증",
            "badge": "Backtest",
            "badge_color": "rgba(168,85,247,0.15)",
            "badge_text": "#c084fc",
            "border": "rgba(168,85,247,0.4)",
            "desc": "v7E 모델로 과거를 재현한 Long/Short 전략의 누적 수익률. BTC 단순 보유(Buy & Hold) 대비 AI 전략 성과 비교."
        },
    ]

    search_dirs = [
        "c:\\25WinterProject",
        "c:\\25WinterProject\\insta_image",
        "c:\\25WinterProject\\models\\production\\v7E_production_highAccuracy_dynH"
    ]

    def resolve_chart(name):
        # Supabase Storage 우선
        url = get_chart_url(name)
        if url:
            return url
        # 로컬 폴백
        for d in search_dirs:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
        return None

    # 2열 그리드 렌더링
    col_left, col_right = st.columns(2, gap="medium")
    cols = [col_left, col_right]

    any_found = False
    for i, meta in enumerate(CHART_META):
        src = resolve_chart(meta["name"])
        if not src:
            continue
        any_found = True
        with cols[i % 2]:
            st.markdown(f"""
            <div class="glass-card" style="border-color: {meta['border']}; margin-bottom: 1.25rem;">
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:0.5rem;">
                    <span style="font-size:1.1rem; font-weight:700; color:#e2e8f0;">{meta['title']}</span>
                    <span style="
                        background:{meta['badge_color']};
                        color:{meta['badge_text']};
                        border:1px solid {meta['border']};
                        font-size:11px; font-weight:700;
                        padding:2px 10px; border-radius:99px;
                    ">{meta['badge']}</span>
                </div>
                <p style="color:#94a3b8; font-size:0.82rem; margin:0 0 0.75rem 0; line-height:1.5;">{meta['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
            st.image(src, use_container_width=True)
            st.markdown("<div style='margin-bottom:0.5rem'></div>", unsafe_allow_html=True)

    if not any_found:
        st.info("차트 이미지가 없습니다. Supabase Storage `charts` 버킷에 업로드하거나 파이프라인을 실행하면 자동 생성됩니다.")

# ==============================================================================
# 탭 4: 일간/주간 마켓 리포트
# ==============================================================================
with tab_report:
    st.markdown("### 📝 일간/주간 마켓 리포트")
    st.markdown("<p style='color: #8b949e; font-size: 0.95rem; margin-bottom: 1.5rem;'>일간 AI 예측 리포트 및 종합 마켓 분석 리포트를 확인합니다.</p>", unsafe_allow_html=True)

    import requests as _rq
    import re

    def _fetch_report_direct(prefix):
        """Storage 전체 목록 조회 후 Python에서 prefix 필터링"""
        if not SUPABASE_URL or not SUPABASE_KEY:
            return None, "데이터 소스 미설정"
        try:
            headers = {
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json"
            }
            list_url = f"{SUPABASE_URL}/storage/v1/object/list/{CHARTS_BUCKET}"
            r = _rq.post(list_url, headers=headers,
                         json={"prefix": "", "sortBy": {"column": "name", "order": "desc"}},
                         timeout=10)
            if not r.ok:
                return None, "데이터 로딩 실패"
            all_files = r.json()
            if not isinstance(all_files, list):
                return None, "데이터 로딩 실패"
            matching = [f for f in all_files if isinstance(f, dict) and f.get('name', '').startswith(prefix)]
            if not matching:
                return None, "파일 없음"
            fname = sorted(matching, key=lambda x: x['name'])[-1]['name']
            file_url = f"{SUPABASE_URL}/storage/v1/object/public/{CHARTS_BUCKET}/{fname}"
            fr = _rq.get(file_url, timeout=15)
            if not fr.ok:
                return None, "데이터 로딩 실패"
            return fr.content.decode('utf-8'), fname
        except Exception:
            return None, "데이터 로딩 실패"

    def parse_daily_report(text):
        data = { "date": "N/A", "direction": "N/A", "acc": "N/A", "conf": "N/A", "summary": "", "models": [] }
        lines = text.split('\n')
        # Extract headers
        for line in lines:
            if "분석 시점:" in line:
                try: data["date"] = line.split("분석 시점:")[-1].strip()
                except: pass
            elif "최종 예측:" in line:
                m = re.search(r"최종 예측:\s*([^\s—]+)\s*—.*?정확도\s*(\d+%?).*?신뢰도\s*(\d+%?)", line)
                if m:
                    data["direction"] = m.group(1).strip()
                    data["acc"] = m.group(2).strip()
                    data["conf"] = m.group(3).strip()
        # Extract summary
        try:
            idx = lines.index("📌 한줄 요약")
            summary = ""
            for i in range(idx+1, len(lines)):
                if "━" in lines[i]: break
                if lines[i].strip():
                    summary += lines[i].strip() + " "
            data["summary"] = summary.strip()
        except: pass
        
        # Extract models
        current_model = None
        for line in lines:
            line_clean = line.strip()
            if line_clean.startswith("④") or line_clean.startswith("⑤") or line_clean.startswith("⑥") or line_clean.startswith("━"):
                if current_model:
                    data["models"].append(current_model)
                    current_model = None
            elif line_clean.startswith("①") or line_clean.startswith("②") or line_clean.startswith("③"):
                if current_model:
                    data["models"].append(current_model)
                name = line_clean.split('-', 1)[0].replace('①','').replace('②','').replace('③','').strip()
                current_model = {"name": name, "raw_lines": []}
                if '-' in line_clean:
                    current_model["raw_lines"].append(line_clean.split('-', 1)[1].strip())
            elif current_model and line_clean:
                current_model["raw_lines"].append(line_clean)
                
        if current_model:
            data["models"].append(current_model)
            
        # Post-process models
        for m in data["models"]:
            # Name cleanup
            m["name"] = m["name"].replace('\n', ' ').strip()
            if "(Kaggle 전체 학습)" in m["name"]:
                m["name"] = m["name"].replace("(Kaggle 전체 학습)", "").strip()
            if "Fold 앙상블" in m["name"]:
                m["name"] = m["name"].replace("(5 Fold 앙상블)", "").replace("(5", "").replace("Fold 앙상블)", "").replace(")", "").strip()
                
            # Add roles
            if "PatchTST" in m["name"]:
                m["name"] = f"PatchTST <span style='font-size:0.8rem; font-weight:normal; color:#8b949e;'>(장기 패턴 학습)</span>"
            elif "CNN" in m["name"]:
                m["name"] = f"CNN <span style='font-size:0.8rem; font-weight:normal; color:#8b949e;'>(단기 패턴 학습)</span>"
            elif "CatBoost" in m["name"]:
                m["name"] = f"CatBoost <span style='font-size:0.8rem; font-weight:normal; color:#8b949e;'>(기술적 추세 학습)</span>"
            
            # Process lines
            raw_text = " ".join(m["raw_lines"])
            
            # Find %
            pct_m = re.search(r"(\d+(?:\.\d+)?)%", raw_text)
            pct = pct_m.group(0) if pct_m else ""
            
            # Find direction
            dir_val = "중립 ➖"
            if "상승" in raw_text: dir_val = "상승 📈"
            elif "하락" in raw_text: dir_val = "하락 📉"
            
            m["val"] = f"{pct} {dir_val}".strip()
            
            # Extract desc (filter out noise, percentages, arrows, direction words)
            desc_lines = []
            for rl in m["raw_lines"]:
                # skip lines that are just numbers or arrows
                rl = rl.replace('→', '').replace('📈', '').replace('📉', '').replace('➖', '')
                if pct:
                    rl = rl.replace(pct, "").replace("-", "").strip()
                
                rl_clean = rl.replace("상승", "").replace("하락", "").replace("중립", "").replace("예측", "").strip()
                
                if len(rl_clean) > 2: # Has actual content
                    desc_lines.append(rl.strip(' -,'))
            
            desc_text = " ".join([dl for dl in desc_lines if dl]).strip()
            if desc_text.startswith("-"): desc_text = desc_text[1:].strip()
            
            m["desc"] = desc_text
            
        return data

    def render_daily_ui(data, raw_text):
        if not data or data["direction"] == "N/A":
            st.code(raw_text, language="markdown")
            return
        
        is_up = "상승" in data["direction"]
        bg_color = "rgba(34, 197, 94, 0.1)" if is_up else "rgba(239, 68, 68, 0.1)"
        border_color = "rgba(34, 197, 94, 0.3)" if is_up else "rgba(239, 68, 68, 0.3)"
        text_color = "#4ade80" if is_up else "#f87171"
        icon = "🚀" if is_up else "🛡️"
        
        html = f"""
<div class="glass-card" style="background: {bg_color}; border-color: {border_color}; margin-bottom: 1.5rem;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; gap: 10px;">
        <div>
            <span style="font-size: 14px; color: #94a3b8;">분석 시점: {data['date']}</span>
            <h3 style="margin: 0; padding: 0; color: {text_color}; font-size: 2rem;">{icon} {data['direction']} 예측</h3>
        </div>
        <div style="text-align: right; background: rgba(0,0,0,0.3); padding: 0.75rem 1.25rem; border-radius: 8px;">
            <div style="font-size: 13px; color: #94a3b8; display: inline-block; margin-right: 1.5rem;">기대 정확도 <br><span style="color: #e2e8f0; font-weight: bold; font-size: 1.3rem;">{data['acc']}</span></div>
            <div style="font-size: 13px; color: #94a3b8; display: inline-block;">AI 신뢰도 <br><span style="color: #e2e8f0; font-weight: bold; font-size: 1.3rem;">{data['conf']}</span></div>
        </div>
    </div>
    <div style="background: rgba(0,0,0,0.25); padding: 1.25rem; border-radius: 8px; border-left: 4px solid {text_color};">
        <span style="color: #e2e8f0; font-size: 1.05rem; line-height: 1.6;">{data['summary']}</span>
    </div>
</div>
"""
        st.markdown(clean_html(html), unsafe_allow_html=True)
        
        if data["models"]:
            cols = st.columns(len(data["models"]))
            for i, m in enumerate(data["models"]):
                m_val = m["val"]
                m_color = "#4ade80" if "상승" in m_val else "#f87171" if "하락" in m_val else "#94a3b8"
                with cols[i]:
                    st.markdown(clean_html(f"""
                    <div class="glass-card" style="padding: 1.25rem; height: 100%;">
                        <div style="font-weight: 800; color: #e2e8f0; margin-bottom: 0.5rem; font-size: 1.15rem;">{m['name']}</div>
                        <div style="color: {m_color}; font-weight: 800; margin-bottom: 0.75rem; font-size: 1.05rem;">{m['val']}</div>
                        <div style="color: #94a3b8; font-size: 0.9rem; line-height: 1.5;">{m['desc'].strip()}</div>
                    </div>
                    """), unsafe_allow_html=True)
               
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📄 [클릭] 일간 리포트 전문 보기"):
            st.code(raw_text, language="markdown")

    def parse_weekly_report(text, weekly_db_data=None):
        data = { "date": "N/A", "period": "N/A", "summary": "", "risk": {}, "scenario": [], "points": [] }
        lines = text.split('\n')
        
        # 날짜(period)를 텍스트 파싱 대신, 확실한 DB 데이터를 기준으로 생성합니다.
        if weekly_db_data and weekly_db_data.get("prediction_week_start"):
            try:
                # DB의 prediction_week_start (예: "2026-03-02")
                start_str = str(weekly_db_data["prediction_week_start"])
                # "YYYY-MM-DD" 포맷이라고 가정
                start_date = datetime.strptime(start_str[:10], "%Y-%m-%d").date()
                end_date = start_date + timedelta(days=6)
                data["period"] = f"{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"
            except Exception as e:
                pass

        for line in lines:
            line = line.strip()
            # Date Parsing (작성일 / 분석일 등)
            if "리포트 작성일:" in line:
                data["date"] = line.split("리포트 작성일:")[-1].strip()
            elif "분석일:" in line:
                data["date"] = line.split("분석일:")[-1].strip()
            elif "분석 기준일:" in line:
                data["date"] = line.split("분석 기준일:")[-1].strip()
            
            # Period Parsing 백업 (DB 데이터가 없을 경우에만 텍스트 파싱 시도)
            if data["period"] == "N/A":
                if "이번 주 예측 기간:" in line:
                    data["period"] = line.split("이번 주 예측 기간:")[-1].strip()
                elif "앞으로 7일:" in line:
                    data["period"] = line.split("앞으로 7일:")[-1].strip()
                elif "예측 기간:" in line:
                    data["period"] = line.split("예측 기간:")[-1].strip()
        try:
            idx = lines.index("📌 한줄 요약")
            summary = ""
            for i in range(idx+1, len(lines)):
                if "┌" in lines[i] or "━" in lines[i]: break
                if lines[i].strip():
                    summary += lines[i].strip() + " "
            data["summary"] = summary.strip()
        except: pass
        
        for line in lines:
            if "│" in line:
                parts = [p.strip() for p in line.split("│")]
                if len(parts) >= 3:
                    k, v = parts[1], parts[2]
                    v_clean = re.sub(r'<[^>]+>', '', v) # Strip HTML
                    if "신뢰도" in k: data["risk"]["daily"] = v_clean
                    if "변동성" in k: data["risk"]["weekly"] = v_clean
                    if "합의" in k: data["risk"]["model"] = v_clean
                    if "리스크" in k: data["risk"]["total"] = v_clean
                    
            if line.startswith("▶"):
                clean_scen = re.sub(r'<[^>]+>', '', line.replace("▶", "")).strip()
                data["scenario"].append(clean_scen)
                
            if line.startswith("①") or line.startswith("②") or line.startswith("③") or line.startswith("④"):
                clean_point = re.sub(r'<[^>]+>', '', line).strip()
                data["points"].append(clean_point)
                
        return data

    def render_weekly_ui(data, raw_text):
        if not data or not data["summary"]: 
            st.code(raw_text, language="markdown")
            return
        
        html = "" 
        html += f"""
<div style="background: linear-gradient(135deg, rgba(30,41,59,0.8) 0%, rgba(15,23,42,0.95) 100%); border: 1px solid rgba(148,163,184,0.25); border-radius: 12px; padding: 1.75rem; margin-bottom: 1.5rem; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3);">
    <div style="border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 1.25rem; margin-bottom: 1.5rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <div style="font-size: 1.4rem; font-weight: 800; color: #e2e8f0;">📆 이번 주 예측 기간: <span style="color: #60a5fa;">{data['period']}</span></div>
        </div>
        <div style="font-size: 14px; color: #94a3b8; background: rgba(0,0,0,0.3); padding: 4px 12px; border-radius: 16px;">
            작성일: {data['date']}
        </div>
    </div>
    
    <div style="background: rgba(0,0,0,0.25); padding: 1.25rem; border-radius: 8px; border-left: 4px solid #60a5fa; margin-bottom: 2rem;">
        <span style="color: #e2e8f0; font-size: 1.1rem; line-height: 1.6;">{data['summary']}</span>
    </div>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem;">
        <div>
            <div style="display:flex; align-items:center; gap:8px; margin-bottom: 1.25rem;">
                <span style="font-size: 1.2rem;">📋</span><h4 style="color: #e2e8f0; margin: 0; font-size: 1.15rem;">리스크 매트릭스</h4>
            </div>
            <div style="background: rgba(0,0,0,0.2); border-radius: 8px; padding: 0.5rem 1rem;">
"""
        
        risk = data.get("risk", {})
        for k, label in [("daily", "일간 신뢰도"), ("weekly", "주간 변동성"), ("model", "모델 합의"), ("total", "종합 리스크")]:
            val = risk.get(k, "N/A")
            val_color = "#f87171" if "ACTIVE" in val or "위험" in val or "⚠️" in val or "중상" in val else "#e2e8f0"
            html += f"""
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding: 1rem 0;">
                    <span style="color: #94a3b8; font-size:0.95rem;">{label}</span>
                    <span style="font-weight: 800; color: {val_color}; font-size:0.95rem;">{val}</span>
                </div>
            """
            
        html += """
            </div>
        </div>
        <div>
            <div style="display:flex; align-items:center; gap:8px; margin-bottom: 1.25rem;">
                <span style="font-size: 1.2rem;">🎯</span><h4 style="color: #e2e8f0; margin: 0; font-size: 1.15rem;">시나리오 분석</h4>
            </div>
"""
        for sc in data.get("scenario", []):
            icon = "📈" if "상승" in sc.split(":")[0] else "📉" if "하락" in sc.split(":")[0] else "▶"
            try:
                title, desc = sc.split(':', 1)
            except:
                title, desc = sc, ""
            title_color = "#4ade80" if "상승" in title else "#f87171" if "하락" in title else "#fb923c"
                
            html += f"""
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 1.25rem; margin-bottom: 1rem;">
                <div style="color: {title_color}; font-weight: 800; margin-bottom: 0.5rem; font-size:1.05rem;">{icon} {title}</div>
                <div style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.5;">{desc.strip()}</div>
            </div>
            """
            
        html += """
        </div>
    </div>
    
    <div style="margin-top: 2rem;">
        <div style="display:flex; align-items:center; gap:8px; margin-bottom: 1.25rem;">
            <span style="font-size: 1.2rem;">💡</span><h4 style="color: #e2e8f0; margin: 0; font-size: 1.15rem;">실행 포인트 (권장 전략)</h4>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1.25rem;">
"""
        
        for i, pt in enumerate(data.get("points", [])):
            parts = pt.split(':', 1)
            title = parts[0]
            desc = parts[1] if len(parts) > 1 else ""
            html += f"""
            <div style="background: rgba(56,189,248,0.1); border: 1px solid rgba(56,189,248,0.25); border-radius: 8px; padding: 1.25rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="color: #38bdf8; font-weight: 800; margin-bottom: 0.5rem; font-size: 1.05rem;">{title}</div>
                <div style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.5;">{desc.strip()}</div>
            </div>
            """
            
        html += """
        </div>
    </div>
</div>
"""
        st.markdown(clean_html(html), unsafe_allow_html=True)
        
        with st.expander("📄 [클릭] 주간 리포트 전문 보기"):
            st.code(raw_text, language="markdown")

    # 일간 예측 리포트 렌더링
    st.markdown("#### ✨ 일간 AI 예측 브리프")
    daily_text, daily_info = _fetch_report_direct('prediction_report_')
    if daily_text:
        parsed_daily = parse_daily_report(daily_text)
        render_daily_ui(parsed_daily, daily_text)
    else:
        st.info("일간 예측 리포트가 없습니다. 파이프라인을 실행하면 생성됩니다.")

    st.markdown("<br><br>", unsafe_allow_html=True)

    # 마켓 리포트 렌더링
    st.markdown("#### ✨ 주간 마켓 종합 애널리틱스")
    market_text, market_info = _fetch_report_direct('market_analysis_report_')
    if market_text:
        # DB에서 가져온 weekly_prediction 데이터를 넘겨주어 정확한 날짜 연산 보장
        weekly_db = data.get("weekly_prediction", {})
        parsed_weekly = parse_weekly_report(market_text, weekly_db)
        render_weekly_ui(parsed_weekly, market_text)
    else:
        st.info("마켓 리포트가 없습니다. 파이프라인을 실행해주세요.")


