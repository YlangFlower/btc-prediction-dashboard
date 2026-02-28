import streamlit as st
import pandas as pd
import os
import json
import glob
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, timezone
from supabase import create_client
from dotenv import load_dotenv

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
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px !important;
        background: rgba(22, 27, 34, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        padding: 0.5rem !important;
    }
    .badge {
        display: inline-block; padding: 4px 12px; border-radius: 16px; font-size: 13px; font-weight: 700; margin-bottom: 8px; margin-right: 8px;
    }
    .badge.bull { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
    .badge.bear { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
    .badge.neutral { background: rgba(148, 163, 184, 0.15); color: #94a3b8; border: 1px solid rgba(148,163,184,0.3); }
    .highlight-val { font-size: 2rem; font-weight: 800; margin: 0; padding: 0; }
    .pred-up { color: #4ade80; }
    .pred-down { color: #f87171; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 800 !important; }
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

        # 5. 30d Accuracy
        res_acc = supabase.table('predictions').select('is_correct').gte('date', thirty_days_ago).not_.is_('is_correct', 'null').execute()
        if res_acc.data:
            out["acc_30d"]["total"] = len(res_acc.data)
            out["acc_30d"]["correct"] = sum(1 for r in res_acc.data if r.get('is_correct'))

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
    content, fname = _load_text_from_storage(SUPABASE_URL, SUPABASE_KEY, 'prediction_report_')
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

    # ── 상단 3단 요약 카드 ──────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([1.2, 1, 1])

    with c1:
        with st.container(border=True):
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
        with st.container(border=True):
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

            acc = data["acc_30d"]
            acc_pct = (acc["correct"]/acc["total"]*100) if acc["total"] > 0 else 0
            st.markdown(f"<div style='margin-top: 5px; color:#8b949e;'>최근 30일 적중률: <strong style='color:#fff'>{acc_pct:.1f}%</strong> ({acc['correct']}/{acc['total']})</div>", unsafe_allow_html=True)

    with c3:
        with st.container(border=True):
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

        gc1, gc2 = st.columns([1, 2])
        with gc1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "주간 평균 감성 지표 (Sentiment)", 'font': {'color': '#c9d1d9'}},
                gauge={
                    'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "#c9d1d9"},
                    'bar': {'color': "#f59e0b"},
                    'bgcolor': "rgba(255,255,255,0.05)",
                    'steps': [
                        {'range': [-1, -0.3], 'color': "rgba(239, 68, 68, 0.4)"},
                        {'range': [-0.3, 0.3], 'color': "rgba(148, 163, 184, 0.2)"},
                        {'range': [0.3, 1.0], 'color': "rgba(34, 197, 94, 0.4)"}
                    ],
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#c9d1d9"}, height=250)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with gc2:
            st.markdown("#### 📌 딥러닝 입력 피처 기준 감성 평가")
            if avg_score > 0.3:
                st.success("🟢 **주간 모멘텀 긍정적:** 기관 매수세, 호재성 뉴스가 가격 하락을 강하게 방어하고 있습니다.")
            elif avg_score < -0.3:
                st.error("🔴 **주간 모멘텀 부정적:** 거시적 불안감 혹은 악재가 하방 압력을 높이고 있습니다.")
            else:
                st.info("⚪ **주간 모멘텀 중립적:** 뚜렷한 재료 없이 기술적 지표에 의해 방향이 결정될 확률이 높습니다.")

        st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        st.markdown("#### 🕒 최근 14일 헤드라인 분석 내역")

        for idx, row in df_news.iterrows():
            date_str = row['date'].strftime("%Y-%m-%d")
            score = row.get('sentiment_score', 0)
            imp = row.get('impact_score', 0)
            head = row.get('headline_summary', '(API 로딩 실패 또는 빈 헤드라인)')

            s_badge = "bull" if score > 0.3 else ("bear" if score < -0.3 else "neutral")
            s_txt = f"감성: {score:.2f}"
            # 임팩트 0.8 이상이면 형광 시안 강조
            if imp >= 0.8:
                imp_style = "background: rgba(34, 211, 238, 0.2); color: #22d3ee; border: 1px solid rgba(34,211,238,0.5); font-weight: 800;"
                imp_icon = "🔥"
            else:
                imp_style = "background: rgba(148, 163, 184, 0.15); color: #94a3b8; border: 1px solid rgba(148,163,184,0.3);"
                imp_icon = ""

            with st.container(border=True):
                col_h1, col_h2 = st.columns([1, 4])
                with col_h1:
                    st.markdown(
                        f"<span style='color:#8b949e; font-size: 14px;'>{date_str}</span><br>"
                        f"<span class='badge {s_badge}'>{s_txt}</span><br>"
                        f"<span style='display:inline-block; padding: 4px 12px; border-radius: 16px; font-size: 13px; margin-bottom:4px; {imp_style}'>{imp_icon} 임팩트: {imp:.2f}</span>",
                        unsafe_allow_html=True
                    )

                with col_h2:
                    st.write(head)
    else:
        st.write("최근 뉴스 감성 데이터가 존재하지 않습니다.")

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
            <div style="
                border: 1px solid {meta['border']};
                border-radius: 16px;
                padding: 1.25rem 1.25rem 0.75rem;
                margin-bottom: 1.25rem;
                background: rgba(22,27,34,0.6);
                backdrop-filter: blur(8px);
            ">
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
    st.markdown("일간 AI 예측 리포트 및 종합 마켓 분석 리포트를 확인합니다.")

    import requests as _rq

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

    # 일간 예측 리포트
    st.markdown("#### 📋 일간 AI 예측 리포트")
    daily_text, daily_info = _fetch_report_direct('prediction_report_')
    if daily_text:
        st.caption(f"파일: {daily_info}")
        with st.container(border=True):
            st.code(daily_text, language="markdown")
    else:
        st.info("일간 예측 리포트가 없습니다. `32FA_daily_predict_report_v7E.ipynb`를 실행하면 생성됩니다.")

    st.markdown("<br>", unsafe_allow_html=True)

    # 마켓 리포트
    st.markdown("#### 📊 일간/주간 마켓 종합 분석 리포트")
    market_text, market_info = _fetch_report_direct('market_analysis_report_')
    if market_text:
        st.caption(f"파일: {market_info}")
        with st.container(border=True):
            st.code(market_text, language="markdown")
    else:
        st.info("마켓 리포트가 없습니다. 해당 노트북을 실행해주세요.")

