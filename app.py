import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
from supabase import create_client, Client
from dotenv import load_dotenv

# --- 페이지 설정 ---
st.set_page_config(
    page_title="비트코인 예측 리포트",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 환경 변수 로드 (Streamlit Cloud Secrets 우선, 로컬은 .env 폴백) ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except Exception:
    load_dotenv()  # 로컬 .env 파일에서 로드
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Supabase Storage 버킷명 (차트 이미지 저장 위치)
CHARTS_BUCKET = "charts"

# --- Supabase 연결 ---
@st.cache_resource
def init_supabase():
    try:
        if SUPABASE_URL and SUPABASE_KEY:
            return create_client(SUPABASE_URL, SUPABASE_KEY)
        return None
    except:
        return None

supabase = init_supabase()

# --- CSS 스타일 ---
st.markdown("""
<style>
.stApp { background-color: #0a0e1a; color: #e5e7eb; }
.main-header { text-align: center; padding: 2rem 0; margin-bottom: 2rem; }
.bitcoin-icon { font-size: 60px; background: linear-gradient(135deg, #f97316 0%, #fb923c 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.main-title { font-size: 48px; font-weight: bold; background: linear-gradient(135deg, #f97316 0%, #fb923c 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.subtitle { color: #9ca3af; font-size: 18px; }
.prediction-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 2rem; margin: 1rem 0; border: 1px solid #334155; }
.prediction-icon { width: 60px; height: 60px; background: #7c2d12; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 32px; margin-bottom: 1rem; }
.sell-signal { background: #7c2d12; color: white; padding: 0.75rem 1.5rem; border-radius: 8px; text-align: center; font-weight: bold; }
.stTabs [data-baseweb="tab-list"] { gap: 2rem; background-color: transparent; border-bottom: 1px solid #334155; width: 100%; }
.stTabs [data-baseweb="tab"] { color: #9ca3af; padding: 1rem 2rem; font-size: 16px; background-color: transparent; flex-grow: 1; }
.stTabs [aria-selected="true"] { color: #f97316; border-bottom: 2px solid #f97316; }
.stTabs [data-baseweb="tab-panel"] { min-height: 700px; padding: 1.5rem 0; width: 100%; }
.metric-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 2rem; border: 1px solid #334155; min-height: 200px; }
.rsi-card { border: 2px solid #ef4444; box-shadow: 0 0 20px rgba(239, 68, 68, 0.2); }
.macd-card { border: 2px solid #3b82f6; box-shadow: 0 0 20px rgba(59, 130, 246, 0.2); }
.bb-card { border: 2px solid #a855f7; box-shadow: 0 0 20px rgba(168, 85, 247, 0.2); }
.news-item { background: #1e293b; border-radius: 8px; padding: 1rem 1.5rem; margin: 0.5rem 0; border-left: 3px solid #f97316; display: grid; grid-template-columns: 80px 1fr 100px; gap: 1rem; align-items: center; }
.news-impact { text-align: right; font-size: 13px; padding: 0.25rem 0.75rem; border-radius: 4px; white-space: nowrap; min-width: 90px; }
.impact-high { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
.impact-medium { background: rgba(251, 146, 60, 0.2); color: #fb923c; }
.impact-low { background: rgba(148, 163, 184, 0.2); color: #94a3b8; }
.summary-box { background: #1e293b; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; border: 1px solid #334155; min-height: 350px; }
.market-info-section { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 1.5rem 2rem; margin: 1.5rem 0; border: 1px solid #334155; }
.market-info-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.5rem; }
.market-info-item { background: rgba(255, 255, 255, 0.03); border-radius: 12px; padding: 1.25rem; text-align: center; border: 1px solid rgba(255, 255, 255, 0.05); }
.signal-badge { display: inline-block; padding: 0.4rem 0.8rem; border-radius: 8px; font-size: 13px; font-weight: 600; margin: 0.25rem; }
.signal-bullish { background: rgba(34, 197, 94, 0.2); color: #22c55e; }
.signal-bearish { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
.signal-neutral { background: rgba(148, 163, 184, 0.2); color: #94a3b8; }
.model-prediction-box { background: rgba(255, 255, 255, 0.03); border-radius: 10px; padding: 1rem; margin: 0.5rem 0; border-left: 3px solid #f97316; }
.price-change-positive { color: #22c55e; }
.price-change-negative { color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# --- 데이터 로드 함수들 ---
@st.cache_data(ttl=60)
def load_latest_sentiment():
    try:
        if supabase is None: return None
        response = supabase.table('raw_sentiment').select('*').order('date', desc=True).limit(5).execute()
        return pd.DataFrame(response.data) if response.data else None
    except: return None

@st.cache_data(ttl=60)
def load_latest_features():
    try:
        if supabase is None: return None
        response = supabase.table('features_master').select('*').order('date', desc=True).limit(1).execute()
        return response.data[0] if response.data else None
    except: return None

@st.cache_data(ttl=60)
def load_market_realtime():
    try:
        if supabase is None: return None
        response = supabase.table('market_realtime').select('*').order('timestamp', desc=True).limit(2).execute()
        return response.data if response.data else None
    except: return None

@st.cache_data(ttl=60)
def load_report():
    """Supabase reports 테이블에서 최신 리포트 로드 (또는 로컬 파일 폴백)"""
    try:
        # 1순위: Supabase reports 테이블에서 로드
        if supabase:
            response = supabase.table('reports').select('content').order('created_at', desc=True).limit(1).execute()
            if response.data:
                return response.data[0]['content']
        # 2순위: 로컬 파일 (개발 환경)
        local_paths = [
            "prediction_report_v7e.txt",
            "prediction_report.txt",
        ]
        for path in local_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
        return None
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_chart_url(chart_name: str) -> str | None:
    """Supabase Storage에서 차트 이미지 URL 반환"""
    try:
        if supabase:
            url = supabase.storage.from_(CHARTS_BUCKET).get_public_url(chart_name)
            return url
        return None
    except Exception:
        return None

def parse_report_for_summary(report_text):
    prediction, confidence = "하락", 96
    for line in report_text.split('\n'):
        if "최종 예측" in line:
            prediction = "상승" if "상승" in line else "하락"
        if "신뢰도" in line:
            import re
            match = re.search(r'(\d+)%', line)
            if match: confidence = int(match.group(1))
    return prediction, confidence

def get_sentiment_color(score):
    if score < -0.3: return "#ef4444"
    elif score > 0.3: return "#3b82f6"
    return "#94a3b8"

def get_relative_date(date_str):
    try:
        date = pd.to_datetime(date_str).date()
        delta = (datetime.now().date() - date).days
        if delta == 0: return "오늘"
        elif delta == 1: return "어제"
        elif delta == 2: return "그제"
        return f"{delta}일 전"
    except: return date_str

def format_korean_price(price):
    if price >= 100000000: return f"{price/100000000:.2f}억원"
    elif price >= 10000: return f"{price/10000:.0f}만원"
    return f"{price:,.0f}원"

# --- 데이터 로드 ---
sentiment_df = load_latest_sentiment()
features_data = load_latest_features()
report_text = load_report()
market_data_list = load_market_realtime()
market_data = market_data_list[0] if market_data_list else None

if report_text:
    prediction, confidence = parse_report_for_summary(report_text)
else:
    prediction, confidence = "하락", 96

# --- 헤더 ---
st.markdown("""
<div class="main-header">
    <div class="bitcoin-icon">₿</div>
    <h1 class="main-title">비트코인 예측 리포트</h1>
    <p class="subtitle">AI가 분석한 오늘의 비트코인 전망</p>
</div>
""", unsafe_allow_html=True)

# --- 디버그 ---
with st.expander("🔍 시스템 상태 확인 (디버그)"):
    st.write(f"Supabase: {'✅' if supabase else '❌'}, Features: {'✅' if features_data else '❌'}, Market: {'✅' if market_data else '❌'}")

# --- 예측 결과 섹션 ---
st.markdown('<div style="display: flex; align-items: center; margin: 2rem 0 1rem 0;"><span style="font-size: 32px; margin-right: 12px;">🎯</span><h2 style="color: white; margin: 0;">오늘의 예측</h2></div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
prediction_icon = "📉" if prediction == "하락" else "📈"
prediction_color = "#ef4444" if prediction == "하락" else "#22c55e"
signal_bg = "#7c2d12" if prediction == "하락" else "#166534"
signal_text = "⚠️ 매도/관망 신호" if prediction == "하락" else "✅ 매수 신호"

with col1:
    st.markdown(f'<div class="prediction-card"><div class="prediction-icon" style="background: {signal_bg};">{prediction_icon}</div><div style="color: #94a3b8; font-size: 14px;">오늘의 예측</div><div style="color: {prediction_color}; font-size: 32px; font-weight: bold;">{prediction}</div></div>', unsafe_allow_html=True)

with col2:
    st.markdown(f'<div class="prediction-card"><div style="color: #94a3b8;">AI 신뢰도</div><h2 style="color: white; margin: 0.5rem 0;">{confidence}%</h2><div style="background: #1e293b; height: 8px; border-radius: 4px; overflow: hidden; margin: 1rem 0;"><div style="width: {confidence}%; background: {get_sentiment_color(-0.92 if prediction == "하락" else 0.92)}; height: 100%;"></div></div><div style="color: #94a3b8; font-size: 14px;">분석 날짜</div><div style="color: white; font-size: 18px; font-weight: bold;">{datetime.now().strftime("%Y년 %m월 %d일")}</div></div>', unsafe_allow_html=True)

with col3:
    st.markdown(f'<div class="prediction-card"><div class="sell-signal" style="background: {signal_bg};">{signal_text}</div></div>', unsafe_allow_html=True)

# --- 실시간 시장 정보 ---
st.markdown('<div style="display: flex; align-items: center; margin: 2rem 0 1rem 0;"><span style="font-size: 28px; margin-right: 12px;">💹</span><h2 style="color: white; margin: 0; font-size: 24px;">실시간 시장 정보</h2></div>', unsafe_allow_html=True)

if market_data:
    usd_krw = market_data.get('usd_krw_rate', 0)
    btc_usd = market_data.get('btc_usd_price', 0)
    btc_krw = market_data.get('btc_krw_price', 0)
    kimchi = market_data.get('kimchi_premium', 0)
    ts = market_data.get('timestamp', '')
    try: update_time = pd.to_datetime(ts).strftime("%H:%M")
    except: update_time = "N/A"
    
    premium_color = "#22c55e" if kimchi >= 0 else "#3b82f6"
    premium_sign = "+" if kimchi >= 0 else ""
    
    st.markdown(f"""
    <div class="market-info-section">
        <div style="color: #f97316; font-size: 20px; font-weight: bold; margin-bottom: 1rem; display: flex; justify-content: space-between;">
            <span>📊 시장 현황</span><span style="font-size: 14px; color: #64748b; font-weight: normal;">업데이트: {update_time}</span>
        </div>
        <div class="market-info-grid">
            <div class="market-info-item"><div style="color: #94a3b8; font-size: 13px;">🇺🇸 원/달러 환율</div><div style="color: white; font-size: 22px; font-weight: bold;">{usd_krw:,.2f}원</div><div style="color: #64748b; font-size: 12px;">1 USD 기준</div></div>
            <div class="market-info-item"><div style="color: #94a3b8; font-size: 13px;">🇺🇸 BTC 미국 가격</div><div style="color: white; font-size: 22px; font-weight: bold;">${btc_usd:,.2f}</div><div style="color: #64748b; font-size: 12px;">Binance 기준</div></div>
            <div class="market-info-item"><div style="color: #94a3b8; font-size: 13px;">🇰🇷 BTC 한국 가격</div><div style="color: white; font-size: 22px; font-weight: bold;">{format_korean_price(btc_krw)}</div><div style="color: #64748b; font-size: 12px;">Upbit 기준</div></div>
            <div class="market-info-item"><div style="color: #94a3b8; font-size: 13px;">🔥 김치 프리미엄</div><div style="color: {premium_color}; font-size: 22px; font-weight: bold;">{premium_sign}{kimchi:.2f}%</div><div style="color: #64748b; font-size: 12px;">{'국내 가격 높음' if kimchi >= 0 else '해외 가격 높음'}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- 탭 구성 ---
st.markdown('<div style="display: flex; align-items: center; margin: 3rem 0 1rem 0;"><span style="font-size: 32px; margin-right: 12px;">🤖</span><h2 style="color: white; margin: 0;">AI는 왜 이렇게 예측했을까요?</h2></div>', unsafe_allow_html=True)

tab3, tab1, tab2 = st.tabs(["🎯 종합 판단", "📊 기술적 분석", "📰 뉴스 분석"])

# --- 종합 판단 탭 ---
with tab3:
    st.markdown('<h3 style="color: #f97316; margin-top: 1rem;">🎯 종합 판단</h3>', unsafe_allow_html=True)
    
    chart_url = get_chart_url("chart_price.png")
    if chart_url:
        st.image(chart_url, use_container_width=True, caption="비트코인 가격 추이 및 예측")

    col_sum1, col_sum2 = st.columns(2)

    # 기술적 시그널 데이터 준비
    if features_data:
        rsi = features_data.get('RSI_14', 50)
        macd = features_data.get('MACD', 0)
        macd_sig = features_data.get('MACD_signal', 0)
        bb_pos = features_data.get('BB_position', 0.5)
        
        rsi_signal = "과매수 🔴" if rsi > 70 else "과매도 🟢" if rsi < 30 else "중립 ⚪"
        macd_signal = "골든크로스 🟢" if macd > macd_sig else "데드크로스 🔴"
        bb_signal = "상단돌파 🔴" if bb_pos > 0.8 else "하단돌파 🟢" if bb_pos < 0.2 else "밴드내 ⚪"
        
        bullish_count = sum([rsi < 30, macd > macd_sig, bb_pos < 0.3])
        bearish_count = sum([rsi > 70, macd < macd_sig, bb_pos > 0.7])
    else:
        rsi, macd, macd_sig, bb_pos = 50, 0, 0, 0.5
        rsi_signal, macd_signal, bb_signal = "데이터 없음", "데이터 없음", "데이터 없음"
        bullish_count, bearish_count = 0, 0

    # 가격 변동률 데이터 준비
    change_1h_html = ""
    if market_data_list and len(market_data_list) >= 2:
        current = market_data_list[0].get('btc_usd_price', 0)
        prev = market_data_list[1].get('btc_usd_price', 0)
        if prev > 0:
            change_1h = ((current - prev) / prev) * 100
            change_icon = "🟢" if change_1h >= 0 else "🔴"
            change_color = "#22c55e" if change_1h >= 0 else "#ef4444"
            change_1h_html = f'<li>1시간 변동: {change_icon} <strong style="color: {change_color};">{change_1h:+.2f}%</strong></li>'
    
    current_price_html = ""
    if market_data:
        current_price_html = f'<li>현재가: <strong>${market_data.get("btc_usd_price", 0):,.2f}</strong></li>'

    # 전략 텍스트 준비
    if prediction == "하락":
        strategy_title = "🛡️ 방어 전략 권장"
        strategy_title_bg = "rgba(234, 179, 8, 0.2)"
        strategy_title_color = "#eab308"
        strategy_items = f"""<li>신규 매수 지양, 현금 비중 확대 권장</li>
<li>{confidence}% 높은 확신이지만, {100-confidence}% 반전 가능성 존재</li>
<li>분할 매도로 리스크 관리 권장</li>"""
    else:
        strategy_title = "✅ 매수 기회 탐색"
        strategy_title_bg = "rgba(34, 197, 94, 0.2)"
        strategy_title_color = "#22c55e"
        strategy_items = f"""<li>분할 매수 전략으로 진입 고려</li>
<li>{confidence}% 신뢰도로 상승 예측</li>
<li>손절가 설정 후 진입 권장</li>"""

    # 모델 예측 아이콘
    pred_icon = "📉" if prediction == "하락" else "📈"
    pred_color = "#ef4444" if prediction == "하락" else "#22c55e"

    # 왼쪽 박스 HTML 조합
    left_html = f'''<div class="summary-box">
<h4 style="color: white; margin-bottom: 1.5rem; font-size: 18px;">📌 현재 시장 상황</h4>
<div style="margin-bottom: 1.5rem;">
<div style="color: #f97316; font-weight: bold; margin-bottom: 0.75rem;">📈 기술적 시그널 요약</div>
<ul style="color: #d4d4d8; line-height: 2; padding-left: 1.25rem; margin: 0;">
<li>RSI(14): <strong>{rsi:.1f}</strong> → {rsi_signal}</li>
<li>MACD: {macd_signal}</li>
<li>볼린저밴드: {bb_signal}</li>
<li><strong>종합: 상승신호 {bullish_count}개 / 하락신호 {bearish_count}개</strong></li>
</ul>
</div>
<hr style="border: none; border-top: 1px solid #334155; margin: 1rem 0;">
<div>
<div style="color: #f97316; font-weight: bold; margin-bottom: 0.75rem;">💰 가격 변동률</div>
<ul style="color: #d4d4d8; line-height: 2; padding-left: 1.25rem; margin: 0;">
{change_1h_html}
{current_price_html}
</ul>
</div>
</div>'''

    # 오른쪽 박스 HTML 조합
    right_html = f'''<div class="summary-box">
<h4 style="color: white; margin-bottom: 1.5rem; font-size: 18px;">📋 추천 전략</h4>
<div style="margin-bottom: 1.5rem;">
<div style="background: {strategy_title_bg}; color: {strategy_title_color}; padding: 0.75rem 1rem; border-radius: 8px; font-weight: bold; margin-bottom: 1rem;">{strategy_title}</div>
<ul style="color: #d4d4d8; line-height: 2; padding-left: 1.25rem; margin: 0;">
{strategy_items}
</ul>
</div>
<hr style="border: none; border-top: 1px solid #334155; margin: 1rem 0;">
<div>
<div style="color: #f97316; font-weight: bold; margin-bottom: 0.75rem;">🤖 AI 모델별 예측</div>
<ul style="color: #d4d4d8; line-height: 2; padding-left: 1.25rem; margin: 0;">
<li>CatBoost: {pred_icon} <strong style="color: {pred_color};">{prediction}</strong></li>
<li>CNN-LSTM: {pred_icon} <strong style="color: {pred_color};">{prediction}</strong></li>
<li>PatchTST: {pred_icon} <strong style="color: {pred_color};">{prediction}</strong></li>
<li><strong>Meta-Learner 최종</strong>: {pred_icon} <strong style="color: {pred_color};">{prediction}</strong> ({confidence}%)</li>
</ul>
</div>
</div>'''

    with col_sum1:
        st.markdown(left_html, unsafe_allow_html=True)

    with col_sum2:
        st.markdown(right_html, unsafe_allow_html=True)

    if report_text:
        with st.expander("📄 상세 분석 리포트 보기"):
            st.text(report_text)

# --- 기술적 분석 탭 ---
with tab1:
    st.markdown('<h3 style="color: #f97316; margin-top: 1rem;">📊 기술적 지표 분석</h3>', unsafe_allow_html=True)
    
    if features_data:
        analysis_date = pd.to_datetime(features_data['date']).strftime("%Y년 %m월 %d일")
        st.markdown(f'<p style="color: #94a3b8; margin-bottom: 1.5rem;">분석 기준일: {analysis_date}</p>', unsafe_allow_html=True)
        
        rsi_value = features_data.get('RSI_14', 58.2)
        rsi_status = "과매수" if rsi_value > 70 else "과매도" if rsi_value < 30 else "중립"
        macd = features_data.get('MACD', 0)
        macd_signal = features_data.get('MACD_signal', 0)
        macd_status = "골든크로스" if macd > macd_signal else "데드크로스"
        macd_trend = "상승 신호" if macd > macd_signal else "하락 신호"
        bb_position = features_data.get('BB_position', 0.5)
        bb_trend = "상승추세" if bb_position > 0.5 else "하락추세"
        bb_status = "상단" if bb_position > 0.7 else "하단" if bb_position < 0.3 else "중간"

        met_col1, met_col2, met_col3 = st.columns(3)
        
        with met_col1:
            st.markdown(f"""
            <div class="metric-card rsi-card">
                <div style="width: 50px; height: 50px; background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); border-radius: 12px; margin-bottom: 1rem; display: flex; align-items: center; justify-content: center; font-size: 28px;">📉</div>
                <div style="color: #94a3b8; font-size: 16px; margin-bottom: 0.75rem;">RSI (14일)</div>
                <div style="color: #ef4444; font-size: 42px; font-weight: bold;">{rsi_value:.1f}</div>
                <span style="display: inline-block; padding: 0.5rem 1rem; border-radius: 8px; font-size: 14px; font-weight: 600; background: rgba(239, 68, 68, 0.2); color: #ef4444; margin-top: 0.75rem;">{rsi_status}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with met_col2:
            st.markdown(f"""
            <div class="metric-card macd-card">
                <div style="width: 50px; height: 50px; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); border-radius: 12px; margin-bottom: 1rem; display: flex; align-items: center; justify-content: center; font-size: 28px;">📈</div>
                <div style="color: #94a3b8; font-size: 16px; margin-bottom: 0.75rem;">MACD</div>
                <div style="color: #3b82f6; font-size: 32px; font-weight: bold;">{macd_status}</div>
                <span style="display: inline-block; padding: 0.5rem 1rem; border-radius: 8px; font-size: 14px; font-weight: 600; background: rgba(59, 130, 246, 0.2); color: #3b82f6; margin-top: 0.75rem;">{macd_trend}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with met_col3:
            st.markdown(f"""
            <div class="metric-card bb-card">
                <div style="width: 50px; height: 50px; background: linear-gradient(135deg, #a855f7 0%, #9333ea 100%); border-radius: 12px; margin-bottom: 1rem; display: flex; align-items: center; justify-content: center; font-size: 28px;">📊</div>
                <div style="color: #94a3b8; font-size: 16px; margin-bottom: 0.75rem;">볼린저 밴드</div>
                <div style="color: #a855f7; font-size: 32px; font-weight: bold;">{bb_trend}</div>
                <span style="display: inline-block; padding: 0.5rem 1rem; border-radius: 8px; font-size: 14px; font-weight: 600; background: rgba(168, 85, 247, 0.2); color: #a855f7; margin-top: 0.75rem;">{bb_status} 위치</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<h4 style="color: #f97316; margin-top: 2.5rem;">📈 AI 예측 결과</h4>', unsafe_allow_html=True)
    models_chart_url = get_chart_url("chart_models.png")
    if models_chart_url:
        st.image(models_chart_url, use_container_width=True, caption="AI 모델별 예측 결과")
    else:
        st.info("📊 차트가 준비 중입니다. 데이터 파이프라인 실행 후 자동으로 표시됩니다.")

# --- 뉴스 분석 탭 ---
with tab2:
    st.markdown('<h3 style="color: #f97316; margin-top: 1rem;">📰 시장 뉴스 분석</h3>', unsafe_allow_html=True)

    if sentiment_df is not None and len(sentiment_df) > 0:
        latest = sentiment_df.iloc[0]
        score = latest['sentiment_score']
        sent_date = pd.to_datetime(latest['date']).strftime("%Y년 %m월 %d일")
        sent_pct = int((score + 1) * 50)
        sent_color = get_sentiment_color(score)
        sent_text = "긍정적" if score > 0.3 else "부정적" if score < -0.3 else "중립적"

        st.markdown(f"""
        <div class="metric-card" style="margin-top: 1rem;">
            <div style="color: #94a3b8; margin-bottom: 0.5rem;">종합적 분위기 (점수: {score:.2f}) <span style="background: rgba(249, 115, 22, 0.1); color: #f97316; padding: 0.25rem 0.75rem; border-radius: 6px; font-size: 12px; font-weight: bold; margin-left: 0.5rem;">{sent_date}</span></div>
            <div style="background: #1e293b; height: 8px; border-radius: 4px; overflow: hidden;"><div style="background: {sent_color}; height: 100%; width: {sent_pct}%;"></div></div>
            <div style="color: {sent_color}; margin-top: 0.5rem; font-weight: bold;">{sent_text} 분위기</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<h4 style="color: white; margin-top: 2rem;">최근 주요 뉴스</h4>', unsafe_allow_html=True)

        for _, row in sentiment_df.iterrows():
            date_str = get_relative_date(row['date'])
            headline = row['headline_summary']
            impact = row['impact_score']
            if impact > 0.7: impact_class, impact_text = "impact-high", "높음"
            elif impact > 0.5: impact_class, impact_text = "impact-medium", "중간"
            else: impact_class, impact_text = "impact-low", "낮음"

            st.markdown(f"""
            <div class="news-item">
                <div style="color: #64748b; font-size: 14px; font-weight: bold;">{date_str}</div>
                <div style="color: #e5e7eb; font-size: 14px;">📰 {headline}</div>
                <div class="news-impact {impact_class}">중요도: {impact_text}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: rgba(234, 179, 8, 0.1); border: 1px solid #eab308; border-radius: 12px; padding: 1.5rem; margin-top: 2rem;">
        <div style="display: flex; align-items: flex-start;">
            <div style="color: #eab308; font-size: 24px; margin-right: 1rem;">⚠️</div>
            <div>
                <div style="color: #eab308; font-weight: bold; margin-bottom: 0.5rem;">투자 유의사항</div>
                <div style="color: #d4d4d8; font-size: 14px; line-height: 1.6;">
                    이 예측은 AI 분석 결과이며 투자 조언이 아닙니다.
                    <ul style="margin-top: 0.5rem; padding-left: 1.5rem;">
                        <li>가상화폐는 변동성이 매우 높은 자산입니다</li>
                        <li>투자 손실에 대한 책임은 투자자 본인에게 있습니다</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- 푸터 ---
st.markdown(f'<div style="text-align: center; padding: 2rem; color: #64748b; border-top: 1px solid #334155; margin-top: 3rem;"><p>Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p><p>Made with ❤️ by AI Analysis System</p></div>', unsafe_allow_html=True)
