# # 🚀 Complete BTC Data Pipeline for Colab
# 
# 이 노트북은 비트코인 가격 예측을 위한 완전한 데이터 파이프라인입니다.
# 
# **주요 기능:**
# 1. 뉴스 수집 + GPT 감성 분석 → `raw_sentiment` 테이블 저장
# 2. 가격/지표 크롤링 (BTC, FRED, Yahoo, 온체인 등)
# 3. 피쳐 엔지니어링 (기술적 지표, 감성 파생 피쳐 등)
# 4. `features_master` 테이블에 저장
# 
# **모드:**
# - `INCREMENTAL` (기본값): DB의 마지막 날짜 이후만 수집
# - `FULL`: 2017-01-01부터 전체 수집

# ## 📦 1. 환경 설정 및 라이브러리 설치

# Google Drive 마운트
import sys, os
_IS_COLAB_EARLY = 'google.colab' in sys.modules or ('google' in sys.modules and hasattr(sys.modules.get('google', None), 'colab'))
if not _IS_COLAB_EARLY:
    try:
        import importlib.util
        _IS_COLAB_EARLY = importlib.util.find_spec('google.colab') is not None and 'COLAB_RELEASE_TAG' in os.environ
    except Exception:
        pass

if _IS_COLAB_EARLY:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✅ Google Drive 마운트 완료")

# 필수 라이브러리 설치

print("✅ 라이브러리 설치 완료")

# ## 🔧 2. 필수 라이브러리 임포트 및 환경변수 로드

import pandas as pd
import numpy as np
import logging
import json
import re
import time
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from tqdm import tqdm

# 크롤링/지표 라이브러리
import ccxt
import yfinance as yf
from fredapi import Fred
import requests
from ta.trend import SMAIndicator, ADXIndicator, CCIIndicator, MACD
from ta.momentum import RSIIndicator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# 감성 분석용 라이브러리
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pygooglenews import GoogleNews

# Supabase 클라이언트
from supabase import create_client

print("✅ 라이브러리 임포트 완료")

# ==============================================================================
# 🔐 1. 어떤 환경에서든 알아서 키를 찾아오는 하이브리드 로드 구성
# ==============================================================================
import sys
import os

IS_COLAB = 'google.colab' in sys.modules
IS_KAGGLE = 'kaggle_secrets' in sys.modules or os.path.exists('/kaggle/working')
IS_GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS') == 'true'

# [로컬 & 기존 Colab 드라이브 사용자용] .env 파일 로드
try:
    from dotenv import load_dotenv
    for env_path in ['/content/drive/MyDrive/2526Winter_Sideproject/.env', '.env', '/content/.env']:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            print(f"🔧 .env 파일 로드 완료: {env_path}")
            break
except ImportError:
    pass

# [Colab Secrets / Kaggle Secrets 전용 사용자용]
if IS_COLAB:
    from google.colab import drive, userdata
    drive.mount('/content/drive')
    os.environ['PROJECT_ROOT'] = '/content/drive/MyDrive/2526Winter_Sideproject'
    for key in ['SUPABASE_URL', 'SUPABASE_KEY', 'SUPABASE_SERVICE_KEY', 'OPENAI_API_KEY', 'FRED_API_KEY']:
        try:
            val = userdata.get(key)
            if val: os.environ[key] = val
        except: pass
elif IS_KAGGLE:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    os.environ['PROJECT_ROOT'] = '/kaggle/working'
    for key in ['SUPABASE_URL', 'SUPABASE_KEY', 'SUPABASE_SERVICE_KEY', 'OPENAI_API_KEY', 'FRED_API_KEY']:
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
FRED_API_KEY = os.getenv("FRED_API_KEY")

if not SUPABASE_URL or not SERVICE_KEY:
    raise ValueError("❌ 치명적 오류: SUPABASE_URL 또는 SUPABASE_SERVICE_KEY를 찾을 수 없습니다!")

from supabase import create_client
supabase = create_client(SUPABASE_URL, SERVICE_KEY)

print("✅ Supabase 마스터 권한(Service Role) 연결 성공 🔓")

import logging
# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("CompletePipeline")
print("✅ 환경 설정 완료")

# ## 🎯 3. 감성 분석 관련 함수
# 
# Google News에서 비트코인 뉴스를 수집하고, GPT-4o-mini를 사용하여 감성 분석을 수행합니다.

def init_sentiment_analyzer():
    """감성 분석기 초기화"""
    if not OPENAI_API_KEY:
        logger.warning("감성 분석 비활성화: OPENAI_API_KEY 없음")
        return None, None
    
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
        gn = GoogleNews(lang='en', country='US')
        
        sentiment_prompt = PromptTemplate(
            input_variables=["headlines_text"],
            template="""
            You are a professional crypto market analyst.
            Below are the top 5 Bitcoin news headlines for a specific day.

            HEADLINES:
            {headlines_text}

            INSTRUCTIONS:
            1. Analyze all headlines collectively to determine the overall market sentiment.
            2. Return ONLY a JSON object with no markdown formatting.
            3. Keys required:
               - "sentiment": float between -1.0 (Negative) to 1.0 (Positive)
               - "impact": float between 0.0 (Low) to 1.0 (High)

            JSON OUTPUT EXAMPLE:
            {{"sentiment": 0.45, "impact": 0.8}}
            """
        )
        sentiment_chain = sentiment_prompt | llm
        logger.info("감성 분석기 초기화 완료 (GPT-4o-mini)")
        return sentiment_chain, gn
    except Exception as e:
        logger.error(f"감성 분석기 초기화 실패: {e}")
        return None, None


def parse_llm_response(response_content):
    """LLM 응답 파싱"""
    content = response_content.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(content)
        return float(data.get('sentiment', 0.0)), float(data.get('impact', 0.0)), None
    except json.JSONDecodeError:
        try:
            match = re.search(r'\{.*"sentiment".*\}', content, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return float(data.get('sentiment', 0.0)), float(data.get('impact', 0.0)), None
        except:
            pass
        return 0.0, 0.0, f"Parse Error: {content[:50]}..."


def fetch_and_analyze_daily(target_date, sentiment_chain, gn):
    """특정 날짜의 뉴스를 수집하고 GPT로 분석"""
    next_day = (datetime.strptime(target_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    # 1. 구글 뉴스 검색
    try:
        search = gn.search(f'Bitcoin OR BTC after:{target_date} before:{next_day}')
        headlines = [entry.title for entry in search['entries'][:5]]
    except Exception as e:
        return None, f"Search Fail: {e}"

    if not headlines:
        return {
            "date": target_date,
            "sentiment_score": 0.0,
            "impact_score": 0.0,
            "headline_summary": "No News",
            "error_log": None
        }, None

    # 2. OpenAI 분석
    headlines_text = "\n".join([f"{i+1}. {h}" for i, h in enumerate(headlines)])
    try:
        res = sentiment_chain.invoke({"headlines_text": headlines_text})
        sentiment, impact, parse_err = parse_llm_response(res.content)

        result = {
            "date": target_date,
            "sentiment_score": sentiment,
            "impact_score": impact,
            "headline_summary": headlines[0],
            "error_log": parse_err
        }
        return result, None

    except Exception as e:
        return {
            "date": target_date,
            "sentiment_score": 0.0,
            "impact_score": 0.0,
            "headline_summary": headlines[0] if headlines else "",
            "error_log": str(e)
        }, f"Analysis Fail: {e}"


def collect_and_save_sentiment(start_date, end_date):
    """
    [Step 1A] 뉴스 수집 + GPT 감성 분석 → raw_sentiment 저장
    """
    logger.info("=" * 80)
    logger.info("Step 1A: 뉴스 수집 및 GPT 감성 분석")
    logger.info("=" * 80)
    
    sentiment_chain, gn = init_sentiment_analyzer()
    if sentiment_chain is None:
        logger.warning("감성 분석 스킵 (비활성화됨)")
        return
    
    # raw_sentiment에서 마지막 날짜 확인
    try:
        response = supabase.table("raw_sentiment") \
            .select("date").order("date", desc=True).limit(1).execute()
        
        if response.data:
            last_date_str = response.data[0]['date']
            sentiment_start = datetime.strptime(last_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
            logger.info(f"raw_sentiment 마지막 날짜: {last_date_str}")
        else:
            sentiment_start = start_date
    except Exception as e:
        logger.warning(f"raw_sentiment 조회 실패: {e}")
        sentiment_start = start_date
    
    # 어제까지만 수집 (오늘 뉴스는 아직 완전하지 않음)
    sentiment_end = min(end_date, datetime.now(timezone.utc) - timedelta(days=1))
    
    total_days = (sentiment_end - sentiment_start).days + 1
    if total_days <= 0:
        logger.info("감성 데이터가 이미 최신입니다.")
        return
    
    logger.info(f"감성 분석 대상: {total_days}일 ({sentiment_start.strftime('%Y-%m-%d')} ~ {sentiment_end.strftime('%Y-%m-%d')})")
    
    curr = sentiment_start
    records_buffer = []
    BATCH_SIZE = 10
    
    pbar = tqdm(total=total_days, desc="뉴스 감성 분석")
    
    while curr <= sentiment_end:
        target_date_str = curr.strftime("%Y-%m-%d")
        
        result, err = fetch_and_analyze_daily(target_date_str, sentiment_chain, gn)
        
        if err:
            logger.warning(f"{target_date_str} 에러: {err}")
        
        if result:
            records_buffer.append(result)
        
        # 배치 저장
        if len(records_buffer) >= BATCH_SIZE:
            try:
                supabase.table("raw_sentiment").upsert(records_buffer, on_conflict="date").execute()
                logger.info(f"{len(records_buffer)}개 감성 데이터 저장 (~{target_date_str})")
                records_buffer = []
            except Exception as e:
                logger.error(f"raw_sentiment 저장 실패: {e}")
        
        curr += timedelta(days=1)
        pbar.update(1)
        time.sleep(0.1)
    
    # 남은 버퍼 저장 (루프 종료 후 잔여 데이터)
    if records_buffer:
        try:
            supabase.table("raw_sentiment").upsert(records_buffer, on_conflict="date").execute()
            logger.info(f"{len(records_buffer)}개 감성 데이터 저장 (잔여)")
        except Exception as e:
            logger.error(f"raw_sentiment 잔여 저장 실패: {e}")
    
    pbar.close()
    logger.info("뉴스 감성 분석 완료!")

print("✅ 감성 분석 함수 정의 완료")

# ## 🔧 4. 유틸리티 함수
# 
# DB 조회 및 데이터 로드 관련 헬퍼 함수들입니다.

def get_latest_date_from_db():
    """features_master에서 최신 날짜 조회"""
    try:
        response = supabase.table("features_master") \
            .select("date").order("date", desc=True).limit(1).execute()
        if not response.data:
            return None
        return pd.to_datetime(response.data[0]['date']).replace(tzinfo=timezone.utc)
    except Exception as e:
        logger.error(f"DB 날짜 조회 실패: {e}")
        return None


def fetch_all_raw_sentiment():
    """raw_sentiment 전체 데이터 로드"""
    logger.info("raw_sentiment 전체 데이터 로드 중...")
    all_data = []
    page_size = 1000
    offset = 0
    
    while True:
        response = supabase.table("raw_sentiment") \
            .select("date, sentiment_score, impact_score") \
            .order("date") \
            .range(offset, offset + page_size - 1) \
            .execute()
        
        if not response.data:
            break
        
        all_data.extend(response.data)
        offset += page_size
        
        if len(response.data) < page_size:
            break
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    logger.info(f"  raw_sentiment: {len(df)}개 로드")
    return df

print("✅ 유틸리티 함수 정의 완료")

# ## 📊 5. 데이터 크롤링 함수들
# 
# BTC 가격, FRED 경제 지표, Yahoo Finance 시장 지표, 온체인 데이터 등을 수집합니다.

def crawl_btc_price(start_dt, end_dt=None):
    """Coinbase BTC 가격 수집 (시간봉)"""
    logger.info(f"BTC 가격 수집: {start_dt.date()}")
    try:
        exchange = ccxt.coinbase()
        limit = 300
        since = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000) if end_dt else int(datetime.now(timezone.utc).timestamp() * 1000)
        all_ohlcv = []
        
        max_retries = 5
        retry_count = 0
        
        while since < end_ts:
            try:
                ohlcv = exchange.fetch_ohlcv('BTC/USD', '1h', since, limit=limit)
                if not ohlcv:
                    break
                all_ohlcv += ohlcv
                last_ts = ohlcv[-1][0]
                since = last_ts + 3600000
                retry_count = 0  # 성공 시 리셋
                if since >= end_ts:
                    break
                time.sleep(0.1)
            except Exception as e:
                retry_count += 1
                logger.warning(f"BTC API Retry ({retry_count}/{max_retries}): {e}")
                if retry_count >= max_retries:
                    logger.error("BTC API 최대 재시도 횟수 초과, 수집된 데이터로 진행")
                    break
                time.sleep(2 ** retry_count)  # exponential backoff
                continue
        
        if not all_ohlcv:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('date', inplace=True)
        df.drop(columns=['timestamp'], inplace=True)
        logger.info(f"  BTC 가격: {len(df)} rows")
        return df[~df.index.duplicated(keep='first')].sort_index()
    except Exception as e:
        logger.error(f"BTC 가격 수집 실패: {e}")
        return pd.DataFrame()


def crawl_fred_indicators(start_date):
    """FRED 경제 지표 수집"""
    logger.info(f"FRED 지표 수집: {start_date.date()}")
    fred_codes = {
        'WALCL': 'Fed_Balance_Sheet', 'WTREGEN': 'TGA', 'RRPONTSYD': 'Reverse_Repo',
        'FEDFUNDS': 'Base_Rate', 'CPIAUCSL': 'CPI', 'UNRATE': 'Unemployment',
        'M2SL': 'M2_Liquidity', 'DGS10': 'US_10Y_Yield', 'DGS2': 'US_2Y_Yield'
    }
    try:
        fred = Fred(api_key=FRED_API_KEY)
        start_str = start_date.strftime('%Y-%m-%d')
        macro_data = {}
        
        for code, name in fred_codes.items():
            try:
                macro_data[name] = fred.get_series(code, observation_start=start_str)
            except:
                pass
        
        df = pd.DataFrame(macro_data)
        if not df.empty:
            df.index = pd.to_datetime(df.index)
            df = df.resample('D').ffill().ffill().bfill()
            
            if all(c in df.columns for c in ['Fed_Balance_Sheet', 'TGA', 'Reverse_Repo']):
                df['net_liquidity'] = df['Fed_Balance_Sheet'] - df['TGA'] - df['Reverse_Repo']
                df['liquidity_lag_7'] = df['net_liquidity'].shift(7)
                df['liquidity_lag_30'] = df['net_liquidity'].shift(30)
        
        logger.info(f"  FRED: {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"FRED 수집 실패: {e}")
        return pd.DataFrame()


def crawl_market_indicators(start_date):
    """Yahoo Finance 시장 지표 수집"""
    logger.info(f"시장 지표 수집: {start_date.date()}")
    tickers = {
        '^NDX': 'NASDAQ', 'DX-Y.NYB': 'DXY', 'GC=F': 'GOLD', 
        'CL=F': 'OIL', '^VIX': 'VIX', '^TNX': 'US10Y', 'ETH-USD': 'ETH'
    }
    try:
        start_str = start_date.strftime('%Y-%m-%d')
        df = yf.download(list(tickers.keys()), start=start_str, progress=False, auto_adjust=True)['Close']
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=tickers)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        df = df.resample('D').ffill()
        logger.info(f"  시장 지표: {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"시장 지표 수집 실패: {e}")
        return pd.DataFrame()


def crawl_exchange_rate(start_date):
    """환율 데이터 수집"""
    try:
        start_str = start_date.strftime('%Y-%m-%d')
        df = yf.download('KRW=X', start=start_str, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={'Close': 'exchange_rate'})[['exchange_rate']]
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.resample('D').ffill()
        return df
    except Exception as e:
        logger.warning(f"환율 수집 실패: {e}")
        return pd.DataFrame()


def crawl_upbit_price(start_date):
    """업비트 BTC/KRW 가격 수집"""
    try:
        import pyupbit
        diff = datetime.now(timezone.utc).replace(tzinfo=None) - start_date.replace(tzinfo=None)
        count_hours = int(diff.total_seconds() / 3600) + 24
        
        df = pyupbit.get_ohlcv('KRW-BTC', interval='minute60', count=min(count_hours, 200))
        if df is not None and not df.empty:
            df = df.rename(columns={'close': 'close_upbit'})
            # timezone 안전 처리: 이미 tz-aware일 수 있음
            if df.index.tz is not None:
                df.index = df.index.tz_convert('UTC').tz_localize(None)
            else:
                df.index = df.index.tz_localize('Asia/Seoul').tz_convert('UTC').tz_localize(None)
            return df[['close_upbit']]
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"업비트 수집 실패: {e}")
        return pd.DataFrame()


def crawl_blockchain_data(chart_name):
    """Blockchain.info 온체인 데이터 수집"""
    try:
        url = f"https://api.blockchain.info/charts/{chart_name}?timespan=all&format=json&sampled=true"
        resp = requests.get(url, timeout=30).json()
        df = pd.DataFrame(resp['values'])
        df['x'] = pd.to_datetime(df['x'], unit='s')
        df.set_index('x', inplace=True)
        col_name = chart_name.replace('-', '_')
        df = df.rename(columns={'y': col_name})
        return df.resample('h').ffill()
    except Exception as e:
        logger.warning(f"온체인 데이터 수집 실패 ({chart_name}): {e}")
        return pd.DataFrame()


def crawl_onchain_data():
    """온체인 지표 수집"""
    logger.info("온체인 데이터 수집 중...")
    df_hash = crawl_blockchain_data('hash-rate')
    df_addr = crawl_blockchain_data('n-unique-addresses')
    return df_hash, df_addr


def crawl_fear_greed_index():
    """Fear & Greed Index 수집"""
    logger.info("Fear & Greed Index 수집 중...")
    try:
        url = "https://api.alternative.me/fng/?limit=0"
        resp = requests.get(url, timeout=30).json()
        fng = pd.DataFrame(resp['data'])
        fng['date'] = pd.to_datetime(fng['timestamp'].astype(int), unit='s')
        fng['fng_value'] = fng['value'].astype(int)
        fng = fng.set_index('date')[['fng_value']]
        fng = fng.resample('h').ffill()
        logger.info(f"  F&G Index: {len(fng)} rows")
        return fng
    except Exception as e:
        logger.error(f"F&G Index 수집 실패: {e}")
        return pd.DataFrame()


def collect_all_data(start_date, end_date):
    """
    [Step 1B] 모든 데이터 소스에서 수집하여 병합
    """
    logger.info("=" * 80)
    logger.info("Step 1B: 가격/지표 데이터 수집")
    logger.info("=" * 80)
    
    # BTC 가격 (기준 데이터프레임)
    df_btc = crawl_btc_price(start_date, end_date)
    if df_btc.empty:
        logger.error("BTC 가격 데이터가 없습니다!")
        return pd.DataFrame()
    
    # 다른 데이터 수집
    df_fred = crawl_fred_indicators(start_date)
    df_market = crawl_market_indicators(start_date)
    df_fx = crawl_exchange_rate(start_date)
    df_upbit = crawl_upbit_price(start_date)
    df_hash, df_addr = crawl_onchain_data()
    df_fng = crawl_fear_greed_index()
    
    # BTC 데이터프레임에 병합
    df = df_btc.copy()
    
    # 일별 데이터 → 시간별로 확장 (forward fill)
    for source_df, name in [(df_fred, 'FRED'), (df_market, 'Market'), (df_fx, 'FX')]:
        if not source_df.empty:
            source_df = source_df.resample('h').ffill()
            for col in source_df.columns:
                if col not in df.columns:
                    df = df.join(source_df[[col]], how='left')
    
    # 시간별 데이터 직접 병합
    for source_df, name in [(df_upbit, 'Upbit'), (df_hash, 'Hash'), (df_addr, 'Addr'), (df_fng, 'FnG')]:
        if not source_df.empty:
            for col in source_df.columns:
                if col not in df.columns:
                    df = df.join(source_df[[col]], how='left')
    
    # Forward fill
    df = df.ffill().bfill()
    
    logger.info(f"데이터 병합 완료: {len(df)} rows, {len(df.columns)} columns")
    return df

print("✅ 크롤링 함수 정의 완료")

# ## 🔨 6. 피쳐 엔지니어링 - 기술적 지표
# 
# 이동평균, 볼린저 밴드, RSI, MACD 등 기술적 지표를 계산합니다.

def calculate_technical_indicators(df):
    """기술적 지표 계산"""
    df = df.copy()
    if 'close' not in df.columns:
        return df
    
    close = df['close']
    high = df['high'] if 'high' in df.columns else close
    low = df['low'] if 'low' in df.columns else close
    volume = df['volume'] if 'volume' in df.columns else pd.Series(0, index=df.index)
    
    # 이동평균
    df['SMA_20'] = SMAIndicator(close, window=20).sma_indicator()
    df['SMA_60'] = SMAIndicator(close, window=60).sma_indicator()
    df['golden_cross'] = (df['SMA_20'] > df['SMA_60']).astype(int)
    df['Dist_SMA_60'] = (close - df['SMA_60']) / df['SMA_60']
    
    # 볼린저 밴드
    bb = BollingerBands(close, window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    df['BB_width'] = bb.bollinger_wband()
    df['BB_position'] = bb.bollinger_pband()
    
    # 모멘텀 지표
    df['RSI'] = RSIIndicator(close, window=14).rsi()
    df['RSI_14'] = df['RSI']
    df['rsi_14'] = df['RSI']
    macd = MACD(close)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist'] = macd.macd_diff()
    df['CCI'] = CCIIndicator(high, low, close, window=20).cci()
    df['WillR'] = WilliamsRIndicator(high, low, close, lbp=14).williams_r()
    df['log_ret_1'] = np.log(close / close.shift(1))
    
    # 변동성 지표
    df['ATR'] = AverageTrueRange(high, low, close, window=14).average_true_range()
    df['ATR_ratio'] = df['ATR'] / close
    df['volatility_30'] = close.pct_change().rolling(window=30*24).std()
    hv = close.pct_change().rolling(window=24*30).std()
    df['HV_Rank'] = hv.rolling(window=24*365, min_periods=24*30).rank(pct=True)
    
    # 추세 지표
    df['ADX'] = ADXIndicator(high, low, close, window=14).adx()
    df['Trend_Strength'] = df['ADX'] * np.where(df['MACD'] > 0, 1, -1)
    
    # 거래량 이상 탐지
    vol_ma = volume.rolling(24*7).mean()
    df['Vol_Anomaly'] = (volume > vol_ma * 2).astype(int)
    
    # 복합 지표
    df['rsi_x_macd'] = df['RSI'] * df['MACD']
    df['rsi_x_bb'] = df['RSI'] * df['BB_position']
    df['price_x_vol'] = close.pct_change() * volume.pct_change()
    
    return df

print("✅ 기술적 지표 함수 정의 완료")

# ## 🔨 7. 피쳐 엔지니어링 - 온체인 프록시
# 
# MVRV Z-Score, Puell Multiple, NUPL, SOPR 등 온체인 지표를 계산합니다.

def calculate_onchain_proxies(df):
    """온체인 프록시 지표 계산"""
    df = df.copy()
    if 'close' not in df.columns:
        return df
    
    close = df['close']
    rv_proxy = close.rolling(window=4800, min_periods=1).mean()
    rv_std = close.rolling(window=4800, min_periods=1).std()
    df['MVRV_Z_Score'] = ((close - rv_proxy) / rv_std).fillna(0)
    ma_365 = close.rolling(window=8760, min_periods=1).mean()
    df['Puell_Multiple'] = (close / ma_365).fillna(1)
    df['NUPL'] = ((close - rv_proxy) / close).fillna(0)
    df['SOPR'] = (close / close.shift(24)).fillna(1)
    
    try:
        rsi_long = RSIIndicator(close=close, window=2000).rsi()
        df['Reserve_Risk'] = (close / (rv_proxy * (100 - rsi_long + 1) + 1e-9)).fillna(0)
    except:
        df['Reserve_Risk'] = 0
    
    if 'hash_rate' in df.columns and 'close' in df.columns:
        hash_norm = df['hash_rate'].pct_change(24*7)
        price_norm = df['close'].pct_change(24*7)
        df['Hash_Price_Div'] = hash_norm - price_norm
    else:
        df['Hash_Price_Div'] = 0
    
    # active_addresses 매핑 (n_unique_addresses와 동일)
    if 'n_unique_addresses' in df.columns:
        df['active_addresses'] = df['n_unique_addresses']
    else:
        df['active_addresses'] = 0
    
    return df

print("✅ 온체인 프록시 함수 정의 완료")

# ## 🔨 8. 피쳐 엔지니어링 - 사이클 및 시간 피쳐
# 
# 반감기 사이클, 시간 정보, 거래 세션 등의 피쳐를 계산합니다.

def calculate_cycle_features(df):
    """반감기 사이클 및 시간 피쳐 계산"""
    df = df.copy()
    halvings = [
        pd.to_datetime('2012-11-28'), pd.to_datetime('2016-07-09'),
        pd.to_datetime('2020-05-11'), pd.to_datetime('2024-04-20')
    ]
    winter_start = pd.to_datetime('2022-01-01')
    
    def get_cycle_info(dt):
        past_halvings = [h for h in halvings if h <= dt]
        if past_halvings:
            last_halving = past_halvings[-1]
            days_since = (dt - last_halving).days
            cycle_idx = len(past_halvings)
        else:
            days_since = 0
            cycle_idx = 0
        cycle_progress = days_since / 1460 if days_since > 0 else 0
        days_since_winter = (dt - winter_start).days if dt >= winter_start else 0
        return days_since, cycle_progress, cycle_idx, days_since_winter
    
    results = df.index.to_series().apply(get_cycle_info)
    df['days_since_halving'] = [r[0] for r in results]
    df['cycle_progress'] = [r[1] for r in results]
    df['halving_cycle_index'] = [r[2] for r in results]
    df['days_since_winter'] = [r[3] for r in results]
    
    # 시간 피쳐
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    hour = df.index.hour
    df['is_ny_open'] = ((hour >= 13) & (hour <= 14)).astype(int)
    df['is_london_close'] = ((hour >= 15) & (hour <= 16)).astype(int)
    df['session_overlap'] = ((hour >= 13) & (hour <= 16)).astype(int)
    df['is_us_market_open'] = ((hour >= 13) & (hour <= 20) & (df['is_weekend'] == 0)).astype(int)
    
    return df

print("✅ 사이클/시간 피쳐 함수 정의 완료")

# ## 🔨 9. 피쳐 엔지니어링 - 매크로 경제 파생 피쳐
# 
# DXY, 금리, VIX, 원유, 금 등 매크로 경제 지표 기반 파생 피쳐를 계산합니다.

def calculate_macro_derived_features(df):
    """매크로 경제 파생 지표 계산"""
    df = df.copy()
    
    if 'DXY' in df.columns:
        df['dxy_ma_deviation'] = (df['DXY'] - df['DXY'].rolling(24*20).mean())
        df['dxy_change_7d_avg'] = df['DXY'].pct_change(24*7).rolling(24).mean()
        df['dxy_change_1d'] = df['DXY'].pct_change(24)
    
    if 'Base_Rate' in df.columns and 'CPI' in df.columns:
        # CPI는 지수(Index)이므로 전년 대비 상승률(Inflation Rate)로 변환
        inflation_rate = df['CPI'].pct_change(365 * 24) * 100
        df['real_interest_rate'] = df['Base_Rate'] - inflation_rate
        df['Real_Yield_Impact'] = (df['real_interest_rate'] > 0).astype(int)
    
    if 'VIX' in df.columns:
        vix_ma = df['VIX'].rolling(24*20).mean()
        df['vix_spike'] = (df['VIX'] > vix_ma * 1.5).astype(int)
    
    if 'OIL' in df.columns:
        df['oil_trend'] = df['OIL'].pct_change(24*7)
    
    if 'close' in df.columns and 'GOLD' in df.columns:
        df['btc_gold_ratio'] = df['close'] / df['GOLD']
    
    if 'NASDAQ' in df.columns and 'close' in df.columns:
        df['ndx_log_return_1d'] = np.log(df['NASDAQ'] / df['NASDAQ'].shift(1))
        df['ndx_log_return_7d'] = np.log(df['NASDAQ'] / df['NASDAQ'].shift(7))
        df['ndx_vol'] = df['NASDAQ'].pct_change().rolling(24*7).std()
        df['ndx_btc_corr'] = df['close'].rolling(24*30).corr(df['NASDAQ'])
        df['BTC_NDX_Weighted'] = df['ndx_btc_corr'] * df['ndx_log_return_1d']
    
    if 'M2_Liquidity' in df.columns:
        df['m2_change'] = df['M2_Liquidity'].pct_change(24*7)
    
    if 'US10Y' in df.columns:
        df['tnx_change'] = df['US10Y'].pct_change(24)
    
    if 'ETH' in df.columns and 'close' in df.columns:
        df['eth_btc_ratio'] = df['ETH'] / df['close']
        df['eth_btc_corr'] = df['close'].rolling(24*30).corr(df['ETH'])
    
    return df

print("✅ 매크로 파생 피쳐 함수 정의 완료")

# ## 🔨 10. 피쳐 엔지니어링 - 가격/거래량 파생 피쳐
# 
# 가격 변화율, 거래량 변화율, 김치 프리미엄 등을 계산합니다.

def calculate_price_derived_features(df):
    """가격/거래량 파생 지표 계산"""
    df = df.copy()
    
    if 'close' in df.columns:
        df['price_change'] = df['close'].pct_change()
        df['return_lag_24'] = df['close'].pct_change(24)
        df['return_lag_48'] = df['close'].pct_change(48)
    
    if 'volume' in df.columns:
        df['vol_change'] = df['volume'].pct_change()
        df['vol_lag_24'] = df['volume'].pct_change(24)
        df['vol_lag_48'] = df['volume'].pct_change(48)
    
    if 'n_unique_addresses' in df.columns and 'close' in df.columns:
        price_trend = df['close'].pct_change(24*7)
        addr_trend = df['n_unique_addresses'].pct_change(24*7)
        df['Price_Addr_Div'] = addr_trend - price_trend
    
    return df


def calculate_kimchi_premium(df):
    """김치 프리미엄 계산"""
    df = df.copy()
    
    if 'close_upbit' in df.columns and 'exchange_rate' in df.columns and 'close' in df.columns:
        upbit_usd = df['close_upbit'] / df['exchange_rate']
        df['kimchi_premium'] = ((upbit_usd - df['close']) / df['close']) * 100
    else:
        df['kimchi_premium'] = 0
    
    return df

print("✅ 가격/거래량 파생 피쳐 함수 정의 완료")

# ## 🔨 11. 피쳐 엔지니어링 - 감성 데이터 병합 및 기본 피쳐
# 
# `raw_sentiment` 테이블의 데이터를 시간별 데이터에 병합하고 기본 감성 파생 피쳐를 계산합니다.

def merge_sentiment_to_hourly(df, df_sentiment):
    """
    raw_sentiment 데이터를 시간별 데이터에 병합
    (일별 감성 → 해당 날짜의 모든 시간에 같은 값)
    """
    if df_sentiment.empty:
        df['sentiment_score'] = 0
        df['impact_score'] = 0
        return df
    
    # 원본 보호를 위해 복사
    df_sentiment = df_sentiment.copy()
    
    # 날짜만 추출하여 병합
    df['date_only'] = df.index.date
    df_sentiment['date_only'] = df_sentiment.index.date
    
    # 중복 제거 (같은 날짜 여러 개 있을 수 있음)
    df_sentiment_daily = df_sentiment.groupby('date_only').first().reset_index()
    
    df = df.reset_index()
    df = df.merge(df_sentiment_daily[['date_only', 'sentiment_score', 'impact_score']], 
                  on='date_only', how='left', suffixes=('', '_new'))
    
    # 병합된 값 사용 (기존 값이 없거나 0인 경우)
    if 'sentiment_score_new' in df.columns:
        df['sentiment_score'] = df['sentiment_score_new'].fillna(0)
        df['impact_score'] = df['impact_score_new'].fillna(0)
        df.drop(columns=['sentiment_score_new', 'impact_score_new'], inplace=True, errors='ignore')
    else:
        df['sentiment_score'] = df.get('sentiment_score', 0)
        df['impact_score'] = df.get('impact_score', 0)
    
    df.drop(columns=['date_only'], inplace=True)
    df.set_index('date', inplace=True)
    
    return df


def calculate_sentiment_features(df):
    """기본 감성 지표 파생 피쳐 계산"""
    df = df.copy()
    
    if 'sentiment_score' not in df.columns:
        df['sentiment_score'] = 0
        df['sentiment_missing_flag'] = 1
    else:
        df['sentiment_missing_flag'] = df['sentiment_score'].isna().astype(int)
        df['sentiment_score'] = df['sentiment_score'].fillna(0)
    
    if 'impact_score' not in df.columns:
        df['impact_score'] = 0
    
    df['sentiment_ma7'] = df['sentiment_score'].rolling(24*7, min_periods=1).mean()
    df['sent_ma_14'] = df['sentiment_score'].rolling(24*14, min_periods=1).mean()
    df['sent_diff'] = df['sentiment_score'].diff()
    df['sentiment_lag_1'] = df['sentiment_score'].shift(1)
    df['sent_volatility'] = df['sentiment_score'].rolling(24*7, min_periods=1).std()
    
    if 'net_liquidity' in df.columns:
        df['liq_x_sent'] = df['net_liquidity'] * df['sentiment_score']
    else:
        df['liq_x_sent'] = 0
    
    if 'fng_value' in df.columns:
        df['fng_fear_stress'] = (df['fng_value'] < 25).rolling(720, min_periods=1).mean()
        df['fng_greed_stress'] = (df['fng_value'] > 75).rolling(720, min_periods=1).mean()
    else:
        df['fng_value'] = 50
        df['fng_fear_stress'] = 0
        df['fng_greed_stress'] = 0
    
    if 'fng_fear_stress' in df.columns:
        df['winter_intensity'] = df['days_since_winter'] * df['fng_fear_stress'] if 'days_since_winter' in df.columns else 0
    
    return df

print("✅ 감성 병합 및 기본 피쳐 함수 정의 완료")

# ## 🔨 12. 피쳐 엔지니어링 - 고급 감성 피쳐 (Part 1)
# 
# 감성 모멘텀, 레짐, 가격 다이버전스 등 고급 감성 피쳐를 계산합니다.

def calculate_advanced_sentiment_features(df):
    """고급 감성 지표 계산"""
    df = df.copy()
    
    if 'sentiment_score' not in df.columns:
        df['sentiment_score'] = 0
    sent = df['sentiment_score'].fillna(0)
    
    # Momentum
    df['sent_acceleration'] = sent.diff().diff(1)
    df['sent_momentum_3d'] = sent.diff(72)
    df['sent_momentum_7d'] = sent.diff(168)
    df['sent_momentum_14d'] = sent.diff(336)
    df['sent_velocity'] = sent.diff().rolling(24, min_periods=1).mean()
    
    # Regime
    try:
        df['sent_quartile'] = pd.qcut(sent, q=4, labels=[0, 1, 2, 3], duplicates='drop').astype(float)
    except:
        df['sent_quartile'] = 1.0
    
    sent_q10 = sent.quantile(0.1)
    sent_q90 = sent.quantile(0.9)
    df['sent_extreme_negative'] = (sent < sent_q10).astype(int)
    df['sent_extreme_positive'] = (sent > sent_q90).astype(int)
    
    try:
        df['sent_zone'] = pd.cut(sent, bins=[-float('inf'), -0.3, 0.3, float('inf')], labels=[-1, 0, 1]).astype(float)
    except:
        df['sent_zone'] = 0
    
    df['sent_regime_change'] = (df['sent_zone'] != df['sent_zone'].shift()).rolling(168, min_periods=1).sum()
    df['sent_neutral_days'] = (df['sent_zone'] == 0).rolling(168, min_periods=1).sum()
    
    # Price Divergence
    if 'close' in df.columns and 'log_ret_1' in df.columns:
        price_std = df['log_ret_1'].rolling(168, min_periods=1).std()
        price_norm = (df['log_ret_1'] / (price_std + 1e-8)).clip(-1, 1)
        df['sent_price_divergence'] = sent - price_norm
        df['sent_price_corr_7d'] = sent.rolling(168, min_periods=24).corr(df['close'])
        df['sent_price_corr_30d'] = sent.rolling(720, min_periods=168).corr(df['close'])
        df['divergence_extreme'] = (np.abs(df['sent_price_divergence']) > 1.5).astype(int)
        df['sent_price_alignment'] = ((sent > 0) == (df['log_ret_1'] > 0)).astype(int)
    else:
        for col in ['sent_price_divergence', 'sent_price_corr_7d', 'sent_price_corr_30d', 'divergence_extreme', 'sent_price_alignment']:
            df[col] = 0
    
    # Volatility
    df['sent_volatility_3d'] = sent.rolling(72, min_periods=24).std()
    df['sent_volatility_14d'] = sent.rolling(336, min_periods=72).std()
    df['sent_volatility_30d'] = sent.rolling(720, min_periods=168).std()
    df['sent_vol_change'] = df['sent_volatility'].diff(24) if 'sent_volatility' in df.columns else 0
    df['sent_vol_ratio'] = df['sent_volatility'] / (df['sent_volatility_30d'] + 1e-8) if 'sent_volatility' in df.columns else 0
    
    # Volume Interaction
    if 'volume' in df.columns:
        vol_mean = df['volume'].rolling(168, min_periods=24).mean()
        vol_std = df['volume'].rolling(168, min_periods=24).std()
        df['volume_zscore'] = (df['volume'] - vol_mean) / (vol_std + 1e-8)
        df['sent_vol_interaction'] = sent * df['volume_zscore']
        df['sent_vol_alignment'] = (((sent > 0.5) & (df['volume_zscore'] > 1)) | ((sent < -0.5) & (df['volume_zscore'] > 1))).astype(int)
        impact = df['impact_score'].fillna(0) if 'impact_score' in df.columns else 0
        df['high_impact_volume'] = ((impact > 0.7) & (df['volume_zscore'] > 1)).astype(int) if 'impact_score' in df.columns else 0
    else:
        for col in ['volume_zscore', 'sent_vol_interaction', 'sent_vol_alignment', 'high_impact_volume']:
            df[col] = 0
    
    return df

print("✅ 고급 감성 피쳐 함수 (Part 1) 정의 완료")

# ## 🔨 13. 피쳐 엔지니어링 - 고급 감성 피쳐 (Part 2)
# 
# 감성 지속성, EMA, RSI, 볼린저 밴드, MACD 등을 계산합니다.

def calculate_advanced_sentiment_features_part2(df):
    """고급 감성 피쳐 Part 2 - Persistence, EMA, RSI, BB, MACD"""
    df = df.copy()
    sent = df['sentiment_score'].fillna(0) if 'sentiment_score' in df.columns else pd.Series(0, index=df.index)
    
    # Persistence
    pos_mask = sent > 0
    pos_groups = (pos_mask != pos_mask.shift()).cumsum()
    df['sent_positive_streak'] = pos_mask.groupby(pos_groups).cumsum()
    
    neg_mask = sent < 0
    neg_groups = (neg_mask != neg_mask.shift()).cumsum()
    df['sent_negative_streak'] = neg_mask.groupby(neg_groups).cumsum()
    
    sent_dir = np.sign(sent)
    df['sent_direction_changes'] = (sent_dir != sent_dir.shift()).astype(int).rolling(168, min_periods=24).sum()
    df['sent_stability_7d'] = 1 / (sent.rolling(168, min_periods=24).std() + 1e-8)
    df['sent_reversal_signal'] = ((df['sent_momentum_3d'] > 0) & (df['sent_momentum_3d'].shift(1) < 0)).astype(int) if 'sent_momentum_3d' in df.columns else 0
    
    # EMA
    df['sent_ema_7'] = sent.ewm(span=7*24).mean()
    df['sent_ema_14'] = sent.ewm(span=14*24).mean()
    df['sent_ema_30'] = sent.ewm(span=30*24).mean()
    
    # RSI
    delta = sent.diff()
    gain = delta.where(delta > 0, 0).rolling(14*24).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14*24).mean()
    rs = gain / (loss + 1e-9)
    df['sent_rsi'] = 100 - (100 / (1 + rs))
    df['sent_rsi_14'] = df['sent_rsi']
    
    # Bollinger Band
    sent_ma20 = sent.rolling(20*24).mean()
    sent_std20 = sent.rolling(20*24).std()
    df['sent_bb_upper'] = sent_ma20 + 2 * sent_std20
    df['sent_bb_lower'] = sent_ma20 - 2 * sent_std20
    df['sent_bb_width'] = (df['sent_bb_upper'] - df['sent_bb_lower']) / (sent_ma20 + 1e-9)
    df['sent_bb_position'] = (sent - df['sent_bb_lower']) / (df['sent_bb_upper'] - df['sent_bb_lower'] + 1e-9)
    
    # MACD
    ema12 = sent.ewm(span=12*24).mean()
    ema26 = sent.ewm(span=26*24).mean()
    df['sent_macd'] = ema12 - ema26
    df['sent_macd_signal'] = df['sent_macd'].ewm(span=9*24).mean()
    df['sent_macd_hist'] = df['sent_macd'] - df['sent_macd_signal']
    
    # Std/Skew
    df['sent_std_7'] = sent.rolling(7*24).std()
    df['sent_std_14'] = sent.rolling(14*24).std()
    df['sent_std_30'] = sent.rolling(30*24).std()
    df['sent_skew'] = sent.rolling(30*24).skew()
    df['sent_kurtosis'] = sent.rolling(30*24).kurt()
    
    # Misc
    if 'volume' in df.columns:
        df['sent_volume_corr_7'] = sent.rolling(7*24).corr(df['volume'])
    else:
        df['sent_volume_corr_7'] = 0
    
    if 'fng_value' in df.columns:
        df['sent_fng_diff'] = sent - df['fng_value'].fillna(50) / 100
    else:
        df['sent_fng_diff'] = 0
    
    df['sent_z_score'] = (sent - sent.rolling(30*24).mean()) / (sent.rolling(30*24).std() + 1e-9)
    df['sent_percentile'] = sent.rolling(365*24, min_periods=24).rank(pct=True)
    
    impact = df['impact_score'].fillna(0) if 'impact_score' in df.columns else pd.Series(0, index=df.index)
    df['impact_ma7'] = impact.rolling(7*24).mean()
    df['impact_volatility'] = impact.rolling(7*24).std()
    df['sent_impact_ratio'] = sent / (impact + 1e-9)
    
    return df

print("✅ 고급 감성 피쳐 함수 (Part 2) 정의 완료")

# ## 🔧 14. 통합 피쳐 계산 및 감성 병합
# 
# 모든 피쳐 계산 함수들을 통합 실행하고 감성 데이터를 병합합니다.

def calculate_all_features(df):
    """
    [Step 2] 모든 피쳐 계산
    """
    logger.info("=" * 80)
    logger.info("Step 2: 피쳐 엔지니어링")
    logger.info("=" * 80)
    
    df = calculate_technical_indicators(df)
    logger.info("  기술적 지표 완료")
    
    df = calculate_onchain_proxies(df)
    logger.info("  온체인 프록시 완료")
    
    df = calculate_cycle_features(df)
    logger.info("  사이클/시간 피쳐 완료")
    
    df = calculate_macro_derived_features(df)
    logger.info("  매크로 파생 피쳐 완료")
    
    df = calculate_price_derived_features(df)
    logger.info("  가격/거래량 파생 피쳐 완료")
    
    df = calculate_kimchi_premium(df)
    logger.info("  김치 프리미엄 완료")
    
    return df


def merge_and_calculate_sentiment(df):
    """
    [Step 3] 감성 데이터 병합 + 감성 파생 피쳐 계산
    """
    logger.info("=" * 80)
    logger.info("Step 3: 감성 데이터 병합 및 파생 피쳐 계산")
    logger.info("=" * 80)
    
    # raw_sentiment 로드
    df_sentiment = fetch_all_raw_sentiment()
    
    # 시간별 데이터에 병합
    df = merge_sentiment_to_hourly(df, df_sentiment)
    logger.info(f"  감성 데이터 병합 완료")
    
    # 감성 파생 피쳐 계산
    df = calculate_sentiment_features(df)
    logger.info("  기본 감성 피쳐 완료")
    
    df = calculate_advanced_sentiment_features(df)
    logger.info("  고급 감성 피쳐 (Part 1) 완료")
    
    df = calculate_advanced_sentiment_features_part2(df)
    logger.info("  고급 감성 피쳐 (Part 2) 완료")
    
    return df

print("✅ 통합 피쳐 계산 함수 정의 완료")

# ## 💾 15. 데이터 정제 및 DB 저장
# 
# NaN/Inf 처리, 중복 제거, DB 스키마 검증 후 `features_master` 테이블에 저장합니다.

def clean_data(df):
    """
    [Step 4] 데이터 정제
    """
    logger.info("=" * 80)
    logger.info("Step 4: 데이터 정제")
    logger.info("=" * 80)
    
    # NaN/Inf 처리
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    # 중복 제거
    df = df[~df.index.duplicated(keep='first')]
    
    # 정렬
    df = df.sort_index()
    
    logger.info(f"  정제 완료: {len(df)} rows, {len(df.columns)} columns")
    return df


def get_features_master_schema():
    """features_master 테이블의 컬럼 목록 조회 (테이블이 비어있어도 작동)"""
    try:
        # 방법 1: 기존 데이터에서 컬럼 추출
        response = supabase.table("features_master").select("*").limit(1).execute()
        if response.data:
            return list(response.data[0].keys())
        
        # 방법 2: 테이블이 비어있으면 RPC로 스키마 조회 시도
        try:
            rpc_response = supabase.rpc("get_table_columns", {"table_name": "features_master"}).execute()
            if rpc_response.data:
                return [col['column_name'] for col in rpc_response.data]
        except:
            pass
        
        # 방법 3: model_features.json에서 full_dataset 로드
        logger.warning("DB 스키마 조회 실패, model_features.json에서 로드")
        try:
            json_path = '/content/drive/MyDrive/2526Winter_Sideproject/model_features.json'
            with open(json_path, 'r', encoding='utf-8') as f:
                features_config = json.load(f)
            if 'full_dataset' in features_config:
                logger.info(f"  model_features.json에서 {len(features_config['full_dataset'])}개 컬럼 로드")
                return features_config['full_dataset']
        except Exception as json_err:
            logger.warning(f"JSON 로드 실패: {json_err}")
        
        # 방법 4: 최소 필수 컬럼 (최후의 수단)
        logger.warning("최소 필수 컬럼만 사용")
        return ['date', 'open', 'high', 'low', 'close', 'volume']
        
    except Exception as e:
        logger.error(f"스키마 조회 오류: {e}")
        return []


def save_to_database(df):
    """
    [Step 5] features_master에 저장
    """
    logger.info("=" * 80)
    logger.info("Step 5: DB 저장")
    logger.info("=" * 80)
    
    # 날짜 형식 변환: "2026-02-05 16:00:00"
    df['date'] = df.index.strftime('%Y-%m-%d %H:%M:%S')
    
    # DB 스키마 확인하여 유효한 컬럼만 선택
    valid_columns = get_features_master_schema()
    
    if valid_columns:
        # DB에 없는 컬럼 확인 및 로깅
        missing_in_db = [c for c in df.columns if c not in valid_columns and c != 'date']
        if missing_in_db:
            logger.warning(f"  DB 스키마에 없는 컬럼 {len(missing_in_db)}개 제외:")
            for col in missing_in_db[:10]:
                logger.warning(f"    - {col}")
            if len(missing_in_db) > 10:
                logger.warning(f"    ... 외 {len(missing_in_db) - 10}개")
        
        # 유효한 컬럼만 선택 (date 컬럼은 항상 포함)
        use_columns = [c for c in df.columns if c in valid_columns or c == 'date']
        df = df[use_columns]
        logger.info(f"  스키마 검증 완료: {len(use_columns)}개 컬럼 사용")
    else:
        logger.warning("  DB 스키마를 가져올 수 없음! 모든 컬럼 시도...")
    
    # INT4 타입 컬럼 목록 (DB 스키마와 일치해야 함)
    INT_COLUMNS = {
        'fng_value', 'golden_cross', 'Vol_Anomaly', 
        'days_since_halving', 'halving_cycle_index', 'days_since_winter',
        'hour', 'dayofweek', 'is_weekend', 
        'is_ny_open', 'is_london_close', 'session_overlap', 'is_us_market_open',
        'Real_Yield_Impact', 'vix_spike', 
        'sentiment_missing_flag', 'sent_extreme_negative', 'sent_extreme_positive',
        'divergence_extreme', 'sent_price_alignment', 'sent_vol_alignment',
        'high_impact_volume', 'sent_reversal_signal', 'label_binary'
    }
    
    # 레코드로 변환
    records = df.to_dict(orient='records')
    
    # 숫자형 데이터 정제 - INT vs FLOAT 구분
    for r in records:
        for key, value in r.items():
            if pd.isna(value):
                r[key] = 0
            elif key in INT_COLUMNS:
                # INT4 컬럼은 정수로 변환
                try:
                    r[key] = int(float(value))
                except (ValueError, TypeError):
                    r[key] = 0
            elif isinstance(value, (np.floating, np.integer)):
                r[key] = float(value)
    
    logger.info(f"  저장할 레코드: {len(records)}개")
    
    # 배치 upsert
    batch_size = 500
    error_count = 0
    success_count = 0
    last_error = None
    
    for i in tqdm(range(0, len(records), batch_size), desc="DB 저장"):
        batch = records[i:i+batch_size]
        try:
            supabase.table("features_master").upsert(batch, on_conflict="date").execute()
            success_count += len(batch)
        except Exception as e:
            error_count += 1
            last_error = str(e)
            logger.error(f"배치 {i//batch_size + 1} 저장 실패: {e}")
    
    # 최종 결과 로깅
    logger.info("=" * 40)
    logger.info(f"DB 저장 결과:")
    logger.info(f"  - 성공: {success_count}개 레코드")
    logger.info(f"  - 실패 배치: {error_count}개")
    if last_error:
        logger.error(f"  - 마지막 에러: {last_error[:200]}")
    logger.info("=" * 40)


def sync_market_realtime(df):
    """
    features_master 최신 1행을 market_realtime 형식으로 INSERT
    (INSERT만 사용 - PK/on_conflict 없이 신규 행 추가)
    """
    if df.empty:
        return
    row = df.iloc[-1]  # 최신 행
    data = {
        'timestamp': df.index[-1].strftime('%Y-%m-%d %H:%M:%S+00:00'),
        'usd_krw_rate': float(row.get('exchange_rate', 0)),
        'btc_usd_price': float(row.get('close', 0)),
        'btc_krw_price': float(row.get('close_upbit', 0)),
        'kimchi_premium': float(row.get('kimchi_premium', 0)),
    }
    try:
        supabase.table('market_realtime').insert(data).execute()
        logger.info("market_realtime 동기화 완료 (INSERT)")
    except Exception as e:
        logger.warning(f"market_realtime 동기화 실패: {e}")

print("✅ 데이터 정제 및 DB 저장 함수 정의 완료")

# ## 🚀 16. 메인 파이프라인 실행
# 
# 모든 단계를 통합하여 실행하는 메인 함수입니다.
# 
# **모드 선택:**
# - `INCREMENTAL`: DB의 마지막 날짜 이후만 수집 (기본값, 추천)
# - `FULL`: 2017-01-01부터 전체 수집 (주의: 시간이 오래 걸림)

def main(mode="INCREMENTAL"):
    """
    메인 파이프라인 실행
    
    Args:
        mode: "FULL" - 2017-01-01부터 전체 수집
              "INCREMENTAL" - DB의 마지막 날짜 이후만 수집 (기본값)
    """
    logger.info("=" * 80)
    logger.info(f"29_COMPLETE_PIPELINE 시작 - 모드: {mode}")
    logger.info("=" * 80)
    
    # INCREMENTAL 모드에서 피쳐 계산용 lookback 기간 (일 단위)
    # HV_Rank 등 최대 365일 rolling window 대응 + 여유분
    LOOKBACK_DAYS = 400
    
    try:
        # 시작/종료 날짜 결정
        if mode == "FULL":
            start_date = datetime(2017, 1, 1, tzinfo=timezone.utc)
            crawl_start_date = start_date
            save_start_date = start_date
            logger.info("FULL 모드: 2017-01-01부터 전체 수집")
        else:
            last_date = get_latest_date_from_db()
            if last_date:
                save_start_date = last_date + timedelta(hours=1)
                # 피쳐 계산을 위해 lookback 기간만큼 더 이전부터 데이터 수집
                crawl_start_date = save_start_date - timedelta(days=LOOKBACK_DAYS)
                start_date = save_start_date  # 감성 분석은 신규 날짜부터만
                logger.info(f"INCREMENTAL 모드: {last_date}부터 수집")
                logger.info(f"  피쳐 lookback: {crawl_start_date.strftime('%Y-%m-%d')}부터 크롤링")
            else:
                start_date = datetime(2017, 1, 1, tzinfo=timezone.utc)
                crawl_start_date = start_date
                save_start_date = start_date
                logger.info("DB가 비어있음. 2017-01-01부터 시작")
        
        end_date = datetime.now(timezone.utc)
        
        # 이미 최신인 경우 종료
        if save_start_date >= end_date:
            logger.info("이미 최신 상태입니다. 수집할 데이터가 없습니다.")
            return
        
        logger.info(f"수집 기간: {start_date} ~ {end_date}")
        
        # Step 1A: 뉴스 수집 + GPT 감성 분석 → raw_sentiment 저장
        collect_and_save_sentiment(start_date, end_date)
        
        # Step 1B: 가격/지표 데이터 수집 (lookback 포함)
        df = collect_all_data(crawl_start_date, end_date)
        if df.empty:
            logger.error("데이터 수집 실패!")
            return
        
        # Step 2: 피쳐 엔지니어링
        df = calculate_all_features(df)
        
        # Step 3: 감성 데이터 병합 + 감성 파생 피쳐
        df = merge_and_calculate_sentiment(df)
        
        # Step 4: 데이터 정제
        df = clean_data(df)
        
        # INCREMENTAL 모드: lookback 데이터 제외, 신규 데이터만 저장
        if mode == "INCREMENTAL" and crawl_start_date != save_start_date:
            save_start_naive = save_start_date.replace(tzinfo=None)
            original_len = len(df)
            df = df[df.index >= save_start_naive]
            logger.info(f"  INCREMENTAL 트리밍: {original_len} → {len(df)}개 (신규 데이터만 저장)")
        
        if df.empty:
            logger.info("저장할 신규 데이터가 없습니다.")
            return
        
        # Step 5: DB 저장
        save_to_database(df)
        sync_market_realtime(df)
        
        logger.info("=" * 80)
        logger.info("모든 파이프라인 완료!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"파이프라인 실행 중 오류: {e}")
        raise e

print("✅ 메인 함수 정의 완료")

# ## ▶️ 17. 파이프라인 실행
# 
# 아래 셀을 실행하여 파이프라인을 시작합니다.
# 
# **참고:**
# - 첫 실행이거나 전체 수집이 필요한 경우: `main(mode="FULL")`
# - 일상적인 업데이트: `main(mode="INCREMENTAL")` (기본값)

# 파이프라인 실행 (INCREMENTAL 모드 - 최신 데이터만 추가)
main(mode="INCREMENTAL")

# 전체 수집이 필요한 경우 아래 주석 해제:
# main(mode="FULL")
