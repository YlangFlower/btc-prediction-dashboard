# # 🔮 Daily Prediction & Fine-tuning Pipeline - v7E highAccuracy dynH
# **Version 7E - 5-Fold Ensemble + 3-Level Stacking + Regime Dynamic**
# **모델 경로: `2526Winter_Sideproject/models/production/v7E_production_highAccuracy_dynH`**
# 
# ## 🎯 주요 기능:
# 1. 📊 Supabase 데이터 로드 & 검증
# 2. 🔍 **과거 예측 검증** (01a 통합: is_correct NULL → 24h 뒤 Binance/Upbit 1분캔들로 정확 검증)
# 3. 🤖 모델 로드 (CatBoost, 5x CNN-LSTM, 5x PatchTST)
# 4. 🧠 **3-Level Stacking Meta-Learner 앙상블**
# 5. 📚 증분 학습 (Fine-tuning)
# 6. 🎯 **Regime-Based Dynamic Ensemble**
# 7. 🔮 내일 가격 예측 (UP/DOWN)
# 8. 💾 예측 결과 Supabase 저장
# 9. ♻️ 학습된 모델 덮어쓰기
# 
# ## 📋 Kaggle 학습 Output 파일:
# - `cnnlstm_f0.pth` ~ `cnnlstm_f4.pth` (5-Fold CNN-LSTM)
# - `patchtst_f0.pth` ~ `patchtst_f4.pth` (5-Fold PatchTST)
# - `scalers.pkl` (RobustScaler × 3)
# - `model_features.json` (Feature lists × 3)
# - `meta_models.pkl` (XGBoost L2 + L3)
# - `confidence_accuracy_coeffs.json`

# ## 📦 0. 패키지 설치 & 임포트

# 필요한 패키지 설치

# ==========================================
# 필수 라이브러리 임포트
# ==========================================
import os
import sys
import json
import pickle
import logging
import warnings
import gc
import math
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score

from catboost import CatBoostClassifier
from xgboost import XGBClassifier

import requests

warnings.filterwarnings('ignore')

print('✅ 모든 패키지를 성공적으로 임포트했습니다!')

# ==========================================
# 로깅 설정 (KST 시간) + Colab 출력 보장
# ==========================================
KST = timezone(timedelta(hours=9))

def log(message, important=False):
    """
    커스텀 로그 함수
    - 항상 KST 타임스탬프와 함께 print (Colab 출력 보장)
    - important=True 면 눈에 띄는 구분선 추가
    """
    kst_now = datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')
    msg = str(message)
    if important:
        print(f'\n{"*"*60}')
        print(f'[{kst_now}] ⭐ {msg}')
        print(f'{"*"*60}')
    else:
        print(f'[{kst_now}] {msg}')
    sys.stdout.flush()  # Colab 버퍼 즉시 플러시

log('✅ 로깅 시스템이 KST(한국 시간) 기준으로 초기화되었습니다.')

# ============================================================
# 💰 실시간 BTC 가격 조회 함수 (USD + KRW)
# ============================================================

def get_realtime_btc_price_usd():
    """실시간 BTC USD 가격 조회 (CoinGecko -> Binance fallback)"""
    try:
        r = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd', timeout=5)
        if r.status_code == 200:
            return float(r.json()['bitcoin']['usd'])
    except:
        pass
    try:
        r = requests.get('https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT', timeout=5)
        if r.status_code == 200:
            return float(r.json()['price'])
    except:
        pass
    return None

def get_krw_bitcoin_price():
    """실시간 BTC KRW 가격 조회 (Upbit)"""
    try:
        r = requests.get('https://api.upbit.com/v1/ticker?markets=KRW-BTC', timeout=5)
        if r.status_code == 200:
            return float(r.json()[0]['trade_price'])
    except:
        pass
    return None

# 현재 시세 표시
usd_price = get_realtime_btc_price_usd()
krw_price = get_krw_bitcoin_price()
now_kst = datetime.now(KST)
target_kst = now_kst + timedelta(hours=24)

print(f'\n{"="*60}')
print(f'💰 실시간 비트코인 시세')
print(f'{"="*60}')
if usd_price:
    print(f'   🇺🇸 USD: ${usd_price:,.2f}')
if krw_price:
    print(f'   🇰🇷 KRW: ₩{krw_price:,.0f}')
if usd_price and krw_price:
    implicit_rate = krw_price / usd_price
    print(f'   환율(암묵적): {implicit_rate:,.1f} KRW/USD')
print(f'\n   ⏰ 예측: {now_kst.strftime("%Y-%m-%d %H:%M")} → {target_kst.strftime("%Y-%m-%d %H:%M")}')

log('✅ 실시간 가격 조회 함수 정의 완료')

# ## 🛠️ 1. 환경 설정 & Supabase 초기화

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
# 🚀 2. 변수 할당 및 모델/Supabase 연결
# ==============================================================================
PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd())
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'production', 'v7E_production_highAccuracy_dynH')

log(f"💻 환경 셋업 완료")
log(f'  PROJECT_ROOT: {PROJECT_ROOT}')
log(f'  MODEL_DIR: {MODEL_DIR}')
os.makedirs(MODEL_DIR, exist_ok=True)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", os.getenv("SUPABASE_KEY")) 

if not SUPABASE_URL or not SERVICE_KEY:
    raise ValueError("❌ 치명적 오류: SUPABASE_URL 또는 SUPABASE_SERVICE_KEY를 찾을 수 없습니다!")

from supabase import create_client
supabase = create_client(SUPABASE_URL, SERVICE_KEY)

log("✅ Supabase 마스터 권한(Service Role) 연결 완료 🔓")

# ==============================================================================
# 📥 [GitHub Actions 전용] Supabase Storage에서 모델 가중치 자동 다운로드
# ==============================================================================
if IS_GITHUB_ACTIONS:
    log("🔽 GitHub Actions: Supabase Storage에서 모델 가중치 다운로드 시작...", important=True)
    STORAGE_BUCKET = "models"
    STORAGE_FOLDER = "daily_v7e_dynH"
    FILES_TO_DOWNLOAD = [
        "cnnlstm_f0.pth", "cnnlstm_f1.pth", "cnnlstm_f2.pth", "cnnlstm_f3.pth", "cnnlstm_f4.pth",
        "patchtst_f0.pth", "patchtst_f1.pth", "patchtst_f2.pth", "patchtst_f3.pth", "patchtst_f4.pth",
        "scalers.pkl", "meta_models.pkl", "model_features.json", "confidence_accuracy_coeffs.json",
    ]
    for filename in FILES_TO_DOWNLOAD:
        dest_path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(dest_path):
            log(f"  ✅ 이미 존재: {filename}")
            continue
        try:
            storage_path = f"{STORAGE_FOLDER}/{filename}"
            data = supabase.storage.from_(STORAGE_BUCKET).download(storage_path)
            with open(dest_path, "wb") as f:
                f.write(data)
            log(f"  ✅ 다운로드 완료: {filename} ({len(data):,} bytes)")
        except Exception as e:
            log(f"  ❌ 다운로드 실패: {filename} → {e}")
            raise
    log("🎉 모든 모델 가중치 다운로드 완료!", important=True)


# ## ⚙️ 2. Config & 모델 아키텍처 (v7E)

# ==========================================
# Config - dynH 학습과 동일 (monster_kaggle_v7E_dynH)
# ==========================================
class Config:
    """dynH 학습 설정과 완벽히 동일한 파라미터"""

    # ========== Window 설정 ==========
    SEQUENCE_LENGTHS = [24, 48, 72]
    PRIMARY_SEQUENCE_LENGTH = 72
    PREDICTION_HORIZON = 24

    # ========== CNN-LSTM (dynH: 3층 256→128, LSTM 3→2) ==========
    CNN_KERNEL_SIZES = [3, 5, 7]
    CNN_FILTERS = [64, 128, 128]  # [dynH] 3층 단순화
    LSTM_HIDDEN = 256
    LSTM_LAYERS = 2  # [dynH] 3→2
    LSTM_BIDIRECTIONAL = True

    # ========== Transformer / PatchTST (dynH: PATCH_LEN 48, STRIDE 24) ==========
    N_HEADS = 8
    D_MODEL = 256
    N_LAYERS = 4
    PATCH_LEN = 48  # [dynH] 24→48h: BTC 2일 사이클
    STRIDE = 24     # [dynH] PATCH_LEN/2
    DROPOUT = 0.4   # [dynH] 정규화 강화

    # ========== 학습 파라미터 ==========
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 3e-4
    MIN_LR = 1e-6
    WARMUP_EPOCHS = 5
    PATIENCE = 20
    ACCUM_STEPS = 4
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA = 0.25
    LABEL_SMOOTHING = 0.1

    # ========== v7E 전용 ==========
    USE_REVIN = True
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2
    USE_SWA = True
    SWA_START = 60

    # ========== CatBoost (on-the-fly 학습용) ==========
    CATBOOST_ITERATIONS = 3000
    CATBOOST_DEPTH = 6
    CATBOOST_LR = 0.03
    CATBOOST_L2 = 5
    CATBOOST_EARLY_STOPPING = 200

    # ========== Dynamic Threshold ==========
    REGIME_WINDOW = 168  # 7일 (168시간)

    # ========== 5-Fold ==========
    N_FOLDS = 5

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
log(f'Config: HORIZON={config.PREDICTION_HORIZON}h, SEQ_LEN={config.PRIMARY_SEQUENCE_LENGTH}, FOLDS={config.N_FOLDS}, DEVICE={config.DEVICE}')

# ==========================================
# v7E 모델 아키텍처 (Kaggle 학습과 동일)
# ==========================================

class RevIN(nn.Module):
    """Reversible Instance Normalization"""
    def __init__(self, n_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps, self.affine = eps, affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(n_features))
            self.beta = nn.Parameter(torch.zeros(n_features))
    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True)
            self.std = x.std(dim=1, keepdim=True) + self.eps
            x = (x - self.mean) / self.std
            if self.affine: x = x * self.gamma + self.beta
        elif mode == 'denorm':
            if self.affine: x = (x - self.beta) / self.gamma
            x = x * self.std + self.mean
        return x

class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    def __init__(self, gamma=2.0, alpha=0.25, smoothing=0.1):
        super().__init__()
        self.gamma, self.alpha, self.smoothing = gamma, alpha, smoothing
    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, ch, ratio=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(ch, ch // ratio), nn.ReLU(),
            nn.Linear(ch // ratio, ch), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1)

class MultiScaleCNN(nn.Module):
    """Multi-scale CNN with SE Block (v7E)"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c3 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.c5 = nn.Conv1d(in_ch, out_ch, 5, padding=2)
        self.c7 = nn.Conv1d(in_ch, out_ch, 7, padding=3)
        self.bn = nn.BatchNorm1d(out_ch * 3)
        self.se = SEBlock(out_ch * 3)
        self.drop = nn.Dropout(config.DROPOUT)
    def forward(self, x):
        out = torch.cat([self.c3(x), self.c5(x), self.c7(x)], dim=1)
        return self.drop(self.se(F.gelu(self.bn(out))))

class EnhancedCNNLSTM(nn.Module):
    """Enhanced CNN-LSTM (v7E - with RevIN, SEBlock, GELU)"""
    def __init__(self, n_feat, seq_len):
        super().__init__()
        self.revin = RevIN(n_feat) if config.USE_REVIN else None
        self.cnn1 = MultiScaleCNN(n_feat, config.CNN_FILTERS[0])
        self.cnn2 = MultiScaleCNN(config.CNN_FILTERS[0] * 3, config.CNN_FILTERS[1])
        self.cnn3 = MultiScaleCNN(config.CNN_FILTERS[1] * 3, config.CNN_FILTERS[2])
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(
            config.CNN_FILTERS[2] * 3, config.LSTM_HIDDEN, config.LSTM_LAYERS,
            batch_first=True, bidirectional=True, dropout=config.DROPOUT
        )
        self.attn = nn.MultiheadAttention(
            config.LSTM_HIDDEN * 2, config.N_HEADS,
            dropout=config.DROPOUT, batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.LSTM_HIDDEN * 2),
            nn.Linear(config.LSTM_HIDDEN * 2, config.LSTM_HIDDEN),
            nn.GELU(), nn.Dropout(config.DROPOUT),
            nn.Linear(config.LSTM_HIDDEN, config.LSTM_HIDDEN // 2),
            nn.GELU(), nn.Dropout(config.DROPOUT),
            nn.Linear(config.LSTM_HIDDEN // 2, 2)
        )

    def forward(self, x):
        if self.revin:
            x = self.revin(x, 'norm')
        x = x.permute(0, 2, 1)
        x = self.pool(self.cnn1(x))
        x = self.pool(self.cnn2(x))
        x = self.pool(self.cnn3(x))
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out, _ = self.attn(out, out, out)
        return self.classifier(out.mean(dim=1) + out[:, -1, :])

class EnhancedPatchTST(nn.Module):
    """Enhanced PatchTST (v7E - with RevIN, mean+max pooling)"""
    def __init__(self, n_feat, seq_len):
        super().__init__()
        self.revin = RevIN(n_feat) if config.USE_REVIN else None
        self.seq_len = seq_len
        self.n_patches = (seq_len - config.PATCH_LEN) // config.STRIDE + 1
        self.embed = nn.Sequential(
            nn.Linear(config.PATCH_LEN * n_feat, config.D_MODEL),
            nn.LayerNorm(config.D_MODEL), nn.GELU(), nn.Dropout(config.DROPOUT)
        )
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, config.D_MODEL))
        enc = nn.TransformerEncoderLayer(
            config.D_MODEL, config.N_HEADS, config.D_MODEL * 4,
            config.DROPOUT, activation='gelu', batch_first=True, norm_first=True
        )
        self.trans = nn.TransformerEncoder(enc, config.N_LAYERS)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.D_MODEL),
            nn.Linear(config.D_MODEL, config.D_MODEL // 2),
            nn.GELU(), nn.Dropout(config.DROPOUT),
            nn.Linear(config.D_MODEL // 2, config.D_MODEL // 4),
            nn.GELU(), nn.Dropout(config.DROPOUT),
            nn.Linear(config.D_MODEL // 4, 2)
        )

    def forward(self, x):
        if self.revin:
            x = self.revin(x, 'norm')
        B = x.shape[0]
        patches = [
            x[:, i * config.STRIDE : i * config.STRIDE + config.PATCH_LEN, :].reshape(B, -1)
            for i in range(self.n_patches)
        ]
        x = self.embed(torch.stack(patches, dim=1)) + self.pos
        x = self.trans(x)
        return self.classifier(x.mean(dim=1) + x.max(dim=1)[0])

log('✅ v7E 모델 아키텍처 정의 완료 (RevIN + SEBlock + MultiScaleCNN)')

# ## 📊 3. 데이터 로드 & 전처리

# ==========================================
# Supabase 데이터 로드 함수
# ==========================================

def fetch_all_features_master():
    """features_master 테이블에서 전체 데이터 로드 (타임아웃 방지)"""
    log('📊 Supabase features_master 데이터 로드 중...')
    all_rows, offset = [], 0
    batch_size = 500  # 타임아웃 방지를 위해 배치 크기 축소
    
    while True:
        try:
            # 타임아웃 방지: 작은 배치 + 재시도
            result = supabase.table('features_master').select('*').order('date').range(offset, offset + batch_size - 1).execute()
            if not result.data:
                break
            all_rows.extend(result.data)
            offset += len(result.data)
            
            # 진행상황 표시 (매 5000개마다)
            if offset % 5000 == 0:
                log(f'  진행: {offset:,} rows 로드됨...')
            
            if len(result.data) < batch_size:
                break
                
        except Exception as e:
            if 'timeout' in str(e).lower():
                log(f'  ⚠️ 타임아웃 발생 (offset={offset}), 배치 크기 축소 후 재시도...')
                batch_size = max(100, batch_size // 2)  # 배치 크기 절반으로
                continue
            else:
                raise
    
    df = pd.DataFrame(all_rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    log(f'  ✅ features_master: {len(df):,} rows ({df["date"].min()} ~ {df["date"].max()})')
    return df

def fetch_sentiment_data():
    """raw_sentiment 테이블에서 감성 데이터 로드"""
    log('📊 raw_sentiment 데이터 로드 중...')
    all_rows, offset = [], 0
    while True:
        result = supabase.table('raw_sentiment').select('date,sentiment_score,impact_score').order('date').range(offset, offset + 999).execute()
        if not result.data:
            break
        all_rows.extend(result.data)
        if len(result.data) < 1000:
            break
        offset += 1000
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df['date'] = pd.to_datetime(df['date'])
    log(f'  raw_sentiment: {len(df):,} rows')
    return df

def load_and_prepare_data(recent_days=None):
    """
    데이터 로드 + 감성 병합 + target 생성
    v7E 학습과 동일한 방식으로 처리
    
    Args:
        recent_days: None이면 전체, 숫자면 최근 N일만 로드 (타임아웃 방지)
    """
    if recent_days is not None:
        log(f'📊 최근 {recent_days}일 데이터만 로드 (타임아웃 방지)')
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=recent_days)).strftime('%Y-%m-%d')
        log(f'  기준일: {cutoff_date}')
        
        all_rows, offset = [], 0
        while True:
            result = supabase.table('features_master').select('*').gte('date', cutoff_date).order('date').range(offset, offset + 999).execute()
            if not result.data:
                break
            all_rows.extend(result.data)
            offset += len(result.data)
            if len(result.data) < 1000:
                break
        df = pd.DataFrame(all_rows)
        df['date'] = pd.to_datetime(df['date'])
        log(f'  ✅ features_master: {len(df):,} rows ({df["date"].min()} ~ {df["date"].max()})')
    else:
        df = fetch_all_features_master()
    df_sent = fetch_sentiment_data()

    # v7E 학습과 동일: features_master의 sentiment 제거 후 raw_sentiment에서 병합
    if not df_sent.empty:
        for col in ['sentiment_score', 'impact_score']:
            if col in df.columns:
                df = df.drop(columns=[col], errors='ignore')
        df = pd.merge(df, df_sent, on='date', how='left')

    # 감성 결측치 채우기
    if 'sentiment_score' not in df.columns:
        df['sentiment_score'] = 0
    if 'impact_score' not in df.columns:
        df['impact_score'] = 0.5
    df['sentiment_score'] = df['sentiment_score'].fillna(0)
    df['impact_score'] = df['impact_score'].fillna(0.5)

    # close 컬럼 이름 통일
    if 'close' not in df.columns and 'close_price' in df.columns:
        df['close'] = df['close_price']
    elif 'close' not in df.columns:
        close_candidates = [c for c in df.columns if 'close' in c.lower()]
        if close_candidates:
            df['close'] = pd.to_numeric(df[close_candidates[0]], errors='coerce')
            log(f'  ⚠️ close 컬럼으로 "{close_candidates[0]}" 사용')

    # target 생성: 24시간 뒤 가격 상승 여부
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['target'] = (df['close'].shift(-config.PREDICTION_HORIZON) > df['close']).astype(int)

    df = df.sort_values('date').reset_index(drop=True)
    log(f'✅ 데이터 준비 완료: {len(df):,} rows, target NaN(마지막 24h): {df["target"].isna().sum()}')
    return df


# ==========================================
# On-the-fly 파생 피처 생성 (Kaggle v7E 학습과 동일)
# ==========================================

def add_on_the_fly_features(df):
    """
    Kaggle v7E 학습 시 on-the-fly로 생성된 파생 피처 11개 추가
    - Cross-Asset Lags: NASDAQ/GOLD/DXY/VIX의 lag 및 return
    - Momentum: close 기반 4시간 모멘텀
    - Market Regime: 7일 불마켓 지표
    """
    df = df.copy()
    added = []

    # 1. Cross-Asset Lags & Returns
    asset_lag_map = {
        'NASDAQ': [12],       # NASDAQ_lag_12h
        'GOLD': [24],         # GOLD_lag_24h
        'DXY': [12],          # DXY_lag_12h
        'VIX': [12, 24],      # VIX_lag_12h, VIX_lag_24h
    }
    asset_ret_map = {
        'NASDAQ': [24],       # NASDAQ_ret_24h
        'GOLD': [24],         # GOLD_ret_24h
        'DXY': [24],          # DXY_ret_24h
        'VIX': [24],          # VIX_ret_24h
    }

    for asset, lags in asset_lag_map.items():
        if asset in df.columns:
            for lag in lags:
                feat_name = f'{asset}_lag_{lag}h'
                if feat_name not in df.columns:
                    df[feat_name] = df[asset].shift(lag)
                    added.append(feat_name)
        else:
            for lag in lags:
                feat_name = f'{asset}_lag_{lag}h'
                if feat_name not in df.columns:
                    df[feat_name] = 0
                    log(f'  ⚠️ {asset} 컬럼 없음 → {feat_name}=0')

    for asset, rets in asset_ret_map.items():
        if asset in df.columns:
            for ret in rets:
                feat_name = f'{asset}_ret_{ret}h'
                if feat_name not in df.columns:
                    df[feat_name] = df[asset].pct_change(ret)
                    added.append(feat_name)
        else:
            for ret in rets:
                feat_name = f'{asset}_ret_{ret}h'
                if feat_name not in df.columns:
                    df[feat_name] = 0
                    log(f'  ⚠️ {asset} 컬럼 없음 → {feat_name}=0')

    # 2. Momentum (4시간)
    if 'close' in df.columns and 'momentum_4h' not in df.columns:
        df['momentum_4h'] = df['close'].pct_change(4)
        added.append('momentum_4h')

    # 3. Market Regime (7일 불마켓)
    if 'close' in df.columns and 'regime_bull_7d' not in df.columns:
        df['regime_bull_7d'] = (df['close'] > df['close'].rolling(168, min_periods=1).mean()).astype(int)
        added.append('regime_bull_7d')

    # NaN 처리 (lag/rolling 초기값)
    for col in added:
        df[col] = df[col].bfill().fillna(0)

    log(f'  ✅ On-the-fly 파생 피처 {len(added)}개 생성: {added}')
    return df
def get_latest_date_from_supabase():
    """Supabase에서 가장 최근 날짜 조회"""
    try:
        result = supabase.table('features_master').select('date').order('date', desc=True).limit(1).execute()
        if result.data:
            return pd.to_datetime(result.data[0]['date'])
    except:
        pass
    return None

def get_model_metadata():
    """모델 메타데이터 로드 (마지막 학습 날짜 등)"""
    metadata_path = os.path.join(MODEL_DIR, 'model_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None

log('✅ 데이터 로드 함수 정의 완료')

# ## 🤖 4. 모델 로드 (5-Fold + Meta-Learner + CatBoost)

# ==========================================
# 모델 로드 함수
# ==========================================

def load_model_features_v7e():
    """
    model_features.json 로드
    v7E 형식: {'catboost': [...], 'cnnlstm': [...], 'patchtst': [...]}
    기존 형식: {'Boruta_CatBoost_121': [...], ...}
    """
    features_path = os.path.join(MODEL_DIR, 'model_features.json')
    if not os.path.exists(features_path):
        features_path = os.path.join(PROJECT_ROOT, 'model_features.json')
    if not os.path.exists(features_path):
        raise FileNotFoundError(f'model_features.json을 찾을 수 없습니다')

    with open(features_path, 'r') as f:
        data = json.load(f)

    # v7E 형식 우선
    if 'catboost' in data:
        features = {
            'catboost': [f for f in data['catboost'] if f != 'date'],
            'cnnlstm': [f for f in data['cnnlstm'] if f != 'date'],
            'patchtst': [f for f in data['patchtst'] if f != 'date'],
        }
    else:
        # 기존 Boruta 형식 호환
        cat_key = [k for k in data.keys() if 'CatBoost' in k or 'catboost' in k][0]
        cnn_key = [k for k in data.keys() if 'CNNLSTM' in k or 'cnnlstm' in k][0]
        patch_key = [k for k in data.keys() if 'PatchTST' in k or 'patchtst' in k][0]
        features = {
            'catboost': [f for f in data[cat_key] if f != 'date'],
            'cnnlstm': [f for f in data[cnn_key] if f != 'date'],
            'patchtst': [f for f in data[patch_key] if f != 'date'],
        }

    log(f'  피처 로드: CatBoost={len(features["catboost"])}, CNN-LSTM={len(features["cnnlstm"])}, PatchTST={len(features["patchtst"])}')
    return features

def load_scalers_v7e():
    """scalers.pkl 로드 (v7E: {'catboost': sc, 'cnnlstm': sc, 'patchtst': sc})"""
    scalers_path = os.path.join(MODEL_DIR, 'scalers.pkl')
    if not os.path.exists(scalers_path):
        raise FileNotFoundError(f'scalers.pkl을 찾을 수 없습니다: {scalers_path}')
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
    log(f'  스케일러 로드 완료')
    return scalers

def load_fold_models(model_type, n_features, seq_len):
    """5-Fold 모델 로드 (cnnlstm or patchtst)"""
    models = []
    for fold in range(config.N_FOLDS):
        filename = f'{model_type}_f{fold}.pth'
        filepath = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(filepath):
            log(f'  ⚠️ {filename} 없음 - 건너뜀')
            continue

        checkpoint = torch.load(filepath, map_location=config.DEVICE, weights_only=False)

        if model_type == 'cnnlstm':
            model = EnhancedCNNLSTM(n_features, seq_len)
        elif model_type == 'patchtst':
            model = EnhancedPatchTST(n_features, seq_len)
        else:
            raise ValueError(f'Unknown model_type: {model_type}')

        # v7E checkpoint 형식: {'model_state_dict': ...}
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        model = model.to(config.DEVICE)
        model.eval()
        models.append(model)

    log(f'  {model_type}: {len(models)}/{config.N_FOLDS} 모델 로드 완료')
    return models

def load_meta_models_v7e():
    """meta_models.pkl 로드: dynG 형식 {'blend_weights': [...], 'best_strategy': '...'}"""
    meta_path = os.path.join(MODEL_DIR, 'meta_models.pkl')
    if not os.path.exists(meta_path):
        log('  ⚠️ meta_models.pkl 없음 - 단순 평균 폴백 사용')
        return None, 'A: Simple Avg'
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    blend_weights = meta.get('blend_weights')
    best_strategy = meta.get('best_strategy', 'A: Simple Avg')
    log(f'  ✅ dynG Meta 로드: best_strategy={best_strategy}')
    if blend_weights:
        log(f'  OOF Blend Weights: Cat={blend_weights[0]:.3f}, CNN={blend_weights[1]:.3f}, Patch={blend_weights[2]:.3f}')
    return blend_weights, best_strategy

def load_confidence_coeffs():
    """confidence_accuracy_coeffs.json 로드"""
    coeff_path = os.path.join(MODEL_DIR, 'confidence_accuracy_coeffs.json')
    if not os.path.exists(coeff_path):
        log('  ⚠️ confidence_accuracy_coeffs.json 없음 - 기본값 사용')
        return {'a': 0, 'b': 1, 'c': 0}
    with open(coeff_path, 'r') as f:
        coeffs = json.load(f)
    log(f'  ✅ Confidence-Accuracy 계수: a={coeffs["a"]:.6f}, b={coeffs["b"]:.6f}, c={coeffs["c"]:.4f}')
    return coeffs

def train_catboost_from_data(df, features, scaler):
    """
    CatBoost 모델 on-the-fly 학습
    (Kaggle 학습에서 CatBoost .cbm 파일이 저장되지 않으므로,
     Supabase 데이터로 직접 학습)
    """
    log('🚀 CatBoost on-the-fly 학습 시작...', important=True)

    # target이 있는 행만 사용
    df_valid = df.dropna(subset=['target']).copy()
    if len(df_valid) < 100:
        log('  ⚠️ 학습 데이터 부족 - CatBoost 생략')
        return None

    # 피처 선택 및 스케일링 (누락 피처는 0으로 채움)
    missing_feat = [f for f in features if f not in df_valid.columns]
    if missing_feat:
        log(f"  ⚠️ CatBoost 누락 피처 {len(missing_feat)}개 → 0으로 채움")
        for f in missing_feat:
            df_valid[f] = 0

    if len([f for f in features if f in df_valid.columns]) == 0:
        log('  ❌ 유효한 CatBoost 피처 없음')
        return None

    X = df_valid[features].fillna(0).values  # 전체 피처 사용 (스케일러 호환)
    y = df_valid['target'].values.astype(int)

    # 저장된 스케일러로 변환 (일관성 유지)
    X_scaled = scaler.transform(X)

    # 시계열 기반 분할: 마지막 20%를 validation으로
    split_idx = int(len(X_scaled) * 0.8)
    X_tr, X_va = X_scaled[:split_idx], X_scaled[split_idx:]
    y_tr, y_va = y[:split_idx], y[split_idx:]

    # v7E 학습과 동일한 파라미터
    cat_model = CatBoostClassifier(
        iterations=config.CATBOOST_ITERATIONS,
        depth=config.CATBOOST_DEPTH,
        learning_rate=config.CATBOOST_LR,
        l2_leaf_reg=config.CATBOOST_L2,
        loss_function='Logloss',
        eval_metric='Accuracy',
        early_stopping_rounds=config.CATBOOST_EARLY_STOPPING,
        verbose=0,
        task_type='GPU' if torch.cuda.is_available() else 'CPU',
        random_seed=42
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)

    val_preds = cat_model.predict(X_va)
    val_acc = accuracy_score(y_va, val_preds)
    log(f'  ✅ CatBoost 학습 완료: Val Acc={val_acc:.4f} (Train={len(X_tr):,}, Val={len(X_va):,})', important=True)

    # 저장
    save_path = os.path.join(MODEL_DIR, 'catboost_model_v7e.cbm')
    cat_model.save_model(save_path)
    log(f'  💾 CatBoost 모델 저장: {save_path}')

    return cat_model

def load_all_models(df_full=None):
    """
    모든 모델 로드 (5-Fold + Meta-Learner + CatBoost + 스케일러)
    CatBoost는 파일이 없으면 on-the-fly 학습
    """
    log('', important=False)
    log('🤖 모델 로드 시작', important=True)
    models = {}

    # 1. Feature lists
    models['features'] = load_model_features_v7e()

    # 2. Scalers
    scalers = load_scalers_v7e()
    models['scaler_catboost'] = scalers['catboost']
    models['scaler_cnnlstm'] = scalers['cnnlstm']
    models['scaler_patchtst'] = scalers['patchtst']

    # 3. CNN-LSTM (5-Fold)
    n_feat_cnn = len(models['features']['cnnlstm'])
    models['cnnlstm_models'] = load_fold_models('cnnlstm', n_feat_cnn, config.PRIMARY_SEQUENCE_LENGTH)

    # 4. PatchTST (5-Fold)
    n_feat_patch = len(models['features']['patchtst'])
    models['patchtst_models'] = load_fold_models('patchtst', n_feat_patch, config.PRIMARY_SEQUENCE_LENGTH)

    # 5. Meta-Learner (dynG: blend_weights + best_strategy)
    models['blend_weights'], models['best_strategy'] = load_meta_models_v7e()

    # 6. Confidence-Accuracy 계수
    models['confidence_coeffs'] = load_confidence_coeffs()

    # 7. CatBoost (Kaggle 전체 학습 모델 + 증분 학습)
    catboost_production_path = os.path.join(MODEL_DIR, 'catboost_production.cbm')
    catboost_v7e_path = os.path.join(MODEL_DIR, 'catboost_model_v7e.cbm')
    
    if os.path.exists(catboost_production_path):
        # Kaggle에서 전체 데이터로 학습한 프로덕션 모델 로드
        cat_model = CatBoostClassifier()
        cat_model.load_model(catboost_production_path)
        models['catboost_model'] = cat_model
        log(f'  ✅ CatBoost 프로덕션 모델 로드: {os.path.basename(catboost_production_path)}')
        
        # 증분 학습 여부는 나중에 Step 6에서 결정
        models['catboost_needs_finetuning'] = True
        
    elif os.path.exists(catboost_v7e_path):
        # 기존 on-the-fly 학습 모델 (fallback)
        cat_model = CatBoostClassifier()
        cat_model.load_model(catboost_v7e_path)
        models['catboost_model'] = cat_model
        log(f'  ✅ CatBoost 기존 모델 로드: {os.path.basename(catboost_v7e_path)}')
        models['catboost_needs_finetuning'] = False
        
    elif df_full is not None and len(df_full.dropna(subset=['target'])) >= 100:
        # 저장된 모델이 없으면 on-the-fly 학습
        log('  ⚠️ Kaggle 프로덕션 모델 없음 - on-the-fly 학습 시작')
        models['catboost_model'] = train_catboost_from_data(
            df_full, models['features']['catboost'], models['scaler_catboost']
        )
        models['catboost_needs_finetuning'] = False
        
    else:
        log('  ⚠️ CatBoost 사용 불가 - 데이터/모델 없음')
        models['catboost_model'] = None
        models['catboost_needs_finetuning'] = False

    # 요약
    print(f'\n{"="*50}')
    print(f'📋 모델 로드 요약')
    print(f'{"="*50}')
    print(f'  CatBoost : {"✅ 로드됨" if models.get("catboost_model") else "❌ 없음"}')
    print(f'  CNN-LSTM : {len(models.get("cnnlstm_models", []))} folds')
    print(f'  PatchTST : {len(models.get("patchtst_models", []))} folds')
    print(f'  Strategy : {models.get("best_strategy", "단순평균")}')
    print(f'  Blend W  : {"✅" if models.get("blend_weights") else "❌"}')
    print(f'{"="*50}')

    return models

log('✅ 모델 로드 함수 정의 완료')

# ## 🔮 5. 예측 함수 (5-Fold Ensemble + Meta-Learner + Regime)

# ==========================================
# 개별 모델 예측 함수
# ==========================================

def predict_with_catboost(model, X):
    """CatBoost 단일 예측: UP 확률 반환"""
    try:
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return float(proba[0, 1])
        return float(proba[0])
    except Exception as e:
        log(f'  ⚠️ CatBoost 예측 실패: {e}')
        return 0.5

def predict_with_fold_models(models_list, X_seq):
    """
    5-Fold 모델 예측 평균
    X_seq: (seq_len, n_features) numpy array
    """
    if not models_list:
        return 0.5

    probs = []
    for model in models_list:
        try:
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_seq).unsqueeze(0).to(config.DEVICE)
                output = model(X_tensor)
                prob = torch.softmax(output, dim=1)[0, 1].item()
                probs.append(prob)
        except Exception as e:
            log(f'  ⚠️ Fold 모델 예측 실패: {e}')
            probs.append(0.5)

    return float(np.mean(probs))

# ==========================================
# Meta Feature 생성 (v7E 학습과 동일)
# ==========================================

def make_meta_features(p_cat, p_cnn, p_patch):
    """
    Meta-Learner용 피처 생성 (v7E 학습과 동일한 10개 피처)
    [p1, p2, p3, |p1-0.5|, |p2-0.5|, |p3-0.5|, |p1-p2|, |p2-p3|, |p1-p3|, vote_count]
    """
    p1, p2, p3 = np.atleast_1d(p_cat), np.atleast_1d(p_cnn), np.atleast_1d(p_patch)
    return np.column_stack([
        p1, p2, p3,
        np.abs(p1 - 0.5), np.abs(p2 - 0.5), np.abs(p3 - 0.5),
        np.abs(p1 - p2), np.abs(p2 - p3), np.abs(p1 - p3),
        (p1 > 0.5).astype(int) + (p2 > 0.5).astype(int) + (p3 > 0.5).astype(int)
    ])

# ==========================================
# Regime Detection (v7E 학습과 동일)
# ==========================================

def detect_regime(df_slice):
    """시장 국면 감지: bull / bear / high_vol / sideways"""
    if len(df_slice) < 168:
        return 'sideways'

    close_col = 'close' if 'close' in df_slice.columns else 'close_price'
    close = pd.to_numeric(df_slice[close_col], errors='coerce').values
    close = close[~np.isnan(close)]

    if len(close) < 168:
        return 'sideways'

    sma_7d = np.mean(close[-168:])
    sma_30d = np.mean(close[-720:]) if len(close) >= 720 else sma_7d

    vol_col = 'volatility_30' if 'volatility_30' in df_slice.columns else None
    vol = float(df_slice[vol_col].iloc[-1]) if vol_col and pd.notna(df_slice[vol_col].iloc[-1]) else 0.02

    if close[-1] > sma_7d * 1.02 and close[-1] > sma_30d:
        return 'bull'
    elif close[-1] < sma_7d * 0.98 or close[-1] < sma_30d * 0.95:
        return 'bear'
    elif vol > 0.04:
        return 'high_vol'
    else:
        return 'sideways'

def get_regime_weights(regime):
    """Regime별 모델 가중치 (cat, cnn, patch)"""
    if regime == 'bull':
        return (0.15, 0.25, 0.60)
    elif regime == 'bear':
        return (0.50, 0.30, 0.20)
    elif regime == 'high_vol':
        return (0.20, 0.55, 0.25)
    else:  # sideways
        return (0.33, 0.34, 0.33)

# ==========================================
# 통합 앙상블 예측
# ==========================================

def ensemble_predict_v7e(models, X_latest_cat, X_seq_cnn, X_seq_patch, df_recent):
    """
    v7E 앙상블 예측 파이프라인:
    1. 개별 모델 예측 (CatBoost + 5-Fold CNN-LSTM + 5-Fold PatchTST)
    2. 3-Level Stacking Meta-Learner
    3. Regime-Based Dynamic Ensemble
    4. 최종 예측 + 신뢰도 + 예상 정확도
    """
    log('🔮 예측 수행', important=True)

    individual_predictions = {}

    # 1. 개별 모델 예측
    if models.get('catboost_model') is not None and X_latest_cat is not None:
        prob_cat = predict_with_catboost(models['catboost_model'], X_latest_cat)
    else:
        prob_cat = 0.5
    individual_predictions['catboost'] = prob_cat

    prob_cnn = predict_with_fold_models(models.get('cnnlstm_models', []), X_seq_cnn)
    individual_predictions['cnnlstm'] = prob_cnn

    prob_patch = predict_with_fold_models(models.get('patchtst_models', []), X_seq_patch)
    individual_predictions['patchtst'] = prob_patch

    print(f'\n  📊 개별 예측 결과:')
    print(f'    CatBoost : {prob_cat:.4f} ({"UP" if prob_cat > 0.5 else "DOWN"})')
    print(f'    CNN-LSTM : {prob_cnn:.4f} ({"UP" if prob_cnn > 0.5 else "DOWN"}) [{len(models.get("cnnlstm_models",[]))} folds]')
    print(f'    PatchTST : {prob_patch:.4f} ({"UP" if prob_patch > 0.5 else "DOWN"}) [{len(models.get("patchtst_models",[]))} folds]')

    # 2. dynG 앙새블 전략 (blend_weights + best_strategy 기반)
    blend_weights = models.get('blend_weights')
    best_strategy = models.get('best_strategy', 'A: Simple Avg')

    # Strategy A: Simple Average
    prob_simple = (prob_cat + prob_cnn + prob_patch) / 3.0

    # Strategy B: Confidence-Weighted
    c1, c2, c3 = abs(prob_cat - 0.5), abs(prob_cnn - 0.5), abs(prob_patch - 0.5)
    total_c = c1 + c2 + c3 + 1e-8
    prob_conf_w = (c1/total_c)*prob_cat + (c2/total_c)*prob_cnn + (c3/total_c)*prob_patch

    # Strategy C: Consensus-Boosted
    votes_up = (prob_cat > 0.5) + (prob_cnn > 0.5) + (prob_patch > 0.5)
    prob_consensus = prob_simple if (votes_up == 3 or votes_up == 0) else prob_conf_w

    # Strategy D: OOF Blend (Ridge)
    if blend_weights and len(blend_weights) >= 3:
        prob_blend = blend_weights[0]*prob_cat + blend_weights[1]*prob_cnn + blend_weights[2]*prob_patch
    else:
        prob_blend = prob_simple

    # dynG Best Strategy 선택
    if 'Simple' in best_strategy:
        prob_stacking = prob_simple
    elif 'Confidence' in best_strategy:
        prob_stacking = prob_conf_w
    elif 'Consensus' in best_strategy:
        prob_stacking = prob_consensus
    else:  # OOF Blend (Ridge)
        prob_stacking = prob_blend

    prob_l2 = prob_stacking  # 하위 호환성 유지
    print(f'\n  🧠 dynG [{best_strategy}]: {prob_stacking:.4f}')

    # 3. Regime-Based Dynamic Ensemble
    regime = detect_regime(df_recent)
    w = get_regime_weights(regime)
    prob_regime = w[0] * prob_cat + w[1] * prob_cnn + w[2] * prob_patch
    print(f'  🌍 Regime: {regime} | Weights=(cat:{w[0]}, cnn:{w[1]}, patch:{w[2]}) | Prob={prob_regime:.4f}')

    # 4. 최종 확률: Stacking(60%) + Regime(40%)
    final_prob = 0.6 * prob_stacking + 0.4 * prob_regime
    print(f'  📐 Final = 0.6*Stacking + 0.4*Regime = {final_prob:.4f}')

    # 5. 예측 + 신뢰도
    prediction = 1 if final_prob > 0.5 else 0
    confidence = max(final_prob, 1 - final_prob)

    # 6. Confidence -> Predicted Accuracy (이차 방정식)
    coeffs = models.get('confidence_coeffs', {'a': 0, 'b': 1, 'c': 0})
    conf_pct = confidence * 100
    predicted_accuracy = coeffs['a'] * conf_pct**2 + coeffs['b'] * conf_pct + coeffs['c']
    predicted_accuracy = np.clip(predicted_accuracy, 50, 100)

    details = {
        'individual_predictions': individual_predictions,
        'individual_avg': float((prob_cat + prob_cnn + prob_patch) / 3),
        'meta_l2_probability': float(prob_l2),
        'meta_stacking_probability': float(prob_stacking),
        'regime': regime,
        'regime_weights': list(w),
        'regime_probability': float(prob_regime),
        'final_probability': float(final_prob),
        'predicted_accuracy_pct': float(predicted_accuracy),
        'used_dynG_strategy': best_strategy,
        'n_cnnlstm_folds': len(models.get('cnnlstm_models', [])),
        'n_patchtst_folds': len(models.get('patchtst_models', [])),
    }

    print(f'\n  {"="*50}')
    print(f'  🎯 예측: {"🟢 UP (상승)" if prediction == 1 else "🔴 DOWN (하락)"}')
    print(f'  📊 신뢰도: {confidence:.4f} ({confidence*100:.1f}%)')
    print(f'  📈 예상 정확도: {predicted_accuracy:.1f}%')
    print(f'  {"="*50}')

    return prediction, confidence, details

log('✅ 예측 함수 정의 완료')

# ## 📚 6. 증분 학습 & 모델 저장

# ==========================================
# 증분 학습 함수 (Fine-tuning)
# ==========================================

class SeqDataset(Dataset):
    """시퀀스 데이터셋 (v7E 학습과 동일)"""
    def __init__(self, X, y, seq_len):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.seq_len = seq_len
    def __len__(self):
        return max(0, len(self.X) - self.seq_len)
    def __getitem__(self, idx):
        return self.X[idx:idx+self.seq_len], self.y[idx+self.seq_len-1]

def incremental_train_catboost(model, X_new, y_new):
    """CatBoost 증분 학습 (init_model 기반)"""
    if len(X_new) < 10:
        log('  ⚠️ CatBoost 증분 학습 생략: 데이터 부족')
        return model

    split_idx = int(len(X_new) * 0.8)
    X_tr, X_va = X_new[:split_idx], X_new[split_idx:]
    y_tr, y_va = y_new[:split_idx], y_new[split_idx:]

    try:
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), init_model=model, verbose=False)
        val_acc = accuracy_score(y_va, model.predict(X_va))
        log(f'  ✅ CatBoost 증분 학습 완료: Val Acc={val_acc:.4f}')
    except Exception as e:
        log(f'  ⚠️ CatBoost 증분 학습 실패: {e}')
    return model

def incremental_train_deep_model(model, X_seq, y_seq, model_name, epochs=10, batch_size=32):
    """딥러닝 모델 증분 학습 (CNN-LSTM / PatchTST)"""
    if len(X_seq) < config.PRIMARY_SEQUENCE_LENGTH + 5:
        log(f'  ⚠️ {model_name} 증분 학습 생략: 시퀀스 데이터 부족')
        return model

    dataset = SeqDataset(X_seq, y_seq, config.PRIMARY_SEQUENCE_LENGTH)
    if len(dataset) < 5:
        log(f'  ⚠️ {model_name} 증분 학습 생략: 시퀀스 부족 ({len(dataset)}개)')
        return model

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = FocalLoss(config.FOCAL_GAMMA, config.FOCAL_ALPHA, config.LABEL_SMOOTHING)

    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(config.DEVICE), batch_y.to(config.DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    model.eval()
    log(f'  ✅ {model_name} 증분 학습 완료 ({epochs} epochs, {len(dataset)} sequences)')
    return model

def run_incremental_training(models, df_gap, features):
    """전체 모델 증분 학습 수행"""
    if df_gap is None or len(df_gap) < 30:
        log('⚠️ 증분 학습 생략: 갭 데이터 부족')
        return False

    log('📚 증분 학습 (Fine-tuning) 시작', important=True)
    log(f'  갭 데이터: {len(df_gap):,} rows')

    trained = False
    df_gap_valid = df_gap.dropna(subset=['target'])

    # CatBoost 증분 학습
    if models.get('catboost_model') is not None:
        # 누락 피처 0으로 채움
        for f in features['catboost']:
            if f not in df_gap_valid.columns:
                df_gap_valid[f] = 0
        if len(df_gap_valid) >= 10:
            X_new = models['scaler_catboost'].transform(df_gap_valid[features['catboost']].fillna(0).values)
            y_new = df_gap_valid['target'].values.astype(int)
            models['catboost_model'] = incremental_train_catboost(models['catboost_model'], X_new, y_new)
            trained = True

    # CNN-LSTM 5-Fold 증분 학습
    if models.get('cnnlstm_models'):
        # 누락 피처 0으로 채움
        for f in features['cnnlstm']:
            if f not in df_gap_valid.columns:
                df_gap_valid[f] = 0
        X_cnn = models['scaler_cnnlstm'].transform(df_gap_valid[features['cnnlstm']].fillna(0).values)
        y_cnn = df_gap_valid['target'].values.astype(int)
        for i, model in enumerate(models.get('cnnlstm_models', [])):
            models['cnnlstm_models'][i] = incremental_train_deep_model(model, X_cnn, y_cnn, f'CNN-LSTM_f{i}', epochs=10)
        trained = True

    # PatchTST 5-Fold 증분 학습
    if models.get('patchtst_models'):
        # 누락 피처 0으로 채움
        for f in features['patchtst']:
            if f not in df_gap_valid.columns:
                df_gap_valid[f] = 0
        X_patch = models['scaler_patchtst'].transform(df_gap_valid[features['patchtst']].fillna(0).values)
        y_patch = df_gap_valid['target'].values.astype(int)
        for i, model in enumerate(models.get('patchtst_models', [])):
            models['patchtst_models'][i] = incremental_train_deep_model(model, X_patch, y_patch, f'PatchTST_f{i}', epochs=10)
        trained = True

    if trained:
        log('✅ 증분 학습 완료', important=True)
    return trained
# ==========================================
# 모델 저장 함수
# ==========================================

def save_updated_models(models, last_data_date=None):
    """업데이트된 모델 저장"""
    log('💾 모델 저장 중...')

    # CatBoost
    if models.get('catboost_model') is not None:
        cat_path = os.path.join(MODEL_DIR, 'catboost_model_v7e.cbm')
        models['catboost_model'].save_model(cat_path)
        log(f'  CatBoost: {cat_path}')

    # CNN-LSTM (5-Fold)
    for i, model in enumerate(models.get('cnnlstm_models', [])):
        path = os.path.join(MODEL_DIR, f'cnnlstm_f{i}.pth')
        torch.save({'model_state_dict': model.state_dict()}, path)
    log(f'  CNN-LSTM: {len(models.get("cnnlstm_models", []))} folds 저장')

    # PatchTST (5-Fold)
    for i, model in enumerate(models.get('patchtst_models', [])):
        path = os.path.join(MODEL_DIR, f'patchtst_f{i}.pth')
        torch.save({'model_state_dict': model.state_dict()}, path)
    log(f'  PatchTST: {len(models.get("patchtst_models", []))} folds 저장')

    # 메타데이터
    metadata = {
        'last_updated': datetime.now(timezone.utc).isoformat(),
        'last_data_date': pd.to_datetime(last_data_date).strftime('%Y-%m-%d %H:%M:%S') if last_data_date is not None else None,
        'model_version': 'v7E',
        'n_folds': config.N_FOLDS,
        'models': ['catboost', 'cnnlstm_5fold', 'patchtst_5fold', 'meta_l2', 'meta_l3']
    }
    with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    log('✅ 모델 저장 완료')

# ==========================================
# 예측 결과 Supabase 저장
# ==========================================

def save_prediction_to_supabase(prediction_date, prediction, confidence, model_details,
                                 current_price=None, current_price_krw=None):
    """예측 결과를 Supabase predictions 테이블에 저장"""
    try:
        if current_price is None:
            current_price = get_realtime_btc_price_usd() or 0
        if current_price_krw is None:
            current_price_krw = get_krw_bitcoin_price() or 0

        data = {
            'date': prediction_date.strftime('%Y-%m-%d %H:%M:%S+00:00'),
            'predicted_price': float(current_price),
            'predicted_price_krw': float(current_price_krw) if current_price_krw else None,
            'direction': 'UP' if prediction == 1 else 'DOWN',
            'confidence_score': float(confidence),
            'model_breakdown': json.dumps(model_details, ensure_ascii=False, default=str)
        }

        response = supabase.table('predictions').upsert(data, on_conflict='date').execute()
        log(f'✅ 예측 저장 완료: {prediction_date.strftime("%Y-%m-%d %H:%M")}')
        return response
    except Exception as e:
        log(f'❌ 예측 저장 실패: {e}')
        return None

log('✅ 증분 학습 / 모델 저장 / 예측 저장 함수 정의 완료')

# ## 🚀 7. 메인 파이프라인

# ==========================================
# 어제 예측 검증 함수
# ==========================================

def validate_yesterday_prediction():
    """어제 예측 결과 검증"""
    try:
        now_utc = datetime.now(timezone.utc)
        yesterday = (now_utc - timedelta(hours=48)).strftime('%Y-%m-%d')

        result = supabase.table('predictions').select('*').gte('date', yesterday).order('date', desc=True).limit(5).execute()
        if not result.data:
            log('  어제 예측 데이터 없음')
            return

        print(f'\n  최근 예측 기록:')
        for pred in result.data:
            pred_date = pred.get('date', '')
            direction = pred.get('direction', '?')
            confidence = pred.get('confidence_score', 0)
            actual = pred.get('actual_result') or pred.get('actual_direction')
            if actual and actual == direction:
                status = '✅'
            elif actual:
                status = '❌'
            else:
                status = '⏳'
            conf_str = f'{confidence:.1%}' if confidence else 'N/A'
            print(f'    {status} {pred_date[:16]} | 예측:{direction} | 실제:{actual or "대기중"} | 신뢰도:{conf_str}')

    except Exception as e:
        log(f'  ⚠️ 검증 실패: {e}')

# ==========================================
# 과거 예측 검증 (01a_validate_past_predictions 로직)
# - is_correct가 NULL인 예측에 대해 24시간 뒤 실제 가격으로 검증
# - Binance/Upbit 1분 캔들 API로 정확한 시점 가격 조회
# ==========================================

def get_historical_usd_price(target_dt):
    """Binance Klines API를 사용하여 특정 시점의 BTC/USDT 종가를 가져옵니다."""
    try:
        timestamp_ms = int(target_dt.timestamp() * 1000)
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": "BTCUSDT", "interval": "1m", "endTime": timestamp_ms, "limit": 1}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data and len(data) > 0:
            return float(data[0][4])
        return None
    except Exception as e:
        log(f'  USD 가격 조회 실패: {e}')
        return None

def get_historical_krw_price(target_dt):
    """Upbit 캔들 API를 사용하여 특정 시점의 BTC/KRW 종가를 가져옵니다."""
    try:
        kst_dt = target_dt.astimezone(timezone(timedelta(hours=9)))
        to_str = kst_dt.strftime('%Y-%m-%dT%H:%M:%S+09:00')
        url = "https://api.upbit.com/v1/candles/minutes/1"
        params = {"market": "KRW-BTC", "to": to_str, "count": 1}
        response = requests.get(url, params=params, headers={"accept": "application/json"}, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data and len(data) > 0:
            return float(data[0]['trade_price'])
        return None
    except Exception as e:
        log(f'  KRW 가격 조회 실패: {e}')
        return None

def validate_past_predictions():
    """
    is_correct가 NULL인 예측에 대해 실제 가격으로 검증 및 업데이트.
    [중요] 31FH는 date에 '예측 대상 시점'(24h 후)을 저장함.
    → target_dt = pred_dt (date 자체가 검증 대상 시점, +24h 하면 안 됨!)
    """
    log('🔄 DB에서 is_correct가 NULL인 예측 데이터를 조회합니다...')
    resp = supabase.table("predictions").select("*").is_("is_correct", "null").execute()
    if not resp.data:
        log('✅ 검증할 누락 데이터가 없습니다.')
        return
    records = resp.data
    log(f'총 {len(records)}개의 미결 예측을 발견했습니다.')
    current_utc = datetime.now(timezone.utc)
    updated_count = 0
    for row in records:
        pred_date_str = row['date']
        try:
            pred_dt = pd.to_datetime(pred_date_str)
            if pred_dt.tzinfo is None:
                pred_dt = pred_dt.replace(tzinfo=timezone.utc)
            # date = 예측 대상 시점 (31FH: now+24h로 저장됨) → 이 시점의 가격이 actual_price
            target_dt = pred_dt  # +24h 하지 않음! date 자체가 목표 시점
            if current_utc < target_dt:
                log(f'⏳ {pred_date_str} 예측은 아직 목표 시점이 도래하지 않았습니다. (목표: {target_dt.strftime("%m-%d %H:%M")})')
                continue
            log(f'🔍 {pred_date_str} 예측 검증 중 ...')
            base_price = row.get('predicted_price')
            if not base_price:
                log('  ❌ 기준 가격(predicted_price)이 없어 계산 우회')
                continue
            actual_usd = get_historical_usd_price(target_dt)
            if not actual_usd:
                log('  ⚠️ Binance 1m API 실패 → features_master/Binance 1d 폴백')
                actual_usd = _get_actual_price_from_features_master(target_dt)
                if not actual_usd:
                    actual_usd = _get_actual_price_from_binance(target_dt)
            actual_krw = get_historical_krw_price(target_dt) if actual_usd else None
            if not actual_krw and actual_usd:
                actual_krw = estimate_krw_price_from_usd(actual_usd)
            time.sleep(0.2)
            if not actual_usd:
                log('  ❌ 실제 가격을 가져오지 못했습니다. (Binance 451/지역제한 시 features_master 확인)')
                continue
            change_pct = ((actual_usd - base_price) / base_price) * 100
            actual_dir = "UP" if change_pct > 0 else "DOWN"
            predicted_dir = row['direction']
            is_correct = (actual_dir == predicted_dir or
                         (actual_dir == "UP" and predicted_dir in ["상승", "UP", "1"]) or
                         (actual_dir == "DOWN" and predicted_dir in ["하락", "DOWN", "0"]))
            update_data = {
                'actual_price': actual_usd,
                'actual_price_krw': actual_krw if actual_krw else row.get('actual_price_krw'),
                'price_change_pct': round(change_pct, 2),
                'actual_result': actual_dir,
                'is_correct': is_correct
            }
            date_key = _format_date_for_eq(pred_date_str)
            supabase.table("predictions").update(update_data).eq("date", date_key).execute()
            log(f'  ✅ 업데이트 완료! 예측:{predicted_dir} | 실제:{actual_dir} | 정답여부:{is_correct} (변동:{change_pct:.2f}%)')
            updated_count += 1
        except Exception as e:
            log(f'  ❌ 에러 발생 ({pred_date_str}): {e}')
    log(f'🎉 총 {updated_count}개의 예측 데이터를 검증 및 업데이트 했습니다.')

# ==========================================
# actual_price / is_correct 백필 함수
# ==========================================

def _format_date_for_eq(date_val):
    """Supabase .eq() 필터용 date 포맷 통일 (insert 시 사용한 형식과 동일)"""
    dt = pd.to_datetime(date_val)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime('%Y-%m-%d %H:%M:%S+00:00')

def estimate_krw_price_from_usd(usd_price, exchange_rate=1450):
    """USD 가격을 KRW로 환산 (Backfill용)"""
    return usd_price * exchange_rate

def _get_actual_price_from_features_master(pred_date):
    """features_master에서 해당 시점 또는 직전의 close 가격 조회 (목표 시점에 가까운 데이터)"""
    try:
        ts_str = pd.to_datetime(pred_date).strftime('%Y-%m-%d %H:%M:%S')
        price_query = supabase.table('features_master').select('close').lte('date', ts_str).order('date', desc=True).limit(1).execute()
        if price_query.data and price_query.data[0].get('close'):
            return float(price_query.data[0]['close'])
        date_str = pd.to_datetime(pred_date).strftime('%Y-%m-%d')
        next_date_str = (pd.to_datetime(pred_date) + timedelta(days=1)).strftime('%Y-%m-%d')
        price_query = supabase.table('features_master').select('close').gte('date', date_str).lt('date', next_date_str).order('date', desc=True).limit(1).execute()
        if price_query.data and price_query.data[0].get('close'):
            return float(price_query.data[0]['close'])
    except Exception:
        pass
    return None

def _get_actual_price_from_binance(pred_date):
    """Binance API로 해당 날짜 종가 조회 (features_master 없을 때 폴백)"""
    try:
        ts = int(pd.Timestamp(pred_date).timestamp() * 1000)
        r = requests.get(f'https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&startTime={ts}&limit=1', timeout=10)
        if r.status_code == 200 and r.json():
            return float(r.json()[0][4])
    except Exception:
        pass
    return None

def backfill_missing_actual_prices():
    """actual_price가 NULL인 예측에 대해 features_master 또는 Binance에서 가격 채우기"""
    try:
        log('  🔄 누락된 actual_price 일괄 채우기...')
        response = supabase.table('predictions').select('*').is_('actual_price', 'null').execute()
        if not response.data:
            log('  ✅ 모든 actual_price가 이미 채워져 있습니다.')
            backfill_missing_actual_price_krw()
            backfill_validation_fields()
            return 0
        log(f'  📋 actual_price NULL인 레코드 {len(response.data)}개 발견')
        updated = 0
        for record in response.data:
            try:
                pred_date = pd.to_datetime(record['date'])
                actual_usd = _get_actual_price_from_features_master(pred_date)
                if actual_usd is None:
                    actual_usd = _get_actual_price_from_binance(pred_date)
                if actual_usd and actual_usd > 0:
                    actual_krw = estimate_krw_price_from_usd(actual_usd)
                    date_key = _format_date_for_eq(record['date'])
                    supabase.table('predictions').update({'actual_price': actual_usd, 'actual_price_krw': actual_krw}).eq('date', date_key).execute()
                    log(f'     {pred_date.strftime("%Y-%m-%d")}: ${actual_usd:,.2f} / ₩{actual_krw:,.0f}')
                    updated += 1
                else:
                    log(f'     ⚠️ {pred_date.strftime("%Y-%m-%d")}: features_master/Binance에서 가격 조회 실패')
            except Exception as e:
                log(f'     ⚠️ {record.get("date", "?")}: {e}')
        if updated > 0:
            log(f'  ✅ {updated}개 actual_price 업데이트 완료')
        backfill_missing_actual_price_krw()
        backfill_validation_fields()
        return updated
    except Exception as e:
        log(f'  ❌ actual_price 백필 실패: {e}')
        import traceback
        traceback.print_exc()
        return 0

def backfill_missing_actual_price_krw():
    """actual_price는 있지만 actual_price_krw가 NULL인 레코드 채우기"""
    try:
        response = supabase.table('predictions').select('*').not_.is_('actual_price', 'null').is_('actual_price_krw', 'null').execute()
        if not response.data:
            return 0
        for record in response.data:
            try:
                actual_krw = estimate_krw_price_from_usd(float(record['actual_price']))
                date_key = _format_date_for_eq(record['date'])
                supabase.table('predictions').update({'actual_price_krw': actual_krw}).eq('date', date_key).execute()
            except Exception:
                pass
        return len(response.data)
    except Exception:
        return 0

def backfill_validation_fields():
    """actual_price는 있지만 is_correct가 NULL인 레코드 채우기"""
    try:
        response = supabase.table('predictions').select('*').not_.is_('actual_price', 'null').is_('is_correct', 'null').execute()
        if not response.data:
            log('  ✅ 모든 검증 필드가 이미 채워져 있습니다.')
            return 0
        log(f'  🔄 검증 필드 채우기 ({len(response.data)}개)...')
        updated = 0
        for record in response.data:
            try:
                actual_price = float(record['actual_price'])
                predicted_price = record.get('predicted_price') or 0
                pred_direction = record.get('direction', 'UNKNOWN')
                if predicted_price > 0:
                    price_change_pct = ((actual_price - predicted_price) / predicted_price) * 100
                    actual_direction = 'UP' if actual_price > predicted_price else 'DOWN'
                    # direction 정규화: 상승/UP/1 → UP, 하락/DOWN/0 → DOWN
                    pred_norm = 'UP' if pred_direction in ['상승', 'UP', '1'] else ('DOWN' if pred_direction in ['하락', 'DOWN', '0'] else pred_direction)
                    is_correct = (pred_norm == actual_direction)
                    update_data = {
                        'actual_result': actual_direction,
                        'is_correct': is_correct,
                        'price_change_pct': round(price_change_pct, 2)
                    }
                    date_key = _format_date_for_eq(record['date'])
                    result = supabase.table('predictions').update(update_data).eq('date', date_key).execute()
                    if not result.data:
                        log(f'     ⚠️ 업데이트 매칭 실패 (date={date_key})')
                    status = '✅' if is_correct else '❌'
                    log(f'     {pd.to_datetime(record["date"]).strftime("%Y-%m-%d")}: {pred_direction} vs {actual_direction} {status}')
                    updated += 1
            except Exception as e:
                log(f'     ⚠️ {record.get("date", "?")}: {e}')
        if updated > 0:
            log(f'  ✅ {updated}개 is_correct 업데이트 완료')
        return updated
    except Exception as e:
        log(f'  ❌ 검증 필드 백필 실패: {e}')
        return 0

# ==========================================
# 메인 파이프라인
# ==========================================

def run_daily_pipeline():
    """
    일일 예측 파이프라인 (v7E)

    Step 1: 데이터 로드
    Step 2: 어제 예측 검증
    Step 3: 모델 로드
    Step 4: 예측 데이터 준비
    Step 5: 앙상블 예측
    Step 6: 증분 학습 (갭 데이터 있을 시)
    Step 7: 예측 저장
    Step 8: 모델 저장
    """
    pipeline_start = datetime.now(KST)
    print(f'\n{"#"*60}')
    print(f'🚀 v7E Daily Prediction Pipeline 시작')
    print(f'   시작 시간: {pipeline_start.strftime("%Y-%m-%d %H:%M:%S")} KST')
    print(f'{"#"*60}')

    try:
        # ============================================================
        # Step 0: 경로 강제 설정 (캐시 문제 방지)
        # ============================================================
        global MODEL_DIR, PROJECT_ROOT
        if IS_COLAB:
            PROJECT_ROOT = '/content/drive/MyDrive/2526Winter_Sideproject'
            MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'production', 'v7E_production_highAccuracy_dynH')
        log(f'  MODEL_DIR: {MODEL_DIR}')
        log(f'  파일 확인: {os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else "폴더 없음!"}')
        
        # ============================================================
        # Step 0.5: 과거 예측 검증 (01a 로직) + actual_price/is_correct 백필
        # ============================================================
        log('Step 0.5: 과거 예측 검증 및 백필 (actual_price, is_correct)', important=True)
        validate_past_predictions()  # 01a: is_correct NULL → 24h 뒤 1분캔들로 정확 검증
        backfill_missing_actual_prices()  # actual_price NULL → features_master/Binance 폴백
        
        # ============================================================
        # Step 1: 데이터 로드 (29F 로직과 동일)
        # ============================================================
        log('Step 1: 데이터 로드', important=True)
        
        # 최신 날짜 확인
        latest_date = get_latest_date_from_supabase()
        if latest_date is None:
            raise ValueError('Supabase에서 최신 날짜를 가져올 수 없습니다')
        log(f'  최신 데이터 날짜: {latest_date}')
        
        # -------------------------------------------------------
        # 데이터 로드 전략:
        #   - CatBoost 학습용: 최근 4000시간 (~167일, ~5.5개월)
        #   - 딥러닝 예측용: 그 중 마지막 72시간 시퀀스 사용
        #   - 타임아웃 방지: date 필터 + 1000행 배치 로드
        # -------------------------------------------------------
        CATBOOST_TRAIN_HOURS = 4000  # CatBoost 학습에 충분한 기간
        catboost_start = latest_date - timedelta(hours=CATBOOST_TRAIN_HOURS)
        log(f'  데이터 로드 범위: {catboost_start} ~ {latest_date} ({CATBOOST_TRAIN_HOURS}시간, ~{CATBOOST_TRAIN_HOURS//24}일)')
        
        # Supabase에서 날짜 필터로 로드 (타임아웃 방지)
        all_rows, offset = [], 0
        start_str = catboost_start.strftime('%Y-%m-%d %H:%M:%S')
        end_str = latest_date.strftime('%Y-%m-%d %H:%M:%S')
        
        while True:
            result = supabase.table('features_master').select('*').gte('date', start_str).lte('date', end_str).order('date').range(offset, offset + 999).execute()
            if not result.data:
                break
            all_rows.extend(result.data)
            offset += len(result.data)
            log(f'    배치 로드: {len(all_rows):,} rows...')
            if len(result.data) < 1000:
                break
        
        df_features = pd.DataFrame(all_rows)
        df_features['date'] = pd.to_datetime(df_features['date'])
        log(f'  features_master: {len(df_features):,} rows')
        
        # 감성 데이터 병합
        df_sent = fetch_sentiment_data()
        if not df_sent.empty:
            for col in ['sentiment_score', 'impact_score']:
                if col in df_features.columns:
                    df_features = df_features.drop(columns=[col], errors='ignore')
            df_features = pd.merge(df_features, df_sent, on='date', how='left')
        
        df_features['sentiment_score'] = df_features.get('sentiment_score', 0).fillna(0)
        df_features['impact_score'] = df_features.get('impact_score', 0.5).fillna(0.5)
        
        # close 컬럼 통일
        if 'close' not in df_features.columns and 'close_price' in df_features.columns:
            df_features['close'] = df_features['close_price']
        
        # target 생성
        df_features['close'] = pd.to_numeric(df_features['close'], errors='coerce')
        df_features['target'] = (df_features['close'].shift(-config.PREDICTION_HORIZON) > df_features['close']).astype(int)
        
        df_full = df_features.sort_values('date').reset_index(drop=True)
        log(f'✅ 데이터 준비 완료: {len(df_full):,} rows')


        # On-the-fly 파생 피처 생성 (Kaggle v7E 학습 시 생성된 11개)
        log('📐 On-the-fly 파생 피처 생성 중...')
        df_full = add_on_the_fly_features(df_full)
        
        # ============================================================
        # Step 2: 어제 예측 검증
        # ============================================================
        log('Step 2: 어제 예측 검증', important=True)
        validate_yesterday_prediction()

        # ============================================================
        # Step 3: 모델 로드
        # ============================================================
        models = load_all_models(df_full=df_full)
        features = models['features']

        # ============================================================
        # Step 4: 예측 데이터 준비
        # ============================================================
        log('Step 4: 예측 데이터 준비', important=True)

        min_rows = config.PRIMARY_SEQUENCE_LENGTH * 3
        df_recent = df_full.tail(max(min_rows, 300)).copy()
        log(f'  예측용 최근 데이터: {len(df_recent)} rows')

        # CatBoost: 마지막 행의 피처로 예측
        # CatBoost: 누락 피처 0으로 채움
        for f in features["catboost"]:
            if f not in df_recent.columns:
                df_recent[f] = 0
        X_cat_all = df_recent[features["catboost"]].fillna(0).values
        X_cat_scaled = models['scaler_catboost'].transform(X_cat_all)
        X_latest_cat = X_cat_scaled[[-1]]
        log(f'  CatBoost 입력: {X_latest_cat.shape} ({len(features["catboost"])} features)')

        # CNN-LSTM: 마지막 seq_len 행의 시퀀스
        # CNN-LSTM: 누락 피처 0으로 채움
        for f in features["cnnlstm"]:
            if f not in df_recent.columns:
                df_recent[f] = 0
        X_cnn_all = df_recent[features["cnnlstm"]].fillna(0).values
        X_cnn_scaled = models['scaler_cnnlstm'].transform(X_cnn_all)
        X_seq_cnn = X_cnn_scaled[-config.PRIMARY_SEQUENCE_LENGTH:]
        log(f'  CNN-LSTM 입력: {X_seq_cnn.shape}')

        # PatchTST: 마지막 seq_len 행의 시퀀스
        # PatchTST: 누락 피처 0으로 채움
        for f in features["patchtst"]:
            if f not in df_recent.columns:
                df_recent[f] = 0
        X_patch_all = df_recent[features["patchtst"]].fillna(0).values
        X_patch_scaled = models['scaler_patchtst'].transform(X_patch_all)
        X_seq_patch = X_patch_scaled[-config.PRIMARY_SEQUENCE_LENGTH:]
        log(f'  PatchTST 입력: {X_seq_patch.shape}')

        # ============================================================
        # Step 5: 앙상블 예측
        # ============================================================
        prediction, confidence, model_details = ensemble_predict_v7e(
            models, X_latest_cat, X_seq_cnn, X_seq_patch, df_recent
        )

        # ============================================================
        # Step 6: 증분 학습
        # ============================================================
        metadata = get_model_metadata()
        training_performed = False

        if metadata and metadata.get('last_data_date'):
            last_train_date = pd.to_datetime(metadata['last_data_date'])
            df_gap = df_full[df_full['date'] > last_train_date].copy()
            df_gap = df_gap.dropna(subset=['target'])

            if len(df_gap) >= 30:
                log(f'  갭 데이터: {len(df_gap)} rows (마지막 학습: {last_train_date})')
                training_performed = run_incremental_training(models, df_gap, features)
            else:
                log(f'  증분 학습 생략: 갭 데이터 {len(df_gap)}행 (최소 30 필요)')
        else:
            log('  모델 메타데이터 없음 - 증분 학습 생략')

        # ============================================================
        # Step 7: 예측 저장
        # ============================================================
        log('Step 7: 예측 결과 저장', important=True)

        now_utc = datetime.now(timezone.utc)
        prediction_target = now_utc + timedelta(hours=24)
        current_usd = get_realtime_btc_price_usd()
        current_krw = get_krw_bitcoin_price()

        save_prediction_to_supabase(
            prediction_target, prediction, confidence, model_details,
            current_price=current_usd, current_price_krw=current_krw
        )

        # ============================================================
        # Step 8: 모델 저장 (학습 수행 시)
        # ============================================================
        if training_performed:
            log('Step 8: 모델 저장', important=True)
            save_updated_models(models, last_data_date=latest_date)
        else:
            log('  학습 미수행 - 모델 저장 생략')

        # ============================================================
        # 최종 결과 출력
        # ============================================================
        pipeline_end = datetime.now(KST)
        elapsed = (pipeline_end - pipeline_start).total_seconds()

        kst_target = prediction_target.replace(tzinfo=timezone.utc).astimezone(KST)

        result = {
            'success': True,
            'prediction_date': kst_target.strftime('%Y-%m-%d %H:%M'),
            'prediction': prediction,
            'prediction_label': 'UP' if prediction == 1 else 'DOWN',
            'confidence': float(confidence),
            'predicted_accuracy_pct': model_details.get('predicted_accuracy_pct', 0),
            'model_details': model_details,
            'training_performed': training_performed,
            'current_price_usd': current_usd,
            'current_price_krw': current_krw,
        }

        print(f'\n\n{"#"*60}')
        print(f'🔮 v7E PREDICTION RESULT')
        print(f'{"#"*60}')
        print(f'')
        print(f'  예측 대상 시점: {kst_target.strftime("%Y-%m-%d %H:%M")} KST')
        print(f'  예측 결과: {"🟢 UP (상승)" if prediction == 1 else "🔴 DOWN (하락)"}')
        if current_usd:
            print(f'  현재 가격: ${current_usd:,.2f}')
        if current_krw:
            print(f'  현재 가격: ₩{current_krw:,.0f}')
        print(f'  신뢰도: {confidence*100:.1f}%')
        print(f'  예상 정확도: {model_details.get("predicted_accuracy_pct", 0):.1f}%')
        print(f'  Regime: {model_details.get("regime", "unknown")}')
        print(f'')
        print(f'  개별 모델:')
        for name, prob in model_details.get('individual_predictions', {}).items():
            print(f'    {name}: {prob:.4f} ({"UP" if prob > 0.5 else "DOWN"})')
        print(f'  Meta-Learner Stacking: {model_details.get("meta_stacking_probability", 0):.4f}')
        print(f'  Regime Dynamic: {model_details.get("regime_probability", 0):.4f}')
        print(f'  Final: {model_details.get("final_probability", 0):.4f}')
        print(f'')
        print(f'  증분 학습: {"수행됨" if training_performed else "생략"}')
        print(f'  소요 시간: {elapsed:.1f}초')
        print(f'{"#"*60}')

        return result

    except Exception as e:
        log(f'❌ 파이프라인 실패: {e}', important=True)
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

log('✅ 메인 파이프라인 정의 완료')

# ## ▶️ 8. 실행!

# ==========================================
# 실행!
# ==========================================
result = run_daily_pipeline()


