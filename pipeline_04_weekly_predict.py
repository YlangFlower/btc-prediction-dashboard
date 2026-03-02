# # 🔮 Weekly BTC Active/Quiet Prediction - v3.5 Load
# **monster_kaggle_weekly_v3.5 가중치 로드 → 예측 → DB 저장**
# 
# ## 🎯 주요 기능:
# 1. 📊 Supabase features_daily 로드
# 2. 🤖 가중치 로드 (CatBoost 5-Fold, PatchTST 5-Fold)
# 3. 🔮 다음 7일 Active/Quiet 예측
# 4. 💾 weekly_predictions 테이블에 저장
# 
# ## 📋 필요한 가중치 파일 (Google Drive):
# `2526_Winter_Sideproject/models/production/weekly_v35/`
# - catboost_f0.cbm ~ catboost_f4.cbm
# - patchtst_f0.pth ~ patchtst_f4.pth
# - scalers_weekly.pkl
# - model_features_weekly.json

# ## 📦 0. 패키지 설치 & 임포트



# ==========================================
# 필수 라이브러리 임포트
# ==========================================
import os
import sys
import json
import pickle
import warnings
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import RobustScaler
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

KST = timezone(timedelta(hours=9))

def log(message, important=False):
    kst_now = datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')
    msg = str(message)
    if important:
        print(f'\n{"*"*60}')
        print(f'[{kst_now}] ⭐ {msg}')
        print(f'{"*"*60}')
    else:
        print(f'[{kst_now}] {msg}')
    sys.stdout.flush()

log('✅ 패키지 임포트 완료')

# ## 🛠️ 1. 환경 설정 & Supabase (.env)

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
# 🚀 2. 변수 할당 및 모델/Supabase 연결
# ==============================================================================
PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd())
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'production', 'weekly_v35')

log(f'  PROJECT_ROOT: {PROJECT_ROOT}')
log(f'  MODEL_DIR: {MODEL_DIR}')

if not os.path.exists(MODEL_DIR):
    # 경고만 해두고 나중에 쓸수있게
    log(f'⚠️ 모델 폴더가 없습니다: {MODEL_DIR}')
else:
    files_in_dir = os.listdir(MODEL_DIR)
    log(f'  파일 목록: {files_in_dir}')

# Supabase 연결
SUPABASE_URL = os.getenv("SUPABASE_URL")
SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", os.getenv("SUPABASE_KEY")) 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SERVICE_KEY:
    raise ValueError("❌ 치명적 오류: SUPABASE_URL 또는 SUPABASE_SERVICE_KEY를 찾을 수 없습니다!")

from supabase import create_client
supabase = create_client(SUPABASE_URL, SERVICE_KEY)

log("✅ Supabase 마스터 권한(Service Role) 연결 성공 🔓")

# ==============================================================================
# 📥 [GitHub Actions 전용] Supabase Storage에서 주간 모델 가중치 자동 다운로드
# ==============================================================================
if IS_GITHUB_ACTIONS:
    log("🔽 GitHub Actions: Supabase Storage에서 주간 모델 가중치 다운로드 시작...", important=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    STORAGE_BUCKET = "models"
    STORAGE_FOLDER = "weekly_v35"
    FILES_TO_DOWNLOAD = [
        "catboost_f0.cbm", "catboost_f1.cbm", "catboost_f2.cbm", "catboost_f3.cbm", "catboost_f4.cbm",
        "patchtst_f0.pth", "patchtst_f1.pth", "patchtst_f2.pth", "patchtst_f3.pth", "patchtst_f4.pth",
        "scalers_weekly.pkl", "model_features_weekly.json",
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
    log("🎉 모든 주간 모델 가중치 다운로드 완료!", important=True)



# ## ⚙️ 2. Config & PatchTST 아키텍처 (v3.5)

# ==========================================
# Config - monster_kaggle_weekly_v3.5와 동일
# ==========================================
class Config:
    SEQ_LEN = 26
    HORIZON = 7
    NUM_CLASSES = 2
    D_MODEL = 128
    N_HEADS = 4
    N_LAYERS = 4
    PATCH_LEN = 7
    STRIDE = 3
    DROPOUT = 0.4
    USE_REVIN = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = Config()
log(f'Config: SEQ_LEN={cfg.SEQ_LEN}, DEVICE={cfg.DEVICE}')

# ==========================================
# RevIN + EnhancedPatchTST (v3.5)
# ==========================================
class RevIN(nn.Module):
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
            if self.affine:
                x = x * self.gamma + self.beta
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.beta) / self.gamma
            x = x * self.std + self.mean
        return x


class EnhancedPatchTST(nn.Module):
    def __init__(self, n_feat, seq_len):
        super().__init__()
        self.revin = RevIN(n_feat) if cfg.USE_REVIN else None
        self.n_patches = (seq_len - cfg.PATCH_LEN) // cfg.STRIDE + 1
        self.embed = nn.Sequential(
            nn.Linear(cfg.PATCH_LEN * n_feat, cfg.D_MODEL),
            nn.LayerNorm(cfg.D_MODEL), nn.GELU(), nn.Dropout(cfg.DROPOUT)
        )
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, cfg.D_MODEL))
        enc = nn.TransformerEncoderLayer(
            cfg.D_MODEL, cfg.N_HEADS, cfg.D_MODEL * 4,
            cfg.DROPOUT, activation='gelu', batch_first=True, norm_first=True
        )
        self.trans = nn.TransformerEncoder(enc, cfg.N_LAYERS)
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.D_MODEL),
            nn.Linear(cfg.D_MODEL, cfg.D_MODEL // 2), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.D_MODEL // 2, cfg.D_MODEL // 4), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.D_MODEL // 4, cfg.NUM_CLASSES)
        )

    def forward(self, x):
        if self.revin:
            x = self.revin(x, 'norm')
        B = x.shape[0]
        patches = [x[:, i * cfg.STRIDE:i * cfg.STRIDE + cfg.PATCH_LEN, :].reshape(B, -1)
                  for i in range(self.n_patches)]
        x = self.embed(torch.stack(patches, dim=1)) + self.pos
        x = self.trans(x)
        return self.classifier(x.mean(dim=1) + x.max(dim=1)[0])

log('✅ PatchTST 아키텍처 정의 완료')

# ## 🤖 3. 모델 로드

# ==========================================
# model_features_weekly.json 로드
# ==========================================
features_path = os.path.join(MODEL_DIR, 'model_features_weekly.json')
with open(features_path, 'r') as f:
    meta = json.load(f)

feat_cat = meta['catboost']
feat_patch = meta['patchtst']
active_threshold = meta['active_threshold']
best_strategy = meta.get('best_strategy', 'A: Simple Avg')
boundary = meta.get('boundary', 0.02)
target_hits = meta.get('target_hits', 4)
model_version = meta.get('version', 'v3.5')

log(f'피처: CatBoost={len(feat_cat)}, PatchTST={len(feat_patch)}')
log(f'active_threshold={active_threshold}, best_strategy={best_strategy}')

# ==========================================
# scalers_weekly.pkl 로드
# ==========================================
scalers_path = os.path.join(MODEL_DIR, 'scalers_weekly.pkl')
with open(scalers_path, 'rb') as f:
    scalers = pickle.load(f)

sc_cat = scalers['catboost']
sc_patch = scalers['patchtst']
log('✅ 스케일러 로드 완료')

# ==========================================
# CatBoost 5-Fold 로드
# ==========================================
cat_models = []
for fold in range(5):
    path = os.path.join(MODEL_DIR, f'catboost_f{fold}.cbm')
    if not os.path.exists(path):
        log(f'  ⚠️ {path} 없음')
        continue
    m = CatBoostClassifier()
    m.load_model(path)
    cat_models.append(m)

if len(cat_models) == 0:
    raise FileNotFoundError('CatBoost 모델이 없습니다. catboost_f0.cbm ~ f4.cbm을 MODEL_DIR에 넣어주세요.')
log(f'✅ CatBoost: {len(cat_models)}/5 fold 로드')

# ==========================================
# PatchTST 5-Fold 로드
# ==========================================
patch_models = []
for fold in range(5):
    path = os.path.join(MODEL_DIR, f'patchtst_f{fold}.pth')
    if not os.path.exists(path):
        log(f'  ⚠️ {path} 없음')
        continue
    model = EnhancedPatchTST(len(feat_patch), cfg.SEQ_LEN)
    ckpt = torch.load(path, map_location=cfg.DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(cfg.DEVICE)
    model.eval()
    patch_models.append(model)

if len(patch_models) == 0:
    raise FileNotFoundError('PatchTST 모델이 없습니다. patchtst_f0.pth ~ f4.pth를 MODEL_DIR에 넣어주세요.')
log(f'✅ PatchTST: {len(patch_models)}/5 fold 로드')

# ## 📊 4. 데이터 로드 & Regime 피처

# ==========================================
# features_daily 로드
# ==========================================
def fetch_features_daily():
    log('Loading features_daily...')
    all_rows, offset = [], 0
    while True:
        result = supabase.table('features_daily').select('*').order('date').range(offset, offset + 999).execute()
        if not result.data:
            break
        all_rows.extend(result.data)
        if len(result.data) < 1000:
            break
        offset += 1000
    df = pd.DataFrame(all_rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    log(f'  {len(df):,} rows ({df["date"].min().date()} ~ {df["date"].max().date()})')
    return df

df_full = fetch_features_daily()

if len(df_full) < cfg.SEQ_LEN:
    raise ValueError(f'데이터 부족: {len(df_full)}행 (최소 {cfg.SEQ_LEN}행 필요)')

# ==========================================
# Regime 피처 생성 (monster_kaggle_weekly Cell 4)
# ==========================================
for col in ['vol_lag_24', 'vol_lag_48']:
    if col in df_full.columns:
        df_full[col] = df_full[col].clip(-10.0, 10.0)

daily_ret = df_full['close'].pct_change()
rolling_vol = daily_ret.rolling(26).std()

existing_cols = set(df_full.columns)
close = df_full['close']
sma20 = close.rolling(20).mean()
sma60 = close.rolling(60).mean()

def add_if_new(col_name, series):
    if col_name not in existing_cols:
        df_full[col_name] = series
        existing_cols.add(col_name)

add_if_new('regime_bull_flag', ((close > sma20) & (sma20 > sma60)).astype(float))
add_if_new('regime_bear_flag', ((close < sma20) | (close < sma60 * 0.95)).astype(float))
vol_med_exp = rolling_vol.expanding(min_periods=26).median()
add_if_new('regime_highvol_flag', (rolling_vol > vol_med_exp).astype(float))
vol_std_exp = rolling_vol.expanding(min_periods=26).std()
add_if_new('vol_regime_zscore', ((rolling_vol - vol_med_exp) / (vol_std_exp + 1e-8)).clip(-3, 3))
low52 = close.rolling(52).min()
high52 = close.rolling(52).max()
add_if_new('price_52w_position', (close - low52) / (high52 - low52 + 1e-8))
add_if_new('volatility_26w', rolling_vol)

log('✅ Regime 피처 생성 완료')

# ## 🔮 5. 예측

# ==========================================
# 예측 입력 준비
# ==========================================
missing_cat = [f for f in feat_cat if f not in df_full.columns]
missing_patch = [f for f in feat_patch if f not in df_full.columns]
if missing_cat:
    log(f'  ⚠️ CatBoost 누락 피처: {missing_cat}')
    for f in missing_cat:
        df_full[f] = 0
if missing_patch:
    log(f'  ⚠️ PatchTST 누락 피처: {missing_patch}')
    for f in missing_patch:
        df_full[f] = 0

df_tail = df_full.tail(cfg.SEQ_LEN).copy()
X_patch_raw = df_tail[feat_patch].fillna(0).values
X_patch_scaled = sc_patch.transform(X_patch_raw)
X_patch_seq = torch.FloatTensor(X_patch_scaled).unsqueeze(0).to(cfg.DEVICE)

X_cat_raw = df_full.iloc[-1:][feat_cat].fillna(0).values
X_cat_scaled = sc_cat.transform(X_cat_raw)

last_date = df_full['date'].iloc[-1]
log(f'예측 기준일: {last_date.date()}')

# ==========================================
# CatBoost 5-Fold 예측 평균
# ==========================================
p_cat_list = []
for m in cat_models:
    p = m.predict_proba(X_cat_scaled)
    p_cat_list.append(p)

p_cat = np.mean(p_cat_list, axis=0)
log(f'CatBoost P(Active)={p_cat[0, 1]:.4f}')

# ==========================================
# PatchTST 5-Fold 예측 평균
# ==========================================
p_patch_list = []
with torch.no_grad():
    for m in patch_models:
        out = m(X_patch_seq)
        p = torch.softmax(out, dim=1).cpu().numpy()
        p_patch_list.append(p)

p_patch = np.mean(p_patch_list, axis=0)
log(f'PatchTST P(Active)={p_patch[0, 1]:.4f}')

# ==========================================
# 앙상블 (best_strategy)
# ==========================================
if 'Simple' in best_strategy or 'A:' in best_strategy:
    probs = (p_cat + p_patch) / 2.0
elif 'B:' in best_strategy or 'OOF' in best_strategy:
    probs = (p_cat + p_patch) / 2.0
elif 'C:' in best_strategy or 'Confidence' in best_strategy:
    conf_cat = np.abs(p_cat[0, 1] - 0.5)
    conf_patch = np.abs(p_patch[0, 1] - 0.5)
    total_c = conf_cat + conf_patch + 1e-8
    probs = (conf_cat * p_cat + conf_patch * p_patch) / total_c
else:
    probs = (p_cat + p_patch) / 2.0

p_active = float(probs[0, 1])
prediction = 1 if p_active >= active_threshold else 0
confidence = max(p_active, 1 - p_active)

log('', important=True)
log(f'🎯 예측: {"🟢 ACTIVE" if prediction == 1 else "🔴 QUIET"}')
log(f'   P(Active)={p_active:.4f}, threshold={active_threshold}')
log(f'   신뢰도={confidence:.4f}')

# ## 💾 6. DB 저장 (weekly_predictions)

# ==========================================
# prediction_week_start = 돌아오는 월요일
# (매주 일요일 실행 기준: 일요일 → 다음날 월요일부터 한 주)
# ==========================================
today_kst = datetime.now(KST).date()

# weekday(): 0=월, 1=화, ..., 6=일
# 돌아오는 월요일까지 남은 일수
days_to_monday = (7 - today_kst.weekday()) % 7
# 오늘이 월요일이면 그대로 이번 주 사용
prediction_week_start = today_kst + timedelta(days=days_to_monday)  # 돌아오는 월요일 (또는 오늘)

log(f'예측 기준: 오늘(실행) {today_kst} ({["월","화","수","목","금","토","일"][today_kst.weekday()]}) → '
    f'예측 주간 {prediction_week_start}(월) ~ {prediction_week_start + timedelta(days=6)}(일)')

# ==========================================
# weekly_predictions 테이블에 저장
# ==========================================
data = {
    'prediction_week_start': prediction_week_start.isoformat(),
    'prediction': int(prediction),
    'p_active': float(p_active),
    'confidence': float(confidence),
    'boundary': float(boundary),
    'target_hits': int(target_hits),
    'model_version': model_version,
    'predicted_at': datetime.now(timezone.utc).isoformat(),
    'model_details': {
        'best_strategy': best_strategy,
        'active_threshold': active_threshold,
        'p_cat_active': float(p_cat[0, 1]),
        'p_patch_active': float(p_patch[0, 1]),
    }
}

try:
    supabase.table('weekly_predictions').upsert(data, on_conflict='prediction_week_start').execute()
    log('✅ weekly_predictions 저장 완료', important=True)
except Exception as e:
    log(f'❌ 저장 실패: {e}', important=True)
    raise

# ==========================================
# 요약 출력 (일일 예측 연동용 참조)
# ==========================================
pred_end = prediction_week_start + timedelta(days=6)
print(f'\n{"="*60}')
print('📋 주간 예측 결과 요약')
print(f'{"="*60}')
print(f'  예측 주: {prediction_week_start} ~ {pred_end}')
print(f'  결과: {"ACTIVE" if prediction == 1 else "QUIET"}')
print(f'  P(Active): {p_active:.4f}')
print(f'  신뢰도: {confidence:.4f}')
print(f'\n  [개별 모델] CatBoost P(Active)={p_cat[0,1]:.4f}  PatchTST P(Active)={p_patch[0,1]:.4f}')
print(f'  [정의] BOUNDARY=±{boundary:.2%}, 7일 중 {target_hits}회 이상 터치 → Active')
print(f'  [거래 해석] {"ACTIVE → 일일 신호 공격적 활용" if prediction == 1 else "QUIET → 일일 신호 보수적 활용"}')
print(f'\n  → 33_market_analysis_daily_weekly에서 predictions + weekly_predictions 조회하여 시장 분석')
print(f'{"="*60}')

# ## 📌 참고: DB 스키마/데이터 조회 SQL
# 33_market_analysis_daily_weekly 작성 시 아래 SQL로 스키마 확인 후, 결과를 공유해주세요.

# 아래 SQL을 Supabase Dashboard > SQL Editor에서 실행 후, predictions 스키마 결과를 복사해서 공유
print('=== 1. predictions 테이블 스키마 조회 (결과 복사해서 공유) ===')
print('''SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'predictions'
ORDER BY ordinal_position;''')
print()
print('=== 2. weekly_predictions 테이블 스키마 조회 ===')
print('''SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'weekly_predictions'
ORDER BY ordinal_position;''')
print()
print('=== 3. predictions 데이터 조회 (최근 30건) ===')
print('''SELECT * FROM predictions ORDER BY date DESC LIMIT 30;''')
print()
print('=== 4. weekly_predictions 데이터 조회 (최근 10건) ===')
print('''SELECT * FROM weekly_predictions ORDER BY prediction_week_start DESC LIMIT 10;''')
