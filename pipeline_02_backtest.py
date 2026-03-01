"""
# 📊 pipeline_02_backtest.py — v7E 앙상블 백테스트 (독립 실행 스크립트)
#
# ⚠️  실행 방법:
#   이 스크립트는 pipeline_02_predict.py의 코드(모델, 함수, 환경)에 의존합니다.
#   반드시 아래 순서로 실행하세요:
#
#   1. python pipeline_02_predict.py   ← 모델 & 환경 세팅
#   2. python pipeline_02_backtest.py  ← 백테스팅 실행
#
#   또는 Colab/Jupyter 에서는 pipeline_02 셀들을 모두 실행한 뒤 이 파일을 실행하세요.
#
# 출력:
#   - 콘솔에 BACKTEST SUMMARY 출력
#   - MODEL_DIR/backtest_v7e.png 저장
"""

# pipeline_02_predict.py 의 전역 변수/함수(log, supabase, MODEL_DIR, 모델 로딩 함수 등)가
# 이미 현재 네임스페이스에 존재한다고 가정합니다.
# 만약 독립 실행할 경우 아래 exec 줄의 주석을 해제하세요:
# exec(open('pipeline_02_predict.py', encoding='utf-8').read())

import io
import contextlib

# ============================================================
# 백테스트 (표준 방식: 과거 학습 → 미래 검증)
# - 앙상블 모델/가중치/세팅 그대로 사용
# - 학습 종료일 이후 구간만 테스트 (데이터 유출 없음)
# ============================================================

# 옵션 (기존 노트북 세팅과 동일)
TRAINING_END_DATE = "2024-02-21"  # 모델 학습 종료일 (이 날짜 이후만 테스트, 과거→미래)
BACKTEST_DAYS = 365               # 백테스트 기간 (1년)
INITIAL_CAPITAL_KRW = 10_000_000
FEE_RATE = 0.00189                # 업비트 원화: 0.05% + 0.139%
CONFIDENCE_THRESHOLD = 0.52       # 최소 신뢰도 (이 이상일 때만 매매, 불확실 구간 스킵)

def run_backtest_v7e():
    """v7E 앙상블 모델로 백테스트 (표준 방식: 과거 학습 → 미래 검증)"""
    log('백테스트 시작 (v7E 앙상블, 과거→미래 검증)', important=True)
    
    if 'supabase' not in globals() or supabase is None:
        log('Supabase가 초기화되지 않았습니다. 위 셀들을 먼저 실행해주세요.')
        return
    
    # 1. 최신 날짜 및 백테스트 구간 (과거 학습 → 미래 검증)
    latest_date = get_latest_date_from_supabase()
    if latest_date is None:
        log('Supabase에서 최신 날짜를 가져올 수 없습니다.')
        return
    if latest_date.tzinfo is None:
        latest_date = latest_date.replace(tzinfo=timezone.utc)
    
    backtest_end = latest_date - timedelta(hours=24)
    if backtest_end.tzinfo is not None:
        backtest_end = backtest_end.tz_localize(None)
    backtest_start = max(
        pd.to_datetime(TRAINING_END_DATE),
        backtest_end - timedelta(days=BACKTEST_DAYS)
    )
    if backtest_start.tzinfo is not None:
        backtest_start = backtest_start.tz_localize(None)
    
    load_start = backtest_start - timedelta(hours=96)  # 72h history + 24h buffer
    load_end = backtest_end + timedelta(hours=24)      # 마지막 포지션 청산용 close 필요
    
    log(f'백테스트 구간: {backtest_start.strftime("%Y-%m-%d")} ~ {backtest_end.strftime("%Y-%m-%d")} (학습 종료일 이후)')
    
    # 2. 데이터 로드 (백테스트 + 72h history + 24h 청산용)
    all_rows, offset = [], 0
    start_str = load_start.strftime('%Y-%m-%d %H:%M:%S')
    end_str = load_end.strftime('%Y-%m-%d %H:%M:%S')
    while True:
        result = supabase.table('features_master').select('*').gte('date', start_str).lte('date', end_str).order('date').range(offset, offset + 999).execute()
        if not result.data:
            break
        all_rows.extend(result.data)
        offset += len(result.data)
        if len(result.data) < 1000:
            break
    if not all_rows:
        log('백테스트용 데이터가 없습니다.')
        return
    
    df = pd.DataFrame(all_rows)
    df['date'] = pd.to_datetime(df['date'])
    if pd.api.types.is_datetime64tz_dtype(df['date']):
        df['date'] = df['date'].dt.tz_localize(None)
    df = df.sort_values('date').reset_index(drop=True)
    
    # 감성 병합
    df_sent = fetch_sentiment_data()
    if not df_sent.empty:
        for col in ['sentiment_score', 'impact_score']:
            if col in df.columns:
                df = df.drop(columns=[col], errors='ignore')
        df = pd.merge(df, df_sent, on='date', how='left')
    df['sentiment_score'] = df.get('sentiment_score', 0).fillna(0)
    df['impact_score'] = df.get('impact_score', 0.5).fillna(0.5)
    
    if 'close' not in df.columns and 'close_price' in df.columns:
        df['close'] = df['close_price']
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = add_on_the_fly_features(df)
    
    # 3. 모델 로드 (데이터 유출 방지: 디스크 원본만 사용, 증분 학습 X)
    log('모델 로드 중 (디스크 원본)...')
    models = load_all_models(df_full=df)
    features = models['features']
    
    # 4. 롤링 예측 (stdout 억제)
    seq_len = config.PRIMARY_SEQUENCE_LENGTH
    preds = {}
    for i in range(seq_len - 1, len(df) - 24):
        row_date = df['date'].iloc[i]
        if row_date < backtest_start or row_date > backtest_end:
            continue
        df_slice = df.iloc[max(0, i - seq_len + 1):i + 1].copy()
        for f in features['catboost']:
            if f not in df_slice.columns:
                df_slice[f] = 0
        for f in features['cnnlstm']:
            if f not in df_slice.columns:
                df_slice[f] = 0
        for f in features['patchtst']:
            if f not in df_slice.columns:
                df_slice[f] = 0
        X_cat = models['scaler_catboost'].transform(df_slice[features['catboost']].fillna(0).values)
        X_cnn = models['scaler_cnnlstm'].transform(df_slice[features['cnnlstm']].fillna(0).values)
        X_patch = models['scaler_patchtst'].transform(df_slice[features['patchtst']].fillna(0).values)
        X_latest_cat = X_cat[[-1]]
        X_seq_cnn = X_cnn[-seq_len:]
        X_seq_patch = X_patch[-seq_len:]
        with contextlib.redirect_stdout(io.StringIO()):
            pred, confidence, _ = ensemble_predict_v7e(models, X_latest_cat, X_seq_cnn, X_seq_patch, df_slice)
        if confidence >= CONFIDENCE_THRESHOLD:
            preds[i] = pred
            if len(preds) % 1000 == 0:
                log(f'  예측 진행: {len(preds):,}개')
    
    log(f'총 {len(preds):,}개 시점 예측 완료')
    
    # 5. 수익률 계산 (24h 겹침, 포지션당 capital/24)
    cum_return = 0.0
    equity_curve = [INITIAL_CAPITAL_KRW]
    date_curve = [backtest_start]
    close_arr = df['close'].values
    wins, total = 0, 0
    for i in sorted(preds.keys()):
        c0, c24 = float(close_arr[i]), float(close_arr[i + 24])
        pred = preds[i]
        if pred == 1:  # UP (롱)
            ret = (c24 - c0) / c0
        else:  # DOWN (숏)
            ret = (c0 - c24) / c0
        ret_after_fee = ret - FEE_RATE
        cum_return += ret_after_fee / 24
        equity_curve.append(INITIAL_CAPITAL_KRW * (1 + cum_return))
        date_curve.append(df['date'].iloc[i + 24])
        total += 1
        if ret_after_fee > 0:
            wins += 1
    
    # 6. 결과 요약 + MDD 등 추가 지표
    final_equity = equity_curve[-1] if equity_curve else INITIAL_CAPITAL_KRW
    total_ret_pct = (final_equity / INITIAL_CAPITAL_KRW - 1) * 100
    win_rate = (wins / total * 100) if total > 0 else 0
    
    eq_arr = np.array(equity_curve)
    peak = np.maximum.accumulate(eq_arr)
    drawdown_pct = (eq_arr - peak) / peak * 100
    mdd_pct = float(np.min(drawdown_pct))
    days_bt = (date_curve[-1] - date_curve[0]).total_seconds() / 86400 if len(date_curve) > 1 else 1
    cagr_pct = ((final_equity / INITIAL_CAPITAL_KRW) ** (365 / max(days_bt, 1)) - 1) * 100 if days_bt > 0 else 0
    
    returns_per_trade = []
    for i in sorted(preds.keys()):
        c0, c24 = float(close_arr[i]), float(close_arr[i + 24])
        pred = preds[i]
        ret = (c24 - c0) / c0 if pred == 1 else (c0 - c24) / c0
        returns_per_trade.append(ret - FEE_RATE)
    ret_std = float(np.std(returns_per_trade)) * 100 if returns_per_trade else 0
    mean_ret = float(np.mean(returns_per_trade)) * 100 if returns_per_trade else 0
    trades_per_day = len(returns_per_trade) / max(days_bt, 1)
    sharpe = (mean_ret / (ret_std + 1e-9)) * np.sqrt(365 * trades_per_day) if ret_std > 0 else 0
    
    log(f'백테스트 결과: {total:,}건 | 승률 {win_rate:.1f}% | 총수익률 {total_ret_pct:+.2f}% | 최종 {final_equity/1e6:.2f}백만원')
    
    # 7. 텍스트 요약
    summary = f'''
========== BACKTEST SUMMARY ==========
Period      : {backtest_start.strftime("%Y-%m-%d")} ~ {backtest_end.strftime("%Y-%m-%d")} ({days_bt:.0f} days)
Trades      : {total:,} (confidence>={CONFIDENCE_THRESHOLD})
Win Rate    : {win_rate:.1f}%

--- Strategy (Long/Short) ---
  Long when UP predicted, Short when DOWN predicted. 24h hold per position.
Initial     : {INITIAL_CAPITAL_KRW/1e6:.2f}M KRW
Final       : {final_equity/1e6:.2f}M KRW
Total Return: {total_ret_pct:+.2f}%
CAGR        : {cagr_pct:+.2f}%

--- Risk ---
MDD         : {mdd_pct:.2f}%
Sharpe      : {sharpe:.2f}
Volatility  : {ret_std:.2f}% (per trade)

--- Tuning Tips ---
  - CONFIDENCE_THRESHOLD: Higher (e.g. 0.55) = fewer trades, potentially higher quality
  - Lower = more trades, may increase noise
======================================
'''
    print(summary)
    
    # 8. 차트 (영문, MDD 포함)
    df_bt = df[(df['date'] >= backtest_start) & (df['date'] <= backtest_end)]
    if df_bt.empty or len(date_curve) < 2:
        log('차트 생성할 데이터 부족')
        return
    
    close_bt = df_bt['close'].values.astype(float)
    btc_index = (close_bt / close_bt[0]) * 100
    dates_bt = df_bt['date'].values
    
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[1.5, 0.8], sharex=True)
    ax1, ax2 = axes[0], axes[1]
    
    ax1.plot(dates_bt, btc_index, color='gray', alpha=0.7, label='BTC (100=Start)')
    eq_dates = pd.to_datetime(date_curve)
    eq_pct = (np.array(equity_curve) / INITIAL_CAPITAL_KRW) * 100
    ax1.plot(eq_dates, eq_pct, color='#22c55e', linewidth=2, label=f'Strategy ({final_equity/1e6:.1f}M)')
    ax1.axhline(100, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Index (100=Initial)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'v7E Backtest ({backtest_start.strftime("%Y-%m-%d")} ~ {backtest_end.strftime("%Y-%m-%d")}) | Return {total_ret_pct:+.2f}% | MDD {mdd_pct:.2f}% | Long/Short')
    
    ax2.fill_between(eq_dates, drawdown_pct, 0, color='#ef4444', alpha=0.5)
    ax2.plot(eq_dates, drawdown_pct, color='#dc2626', linewidth=1)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Maximum Drawdown')
    
    plt.tight_layout()
    
    save_path = os.path.join(MODEL_DIR, 'backtest_v7e.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    log(f'차트 저장: {save_path}')
    plt.show()


# ==========================================
# 실행
# ==========================================
if __name__ == '__main__':
    run_backtest_v7e()
