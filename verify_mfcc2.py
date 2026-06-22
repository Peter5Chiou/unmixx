import librosa
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import sys

def get_metrics(path1, path2, sr=24000):
    # 1. 載入音訊
    y1, _ = librosa.load(path1, sr=sr)
    y2, _ = librosa.load(path2, sr=sr)
    
    # 2. 簡單 VAD 去除前後靜音，避免靜音段干擾相關性計算
    y1_trim, _ = librosa.effects.trim(y1, top_db=25)
    y2_trim, _ = librosa.effects.trim(y2, top_db=25)
    
    # --- 指標 A: MFCC 音色指紋 ---
    def get_mfcc_fp(y):
        # 提取 MFCC (20 階，跳過第 0 階)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=1024, hop_length=256)
        fp = np.mean(mfcc[1:, :], axis=1)
        # 標準化
        fp = (fp - np.mean(fp)) / (np.std(fp) + 1e-6)
        return torch.from_numpy(fp).float()

    fp1 = get_mfcc_fp(y1_trim)
    fp2 = get_mfcc_fp(y2_trim)
    mfcc_sim = F.cosine_similarity(fp1.unsqueeze(0), fp2.unsqueeze(0)).item()

    # --- 指標 B: 時間包絡相關性 (Temporal Correlation) ---
    # 因為要計算相關性，兩者長度必須一致
    min_len = min(len(y1_trim), len(y2_trim))
    s1 = y1_trim[:min_len]
    s2 = y2_trim[:min_len]
    
    # 取能量包絡 (取絕對值並做簡單平滑化)
    # 使用 20ms 的滑動平均來模擬人耳對能量起伏的感知
    win = int(sr * 0.02)
    env1 = np.convolve(np.abs(s1), np.ones(win)/win, mode='same')
    env2 = np.convolve(np.abs(s2), np.ones(win)/win, mode='same')
    
    # 計算皮爾森相關係數 (Pearson Correlation)
    # 公式: cov(1,2) / (std1 * std2)
    env1 = (env1 - np.mean(env1)) / (np.std(env1) + 1e-6)
    env2 = (env2 - np.mean(env2)) / (np.std(env2) + 1e-6)
    time_corr = np.mean(env1 * env2)

    return mfcc_sim, time_corr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file1", help="主聲軌道檔案")
    parser.add_argument("file2", help="合聲軌道檔案")
    args = parser.parse_args()

    try:
        mfcc_sim, time_corr = get_metrics(args.file1, args.file2)
    except Exception as e:
        print(f"處理失敗: {e}")
        sys.exit(1)

    print("\n" + "="*45)
    print(f"【測試結果分析】")
    print(f"1. MFCC 音色相似度: {mfcc_sim:.4f}")
    print(f"2. 時間動作同步率:   {time_corr:.4f}")
    print("="*45)
    
    # 判定邏輯驗證
    print("【判定模擬】")
    if mfcc_sim > 0.94 and time_corr > 0.98:
        print("結果 -> [ ❌ 幻覺/影子 ]")
        print("理由: 音色極度一致且動作完美同步，這不是二重唱，是主聲滲透。")
    elif mfcc_sim > 0.90 and time_corr < 0.96:
        print("結果 -> [ ✅ 真實二重唱 ]")
        print("理由: 雖然唱同一句導致音色接近，但呼吸與動作特徵不一致，這是真人。")
    elif mfcc_sim < 0.5:
        print("結果 -> [ ✅ 真實二重唱 ]")
        print("理由: 音色本質不同 (例如男女聲)。")
    else:
        print("結果 -> [ 模糊地帶 ]")
        print("請檢查是否為背景底噪干擾。")
    print("="*45 + "\n")

if __name__ == "__main__":
    main()