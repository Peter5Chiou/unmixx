import librosa
import numpy as np
import torch
import torch.nn.functional as F
import argparse

def get_mfcc_fingerprint(path, sr=24000, n_mfcc=20):
    # 1. 載入音訊
    y, _ = librosa.load(path, sr=sr)
    
    # 2. 簡單 VAD: 只取有聲音的部分
    yt, _ = librosa.effects.trim(y, top_db=20)
    
    # 3. 提取 MFCC
    # n_fft 和 hop_length 建議與你模型處理時一致
    mfcc = librosa.feature.mfcc(y=yt, sr=sr, n_mfcc=n_mfcc, n_fft=1024, hop_length=256)
    
    # 4. 取平均值作為這段聲音的「音色指紋」
    # 我們通常捨棄第 0 階 (代表直流能量)，從第 1 階開始取，這樣對音量不敏感
    mfcc_mean = np.mean(mfcc[1:, :], axis=1)
    
    # 歸一化 (Standardization) 讓特徵更穩定
    mfcc_mean = (mfcc_mean - np.mean(mfcc_mean)) / (np.std(mfcc_mean) + 1e-6)
    
    return torch.from_numpy(mfcc_mean).float()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file1")
    parser.add_argument("file2")
    args = parser.parse_args()

    # 取得音色指紋
    fp1 = get_mfcc_fingerprint(args.file1)
    fp2 = get_mfcc_fingerprint(args.file2)

    # 計算餘弦相似度
    similarity = F.cosine_similarity(fp1.unsqueeze(0), fp2.unsqueeze(0)).item()

    print("\n" + "="*35)
    print(f"MFCC 音色相似度: {similarity:.4f}")
    print("="*35)
    
    if similarity > 0.85:
        print("結論：音色高度一致 (極可能是同一人)")
    elif similarity > 0.65:
        print("結論：音色相近 (同性別或唱法接近)")
    else:
        print("結論：音色差異明顯 (不同人)")

if __name__ == "__main__":
    main()