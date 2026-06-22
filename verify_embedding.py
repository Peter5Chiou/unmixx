import os
os.environ["SB_DISABLE_LAZY_IMPORT"] = "1"

import torch
import torchaudio
import torch.nn.functional as F
from speechbrain.pretrained import EncoderClassifier
import argparse
import sys

def get_embedding(waveform, model, device):
    # Ensure waveform is [1, time]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    waveform = waveform.to(device)
    
    with torch.no_grad():
        # encode_batch returns [batch, 1, embedding_dim]
        embeddings = model.encode_batch(waveform)
        embedding = embeddings.squeeze() # [embedding_dim]
    
    return embedding

def main():
    parser = argparse.ArgumentParser(description="Verify Deep Embedding similarity between two audio files.")
    parser.add_argument("file1", help="Path to the first audio file")
    parser.add_argument("file2", help="Path to the second audio file")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device selection")
    args = parser.parse_args()

    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Device] Using {device}")

    # Load ECAPA-TDNN model
    print("[Model] Loading spkrec-ecapa-voxceleb...")
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)},
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )
    model.eval()

    # Target sample rate for ECAPA-TDNN is 16kHz
    target_sr = 16000

    def load_and_preprocess(path):
        print(f"[Audio] Loading: {path}")
        waveform, sr = torchaudio.load(path)
        
        # 1. 轉單聲道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 2. 重採樣到 16kHz
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
        
        # 3. 【新增】簡單 VAD：移除靜音 (抹平紅箭頭邏輯的變體)
        # 只保留振幅大於 0.02 的部分
        mask = torch.abs(waveform) > 0.02
        if torch.sum(mask) > 0:
            # 找出所有有聲的索引
            indices = torch.nonzero(mask[0])
            start_idx = indices[0].item()
            end_idx = indices[-1].item()
            waveform = waveform[:, start_idx:end_idx]
        
        # 4. 【新增】音量歸一化 (Normalization)
        # 讓最大值等於 0.9
        max_amp = torch.max(torch.abs(waveform))
        if max_amp > 0:
            waveform = waveform * (0.9 / max_amp)
            
        return waveform

    try:
        wave1 = load_and_preprocess(args.file1)
        wave2 = load_and_preprocess(args.file2)
    except Exception as e:
        print(f"Error loading audio files: {e}")
        sys.exit(1)

    # Extract embeddings
    print("[Process] Extracting embeddings...")
    emb1 = get_embedding(wave1, model, device)
    emb2 = get_embedding(wave2, model, device)
    print(f"向量1模長: {torch.norm(emb1).item():.4f}")
    print(f"向量2模長: {torch.norm(emb2).item():.4f}")
    # Calculate cosine similarity
    # emb1, emb2 are [embedding_dim]
    similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

    print("\n" + "="*30)
    print(f"Result Similarity: {similarity:.4f}")
    print("="*30)
    
    if similarity > 0.8:
        print("Conclusion: The voices are very similar (likely the same person).")
    elif similarity > 0.5:
        print("Conclusion: The voices have some similarity but are likely different people.")
    else:
        print("Conclusion: The voices are very different.")

if __name__ == "__main__":
    main()
