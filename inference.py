#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
from pathlib import Path
from pprint import pprint
import numpy as np 
import soundfile as sf
import scipy.ndimage as ndimage
import torch
import torchaudio
import torchaudio.transforms as T
import look2hear.models
import torch.nn.functional as F
from asteroid.dsp.overlap_add import LambdaOverlapAdd
from functions.overpaladd_chunk_spec_feat import LambdaOverlapAdd_Chunkwise_SpectralFeatures

# (옵션) salience 시각화 유틸이 필요하면 주석 해제
# from basic_pitch_torch.inference import predict
# import numpy as np
# import matplotlib.pyplot as plt

# =========================
# Utils
# =========================
def load_yaml(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def derive_model_name(cfg: dict, ckpt_path: str):
    # 우선 ckpt 파일명 기준
    stem = Path(ckpt_path).stem
    # config에 audionet_name이 있으면 보조적으로 붙임
    try:
        net_name = cfg["audionet"]["audionet_name"]
        model_name = f"{net_name}-{stem}"
    except Exception:
        model_name = stem
    return model_name

def pick_model_class(cfg: dict):
    net_name = cfg["audionet"]["audionet_name"]
    if not hasattr(look2hear.models, net_name):
        raise AttributeError(f"look2hear.models에 '{net_name}' 클래스가 없습니다.")
    return getattr(look2hear.models, net_name)

def build_model(cfg: dict):
    sr = cfg["datamodule"]["data_config"]["sample_rate"]
    aud_cfg = cfg["audionet"]["audionet_config"]
    ModelClass = pick_model_class(cfg)
    model = ModelClass(sample_rate=sr, **aud_cfg)
    return model

def load_state_dict_safely(model: torch.nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Lightning 스타일 또는 pure state_dict 둘 다 처리
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        # 바로 state_dict로 저장된 형태
        state_dict = ckpt

    try:
        # 바로 로드가 되면 끝
        model.load_state_dict(state_dict, strict=True)
        return
    except Exception:
        # prefix 정리 (예: "audio_model.")
        converted = {}
        for k, v in state_dict.items():
            if k.startswith("audio_model."):
                converted[k[len("audio_model."):]] = v
            else:
                converted[k] = v
        model.load_state_dict(converted, strict=True)

def prepare_audio_tensor(audio_path: str, target_sr: int, device: torch.device):
    waveform, original_sr = torchaudio.load(audio_path)  # [C, T]
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    if original_sr != target_sr:
        resampler = T.Resample(orig_freq=original_sr, new_freq=target_sr)
        waveform = resampler(waveform)
    # [B, C, T] 로
    audio_input = waveform.unsqueeze(0).to(device)
    return audio_input, waveform, target_sr  # audio_input: [1, C, T], waveform: [C, T]

def normalize_outputs(ests):
    """
    모델 반환 타입이 tuple/dict 등 다양한 경우를 평탄화해 최종 [B, S, T] 또는 [S, T] 텐서를 반환.
    """
    if isinstance(ests, tuple):
        # 자주 쓰는 규칙 몇 가지
        if len(ests) == 2:
            ests = ests[1]
        elif len(ests) in (3, 5):
            ests = ests[0]
        else:
            ests = ests[0]

    if isinstance(ests, dict):
        # 가능한 키 이름들 통일
        for k in ["output_final", "audio_out_final", "output", "audio_out"]:
            if k in ests:
                return ests[k], ests.get("output_original", ests.get("audio_out_original", ests[k]))
        # 못 찾으면 첫 항목
        first = next(iter(ests.values()))
        return first, first

    # 텐서인 경우
    return ests, ests

def ensure_2d_channels_first(x: torch.Tensor):
    """
    [T] -> [1, T], [C, T]는 그대로
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return x

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x, **kwargs):
        # UNMIXX returns (reconstructed, mag1, phase1, mag2, phase2)
        # reconstructed shape: [B * nch * S, T]
        # Ensure istest=True is passed to the underlying UNMIXX model
        kwargs['istest'] = True
        
        # LambdaOverlapAdd passes chunks as [B_new, T]
        if x.dim() == 2:
            x = x.unsqueeze(1) # [B_new, 1, T]
            
        out = self.model(x, **kwargs)
        reconstructed = out[0]
        
        # reconstructed shape from UNMIXX: [B_new * nch * S, T]
        # We need to return [B_new, S, T]
        if reconstructed.dim() == 3:
            return reconstructed # [B, S, T]
            
        B_new, C, T_chunk = x.shape
        S = reconstructed.shape[0] // (B_new * C)
        return reconstructed.view(B_new, C, S, T_chunk).squeeze(1) # [B_new, S, T_chunk]

def preprocess_clean_wave_gpu(audio_tensor, sr, threshold=0.02, min_dur_sec=0.1, gap_merge_sec=0.2):
    """
    修正長度對齊問題後的 GPU 清洗函式
    """
    device = audio_tensor.device
    T = audio_tensor.shape[-1]  # 紀錄原始長度
    min_samples = int(sr * min_dur_sec)
    gap_samples = int(sr * gap_merge_sec)
    
    # --- 步驟 1：包絡偵測 ---
    k1 = int(sr * 0.02) 
    if k1 % 2 == 0: k1 += 1
    abs_audio = torch.abs(audio_tensor)
    envelope = F.max_pool1d(abs_audio, kernel_size=k1, stride=1, padding=k1 // 2)
    # 強制對齊長度
    envelope = envelope[..., :T]
    
    # --- 步驟 2：門檻判定 ---
    mask = (envelope > threshold).float() 
    
    # --- 步驟 3：合併微小間隙 ---
    if gap_samples > 0:
        k2 = gap_samples
        if k2 % 2 == 0: k2 += 1
        mask = F.max_pool1d(mask, kernel_size=k2, stride=1, padding=k2 // 2)
        # 強制對齊長度
        mask = mask[..., :T]

    mask = mask.view(-1)
    
    # --- 步驟 4：偵測段落 ---
    m = torch.cat([torch.tensor([0.0], device=device), mask, torch.tensor([0.0], device=device)])
    diff = m[1:] - m[:-1]
    
    starts = (diff > 0).nonzero(as_tuple=True)[0]
    ends = (diff < 0).nonzero(as_tuple=True)[0]
    lengths = ends - starts
    
    # --- 步驟 5：長度過濾 ---
    final_mask = torch.zeros_like(mask)
    valid_segments = (lengths >= min_samples).nonzero(as_tuple=True)[0]
    
    for idx in valid_segments:
        final_mask[starts[idx]:ends[idx]] = 1.0
        
    print(f"  [VAD Check] 原始片段: {len(starts)} | 長度達標: {len(valid_segments)}")
    
    # --- 步驟 6：應用 Mask ---
    # 使用 view_as 確保形狀完全一致
    return audio_tensor * final_mask.view(1, 1, -1)
    
# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Separate sources with Look2Hear TIGER-like model.")
    parser.add_argument("--conf_path", required=True, help="YAML config path.")
    parser.add_argument("--ckpt_path", required=True, help="Checkpoint path (.ckpt/.pth).")
    parser.add_argument("--audio_path", required=True, help="Input audio path (wav).")
    parser.add_argument("--output_dir", default="separated_audio", help="Output directory.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device selection.")
    parser.add_argument("--target_sr", type=int, default=None, help="Override target sample rate. (optional)")
    parser.add_argument("--spectral_features", default="mfcc", choices=["mfcc", "spectral_centroid", "deep_embedding"], help="Features for source reordering.")
    parser.add_argument("--chunks_path", default=None, help="Path to chunks.json for manual audio segmentation.")
    args = parser.parse_args()

    if args.chunks_path:
        if not Path(args.chunks_path).exists():
            print(f"\n 指定切割點檔案不存在: {args.chunks_path}")
            return

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Device] {device}")

    # Config
    config = load_yaml(args.conf_path)
    print("[Config] loaded.")
    # pprint(config)

    # Model
    model = build_model(config)
    load_state_dict_safely(model, args.ckpt_path)
    model.to(device).eval()
    
    # Wrap model for LambdaOverlapAdd
    wrapped_model = ModelWrapper(model)

    # Target SR
    cfg_sr = config["datamodule"]["data_config"]["sample_rate"]
    target_sr = args.target_sr if args.target_sr is not None else int(cfg_sr)
    print(f"[SampleRate] target_sr={target_sr}")

    # Audio
    print(f"[Audio] loading: {args.audio_path}")
    audio_input, waveform_cpu, sr = prepare_audio_tensor(args.audio_path, target_sr, device)
    print(f"[Audio] shape (B,C,T)={tuple(audio_input.shape)}, sr={sr}")

   # ======================================================
    # 【GPU 加速清洗】
    # ======================================================
    print(f"[Pre-process] GPU 加速清洗雜訊中 (threshold=0.02, min_dur_sec=0.1)...")
    
    audio_input = preprocess_clean_wave_gpu(
        audio_input, 
        target_sr, 
        threshold=0.02,     # 能量門檻，維持不變
        min_dur_sec=0.5,    # 提高到 0.5 秒，這能砍掉絕大多數吸氣聲與雜訊
        gap_merge_sec=0.1   # 把 0.2 秒內的聲音連起來，保護歌聲不被切碎  
    )
    # 如果你還是想檢查結果，才轉回 CPU 存檔 (這步會耗一點點時間，檢查完可以註解掉)
    debug_path = "debug_cleaned_input.wav"
    debug_wav = audio_input.detach().cpu().squeeze().numpy()
    sf.write(debug_path, debug_wav, target_sr)
    print(f"[Pre-process] 抹平測試檔存於: {debug_path}")
    
    #import sys; sys.exit() 

   
    
    # Forward with chunking
    with torch.no_grad():
        # Use LambdaOverlapAdd for long audio processing to avoid VRAM explosion
        # Increased seq_dur to 4 seconds to improve source consistency (reduce permutation problem)
        # hop_size = 1 second (75% overlap)
        seq_dur = 4
        window_size = int(seq_dur * target_sr)
        hop_size = window_size // 8
        
        # Determine num_sources from model config or a small forward pass
        num_sources = model.num_output if hasattr(model, 'num_output') else 2
        
        # Inference parameters for cleaner separation
        vad_method = "spec"   
        
        # Output paths
        model_name = derive_model_name(config, args.ckpt_path)
        base_dir = Path(args.output_dir) / model_name / Path(args.audio_path).stem
        base_dir.mkdir(parents=True, exist_ok=True)

        continuous_model = LambdaOverlapAdd_Chunkwise_SpectralFeatures(
            nnet=wrapped_model,
            n_src=num_sources,
            window_size=window_size,
            hop_size=hop_size,
            window=None,
            reorder_chunks=True,
            enable_grad=False,
            device=device,
            sr=target_sr,
            vad_method=vad_method,
            spectral_features=args.spectral_features,
            output_dir=str(base_dir),
            chunks_path=args.chunks_path,
        ).to(device)
        
        outs = continuous_model(audio_input)

    ests_speech, ests_speech_original = normalize_outputs(outs)
    # [B, S, T] 또는 [S, T]
    if ests_speech.dim() == 2:
        # [S, T] -> [1, S, T]
        ests_speech = ests_speech.unsqueeze(0)
    if ests_speech_original.dim() == 2:
        ests_speech_original = ests_speech_original.unsqueeze(0)

    # 첫 배치만 사용
    ests_speech = ests_speech[0].cpu()             # [S, T]
    ests_speech_original = ests_speech_original[0].cpu()  # [S, T]
    num_speakers = ests_speech.shape[0]
    print(f"[Separation] detected {num_speakers} streams")

    # Output paths (already created above)

    # Save estimates
    for i in range(num_speakers):
        # torchaudio.save expects [C, T]
        est = ensure_2d_channels_first(ests_speech[i])
        est_org = ensure_2d_channels_first(ests_speech_original[i])
        out_path = base_dir / f"spk{i+1}.wav"
        print(f"[Save] {out_path}")
        torchaudio.save(str(out_path), est, sr)

    # Save mixture, too
    mix_out = base_dir / "mixture.wav"
    print(f"[Save] {mix_out}")
    torchaudio.save(str(mix_out), waveform_cpu, sr)

    print("[Done] All files saved under:", base_dir)
    with open("outputdir.txt", "w") as f:
        f.write(str(base_dir))

if __name__ == "__main__":
    main()
