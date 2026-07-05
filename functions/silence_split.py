import numpy as np
import webrtcvad
import librosa
import scipy.ndimage as ndimage

# Code for silence split using py-webrtcvad (https://github.com/wiseman/py-webrtcvad)
def webrtc_vad(wav, orig_sr, vad_mode=3, frame_size=0.03):
    vad = webrtcvad.Vad(vad_mode)
    if orig_sr not in [8000, 16000, 32000, 48000]:
        wav_resampled = librosa.resample(wav, orig_sr=orig_sr, target_sr=16000)
        target_sr = 16000
    else:
        wav_resampled = wav
        target_sr = orig_sr

    voice_activities = []
    for i in range(wav_resampled.shape[0] // int(16000 * frame_size)):
        voice_activities.append(
            vad.is_speech(
                wav_resampled[
                    i * int(16000 * frame_size) : (i + 1) * int(16000 * frame_size)
                ]
                .astype(np.float16)
                .tobytes(),
                target_sr,
            )
        )
    voice_activities = np.array(voice_activities) * 1

    diff = np.diff(np.pad(voice_activities, (1, 1)))

    seg_start_pos_list = np.where(diff == 1)[0]
    segment_end_pos_list = np.where(diff == -1)[0]

    return seg_start_pos_list * int(frame_size * orig_sr), segment_end_pos_list * int(
        frame_size * orig_sr
    )

def magspec_vad_org(wav, n_fft=1024, hop_length=256):
    stft = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, center=False)
    mag, phase = librosa.magphase(stft)
    mag = mag / np.max(mag)
    mag_sum = mag.sum(0)
    mag_sum[mag_sum >= 0.1] = 1
    mag_sum[mag_sum != 1] = 0

    diff = np.diff(np.pad(mag_sum, (1, 1)))

    seg_start_pos_list = np.where(diff == 1)[0]
    segment_end_pos_list = np.where(diff == -1)[0]

    return seg_start_pos_list * hop_length, segment_end_pos_list * hop_length

def magspec_vad(wav, sr, n_fft=1024, hop_length=256):
    # 修正 UserWarning
    if len(wav) < n_fft:
        n_fft = len(wav)
        
    stft = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, center=False)
    mag, _ = librosa.magphase(stft)
    
    # --- 1. 還原 org 版能量計算方式 ---
    # 先對原始頻譜歸一化，再加總。這會產生較大的數值，使 0.1 門檻變寬鬆
    mag_norm = mag / (np.max(mag) + 1e-8)
    mag_sum = mag_norm.sum(0)
    
    # --- 2. 建立初步 Mask 並進行「形態學平滑」 ---
    # 這步是關鍵！它能防止產生 269 個碎片的現象
    binary_mask = (mag_sum >= 0.1).astype(np.int8)
    
    # 合併微小間隙：將 0.5 秒內的靜音視為「有聲」，避免斷句
    gap_frames = int(0.5 * sr / hop_length)
    if gap_frames > 0:
        # 使用 binary_closing 填補空洞
        binary_mask = ndimage.binary_closing(binary_mask, structure=np.ones(gap_frames)).astype(np.int8)
    
    # 移除太短的雜訊：將 0.2 秒內的聲音視為「無聲」
    min_speech_frames = int(0.2 * sr / hop_length)
    if min_speech_frames > 0:
        binary_mask = ndimage.binary_opening(binary_mask, structure=np.ones(min_speech_frames)).astype(np.int8)

    # 取得平滑後的「初始乾淨段落」
    diff = np.diff(np.pad(binary_mask, (1, 1)))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    initial_segments = list(zip(starts, ends))
    
    print(f"\n{'='*60}")
    print(f"[VAD Debug] 初始段落清理完成，共抓到 {len(initial_segments)} 段。")
    # (此時的數量應該會接近你原本看到的 21 段)

    # --- 3. 處理 Sub-chunk 切分 (> 9秒) ---
    min_len = int(3 * sr / hop_length) 
    max_len = int(6.0 * sr / hop_length)
    
    final_segments = []
    
# 遍歷初步偵測到的每一個有聲段落 (s=起點, e=終點)
    for i, (s, e) in enumerate(initial_segments):
        
        # 計算目前這段段落的總長度 (單位是 frames，而非秒數)
        chunk_len = e - s 
        
        # 如果長度小於等於最大容許長度 (max_len=9秒)
        if chunk_len <= max_len:
            # 安全過關！直接把這一段加入最終結果，不用切
            final_segments.append((s, e))
            
        # 如果長度超過最大容許長度 (> 9秒)，就要開始切分
        else:
            # 設定一個「切割游標」curr_s，記錄現在切到哪裡了，初始值從原始起點 s 開始
            curr_s = s
            
            # 只要從「游標」到「終點」的長度還大於 9 秒，就繼續切
            while (e - curr_s) > max_len:
                
                # 計算還剩下多少長度要處理
                remaining_len = e - curr_s
                
                # 【保護機制】如果剩下 9~13 秒 (max_len=9 + min_len=4)
                # 代表如果硬切掉 9 秒，最後的尾巴會短於 4 秒 (太短可能讓模型崩潰)
                if remaining_len < (max_len + min_len):
                    # 如果剩下 9~13 秒，直接從中間對半切，確保兩邊都有 4.5~6.5 秒
                    split_point = curr_s + (remaining_len // 2)
                    final_segments.append((curr_s, split_point))
                    curr_s = split_point                    
                    break
                
                # ====================================================
                # 若跑到這裡，代表剩餘長度大於 13 秒，可以安心切一刀。
                # 我們不直接切在 9 秒處，而是在 4~9 秒之間找一個「最安靜」的地方切。
                # ====================================================
                
                # 定義搜尋範圍的起點：從游標往後算 4 秒 (確保切出來的片段至少 4 秒)
                search_start = curr_s + min_len
                
                # 定義搜尋範圍的終點：從游標往後算 9 秒 (確保片段不超過 9 秒)
                search_end = curr_s + max_len
                
                # 把這 4~9 秒之間的能量陣列 (mag_sum) 抓出來
                search_region = mag_sum[search_start : search_end]
                
                # 使用 np.argmin 找出這個區間內「能量最低 (數字最小)」的索引值
                # 再加上 search_start 換算回原始音檔的絕對位置，這就是最完美的「切割點」
                split_point = search_start + np.argmin(search_region)
                
                # 將切出來的這段 [游標起點, 切割點] 加入最終結果
                final_segments.append((curr_s, split_point))
                
                # 把游標移到剛剛切下的這刀，當作下一次切分的新起點
                curr_s = split_point
            
            # 當 while 迴圈結束 (可能是切到剩 <9秒，或是觸發了 break 跳出)
            # 把最後剩下來的那段尾巴 [最後的游標位置, 原始終點 e] 加入最終結果
            final_segments.append((curr_s, e))

    # 最終排序確保時序
    final_segments.sort(key=lambda x: x[0]) 

    # 4. 轉換為樣本點索引
    final_starts = np.array([s for s, e in final_segments]) * hop_length
    final_ends = np.array([e for s, e in final_segments]) * hop_length
    
    return final_starts, final_ends