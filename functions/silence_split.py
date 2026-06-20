import numpy as np
import webrtcvad
import librosa

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


# Code for silence split using the method in "Weakly Informed Source Separation, K. Shulcze-Forster, WASPAA 2019."
def magspec_vad(wav, n_fft=1024, hop_length=256, threshold=0.1):
    stft = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, center=False)
    mag, phase = librosa.magphase(stft)
    
    # Local Normalization: Instead of using the global maximum of the entire song,
    # we use a sliding window to find the local maximum magnitude.
    # This prevents quiet singing parts from being misclassified as silence
    # when there are very loud parts elsewhere in the audio.
    
    # 1. Find the maximum magnitude in each frame
    frame_max = np.max(mag, axis=0)
    
    # 2. Find the local maximum of these frame-maxes using a sliding window
    window_size = 500  # Approx 3-5 seconds depending on sample rate
    if len(frame_max) <= window_size:
        local_max_bin = np.max(frame_max) if len(frame_max) > 0 else 1.0
        local_max_bin = np.full_like(frame_max, local_max_bin)
    else:
        # Efficient sliding window max using numpy stride tricks
        padded = np.pad(frame_max, (window_size // 2, window_size // 2), mode='edge')
        shape = (len(frame_max), window_size)
        strides = (padded.strides[0], padded.strides[0])
        windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
        local_max_bin = np.max(windows, axis=1)
    
    # 3. Normalize the magnitude matrix by the local maximum bin value
    mag = mag / (local_max_bin + 1e-8)
    
    # 4. Sum across frequency bins to get energy per frame
    mag_sum = mag.sum(0)
    mag_sum[mag_sum >= threshold] = 1
    mag_sum[mag_sum != 1] = 0

    diff = np.diff(np.pad(mag_sum, (1, 1)))

    seg_start_pos_list = np.where(diff == 1)[0]
    segment_end_pos_list = np.where(diff == -1)[0]

    return seg_start_pos_list * hop_length, segment_end_pos_list * hop_length
