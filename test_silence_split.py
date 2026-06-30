import numpy as np
import librosa
from functions.silence_split import magspec_vad

def test_magspec_vad():
    sr = 24000
    duration = 30  # 30 seconds total
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a signal: 
    # 0-2s: silence
    # 2-22s: sound (20 seconds long)
    # 22-30s: silence
    wav = np.zeros(int(sr * duration))
    wav[int(2 * sr):int(22 * sr)] = np.sin(2 * np.pi * 440 * t[int(2 * sr):int(22 * sr)])
    
    # Add some slight noise to avoid absolute zeros if needed, 
    # but magspec_vad should handle it.
    wav += np.random.normal(0, 0.001, wav.shape)

    print(f"Testing magspec_vad with a 20s sound segment...")
    starts, ends = magspec_vad(wav, sr=sr)
    
    print(f"Detected segments: {len(starts)}")
    for i, (s, e) in enumerate(zip(starts, ends)):
        duration_sec = (e - s) / sr
        print(f"Segment {i+1}: {s/sr:.2f}s - {e/sr:.2f}s (Duration: {duration_sec:.2f}s)")
        
        if duration_sec > 9.1: # Allow small float margin
            print(f"❌ Segment {i+1} is too long: {duration_sec:.2f}s")
            return False
        if duration_sec < 2.4: # Allow small float margin
            print(f"❌ Segment {i+1} is too short: {duration_sec:.2f}s")
            return False
            
    print("✅ All segments are within the expected range (2.5s - 9s).")
    return True

if __name__ == "__main__":
    if test_magspec_vad():
        print("\nTest PASSED")
    else:
        print("\nTest FAILED")
        exit(1)
