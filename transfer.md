# Project Transfer Summary: UNMIXX Source Separation Optimization

## 1. Initial Objectives
- **Stereo to Mono**: Ensure all input audio is converted to mono to match model requirements.
- **VRAM Optimization**: Enable the processing of full-length songs (instead of just short clips) within a 12G VRAM limit.

## 2. Implemented Changes in `inference.py`

### Audio Pre-processing
- Modified `prepare_audio_tensor` to detect stereo files and convert them to mono by averaging channels.

### Chunk-based Inference (VRAM Fix)
- Integrated `asteroid.dsp.overlap_add.LambdaOverlapAdd` to process audio in chunks.
- Implemented a `ModelWrapper` to handle the model's output format and ensure compatibility with the overlap-add mechanism.

### Solving the Permutation Problem (Source Swapping)
- **The Issue**: Encountered "source swapping" where the main vocal and harmony would flip identities between chunks.
- **First Attempt**: Increased `seq_dur` from 2s to 4s to provide more context and stronger correlation.
- **Final Solution**: Replaced the basic `LambdaOverlapAdd` with `LambdaOverlapAdd_Chunkwise_SpectralFeatures`.
    - **Mechanism**: Uses Voice Activity Detection (VAD) and MFCC spectral features to align sources across chunks via cosine similarity.
    - **Configuration**: `spectral_features="mfcc"`, `vad_method="spec"`, `sr=24000`.

## 3. VRAM & Performance Observations
- **`seq_dur = 2s`**: Very low VRAM usage (~1G), but higher risk of source swapping.
- **`seq_dur = 4s`**: Higher VRAM usage (~14G), exceeding the 12G limit on some hardware, but more stable source assignment.
- **Trade-off**: The optimal `seq_dur` needs to be balanced between VRAM capacity and the stability of the spectral alignment.

## 4. Inference Quality Optimization (Latest)

### VAD & Spectral Feature Enhancements
- **Configurable VAD Threshold**: Modified `functions/silence_split.py` and `functions/overpaladd_chunk_spec_feat.py` to allow a configurable `vad_threshold` (default increased to `0.2`). This reduces noise leakage in silent parts.
- **Enhanced Spectral Fingerprint**: Increased `n_mfcc` from `20` to `40` in `functions/overpaladd_chunk_spec_feat.py` to improve source identification and reduce swapping.
- **VAD Method Switching**: Updated `inference.py` to allow switching between `"spec"` and `"webrtc"` VAD methods.

### VRAM Safety Cap (OOM Fix)
- **Segment Length Capping**: Implemented a safety cap in `functions/overpaladd_chunk_spec_feat.py`. If a VAD segment is too long (exceeds `2 * window_size`), it is now split into smaller chunks of `window_size` to prevent VRAM OOM.

## 5. Current State & Pending Issues

### Current State
- The system can process full songs with a VRAM safety cap.
- Stereo inputs are handled automatically.
- VAD and MFCC parameters are now configurable in `inference.py`.

### Pending Issue: Permutation Problem in Long Segments
- **The Problem**: After implementing the VRAM safety cap (splitting long segments into chunks), the "source swapping" problem returned.
- **Root Cause**: The reordering logic was only applied to the final chunk of a VAD segment, while the preceding chunks were assigned to the output tensor without reordering.
- **Required Fix**: Refactor `ola_forward` in `functions/overpaladd_chunk_spec_feat.py` to perform reordering on **every single chunk** processed, ensuring identity consistency across the entire song.
- **Progress**: `_extract_spectral_features` helper method has been implemented, but the `ola_forward` refactor is incomplete due to tool failures.
