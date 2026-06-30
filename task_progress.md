# Task: Add manual chunks support to inference.py

- [ ] Analyze requirements: Add --chunks_json parameter to inference.py
- [ ] Modify LambdaOverlapAdd_Chunkwise_SpectralFeatures to accept chunks parameter
- [ ] Modify inference.py to parse chunks.json and pass to the overlap-add class
- [ ] Ensure fallback to automatic VAD when chunks_json not provided
- [ ] Test the implementation

## Implementation Notes
- chunks.json contains: chunks_id, filtered_starts, filtered_ends, swapped (in seconds)
- When chunks_json specified: use those time ranges instead of VAD
- When not specified: use existing automatic VAD-based chunking