.venv\scripts\python.exe inference.py ^
  --conf_path ckpt/conf.yml ^
  --ckpt_path ckpt/best.ckpt ^
  --audio_path %1 ^
  --output_dir separated_audio ^
  --spectral_features mfcc 
REM  --chunks_path chunks.json