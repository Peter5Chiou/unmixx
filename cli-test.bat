.venv\scripts\python.exe inference.py ^
  --conf_path ckpt/conf.yml ^
  --ckpt_path ckpt/best.ckpt ^
  --audio_path moodtest.mp3 ^
  --output_dir separated_audio ^
  --spectral_features spectral_centroid