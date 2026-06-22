import os
import json
import numpy as np
import soundfile as sf
import sounddevice as sd
import customtkinter as ctk
from tkinter import Canvas, filedialog, messagebox
import time
import threading

class WaveformSequencePlayer(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Look2Hear - Pro Waveform Editor & Stitcher")
        self.after(0, lambda: self.state('zoomed'))

        # 顏色與樣式設定
        self.WAVE_COLOR = "#3B9969"  
        self.BG_COLOR = "#1A1A1A"    
        self.SWAP_BG = "#3D2B1F"     
        self.PLAYHEAD_COLOR = "#3498db" 
        self.BTN_BLUE = "#1f538d"    

        # 數據與狀態變數
        self.sr = 24000
        self.audio_data = [None, None]
        self.chunks = []
        self.duration = 0
        self.folder_path = ""
        
        self.is_playing = False
        self.current_playing_chunk_idx = None
        self.is_playing_top_lane = True
        
        self.play_real_start_time = 0
        self.play_start_sec = 0
        self.play_end_sec = 0
        self.playhead_line = None

        self.setup_ui()
        self.check_default_output_dir()

    def format_time(self, seconds):
        """將秒數格式化為 mm:ss"""
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    def setup_ui(self):
        self.header = ctk.CTkLabel(self, text="wave player", font=("Arial", 32, "bold"), text_color="red")
        self.header.pack(pady=5)

        # 頂部操作欄
        self.top_bar = ctk.CTkFrame(self)
        self.top_bar.pack(fill="x", padx=20, pady=5)
        
        # 左側：載入按鈕
        ctk.CTkButton(self.top_bar, text="載入工程資料夾", width=120, fg_color=self.BTN_BLUE, text_color="white", hover=False, command=self.manual_load_folder).pack(side="left", padx=10)
        
        # 中間偏左：狀態資訊
        self.label_info = ctk.CTkLabel(self.top_bar, text="等待載入...", text_color="gray")
        self.label_info.pack(side="left", padx=10)

        # --- 【核心修改：時間顯示器】 ---
        # 放置在中央，使用等寬字體避免跳動
        self.label_timer = ctk.CTkLabel(
            self.top_bar, 
            text="00:00 / 00:00", 
            font=("Consolas", 20, "bold"), 
            text_color="#ECf0F1"
        )
        self.label_timer.pack(side="left", expand=True)
        # ------------------------------

        # 右側按鈕組
        ctk.CTkButton(self.top_bar, text="匯出最終音軌 (Export)", width=150, fg_color=self.BTN_BLUE, text_color="white", hover=False, command=self.export_final_wav).pack(side="right", padx=10)
        ctk.CTkButton(self.top_bar, text="儲存修改 (JSON)", width=120, fg_color=self.BTN_BLUE, text_color="white", hover=False, command=self.save_json).pack(side="right", padx=10)

        self.canvas_frame = ctk.CTkFrame(self, fg_color=self.BG_COLOR)
        self.canvas_frame.pack(fill="both", expand=True, padx=20, pady=20)

        self.canvas = Canvas(self.canvas_frame, bg=self.BG_COLOR, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        self.canvas.bind("<Button-1>", self.handle_left_click)
        self.canvas.bind("<Button-3>", self.handle_right_click)
        self.canvas.bind("<Button-2>", self.handle_right_click)

    def check_default_output_dir(self):
        txt_file = "outputdir.txt"
        if os.path.exists(txt_file):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f: path = f.read().strip()
                if os.path.isdir(path): self.perform_load(path)
            except: pass

    def manual_load_folder(self):
        path = filedialog.askdirectory()
        if path: self.perform_load(path)

    def perform_load(self, path):
        try:
            json_p, s1_p, s2_p = [os.path.join(path, f) for f in ["chunks.json", "spk1.wav", "spk2.wav"]]
            if not all(os.path.exists(f) for f in [json_p, s1_p, s2_p]): return
            self.folder_path = path
            with open(json_p, 'r') as f: self.chunks = json.load(f)
            s1, sr = sf.read(s1_p); s2, _ = sf.read(s2_p)
            self.audio_data = [s1, s2]; self.sr = sr; self.duration = len(s1) / sr
            
            self.label_info.configure(text=f"載入成功")
            # 初始化時間顯示
            self.label_timer.configure(text=f"00:00 / {self.format_time(self.duration)}")
            
            self.after(200, self.draw_all)
        except Exception as e: messagebox.showerror("錯誤", f"載入失敗: {e}")

    def draw_all(self):
        self.canvas.delete("all")
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w <= 1: return
        mid_y = h // 2
        self.canvas.create_line(0, mid_y, w, mid_y, fill="white", width=1)
        for idx, chunk in enumerate(self.chunks):
            x_s, x_e = (chunk['filtered_starts'] / self.duration) * w, (chunk['filtered_ends'] / self.duration) * w
            if chunk.get('swapped', False): self.canvas.create_rectangle(x_s, 0, x_e, h, fill=self.SWAP_BG, outline="")
            self.canvas.create_line(x_s, 0, x_s, h, fill="#444444", dash=(4, 4))
            self.canvas.create_text(x_s + 5, 10, text=f"CK {idx+1}", fill="white", anchor="nw", font=("Arial", 9))
            top_idx, btm_idx = (1, 0) if chunk.get('swapped', False) else (0, 1)
            self.draw_waveform_segment(x_s, x_e, top_idx, 0, mid_y)
            self.draw_waveform_segment(x_s, x_e, btm_idx, mid_y, h)
        self.playhead_line = self.canvas.create_line(0, 0, 0, h, fill=self.PLAYHEAD_COLOR, width=2)

    def draw_waveform_segment(self, x_s, x_e, audio_idx, top, bottom):
        w_pixels = int(x_e - x_s)
        if w_pixels <= 0: return
        mid_y, max_h = (top + bottom) // 2, (bottom - top) // 2.5
        data = self.audio_data[audio_idx]
        start_samp, end_samp = int((x_s / self.canvas.winfo_width()) * len(data)), int((x_e / self.canvas.winfo_width()) * len(data))
        seg_data = data[start_samp:end_samp]
        step = len(seg_data) // w_pixels if len(seg_data) > w_pixels else 1
        for i in range(w_pixels):
            sample = seg_data[i*step : (i+1)*step]
            if len(sample) > 0:
                amp = np.max(np.abs(sample)) * max_h
                self.canvas.create_line(x_s + i, mid_y - amp, x_s + i, mid_y + amp, fill=self.WAVE_COLOR)

    def handle_left_click(self, event):
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        t = (event.x / w) * self.duration
        clicked_idx, chunk = next(((i, c) for i, c in enumerate(self.chunks) if c['filtered_starts'] <= t <= c['filtered_ends']), (None, None))
        
        if chunk is None:
            if self.is_playing: self.stop_playback()
            return

        is_click_top = event.y < (h // 2)
        if self.is_playing and clicked_idx == self.current_playing_chunk_idx and is_click_top == self.is_playing_top_lane:
            self.stop_playback()
            return

        self.is_playing_top_lane = is_click_top 
        self.start_chunk_playback(clicked_idx)

    def start_chunk_playback(self, chunk_idx):
        sd.stop()
        self.current_playing_chunk_idx = chunk_idx
        chunk = self.chunks[chunk_idx]
        is_swapped = chunk.get('swapped', False)
        audio_idx = (1 if is_swapped else 0) if self.is_playing_top_lane else (0 if is_swapped else 1)
        self.play_start_sec = chunk['filtered_starts']
        self.play_end_sec = chunk['filtered_ends']
        data = self.audio_data[audio_idx][int(self.play_start_sec*self.sr):int(self.play_end_sec*self.sr)]
        sd.play(data, self.sr)
        self.play_real_start_time = time.time()
        self.is_playing = True
        self.animate_playhead()

    def stop_playback(self):
        sd.stop()
        self.is_playing = False
        self.current_playing_chunk_idx = None
        # 停止後將時間重置回 0 (或保持在最後位置)
        # self.label_timer.configure(text=f"00:00 / {self.format_time(self.duration)}")

    def animate_playhead(self):
        if not self.is_playing: return
        elapsed = time.time() - self.play_real_start_time
        curr_pos = self.play_start_sec + elapsed
        
        # --- 【修改：更新時間顯示器】 ---
        self.label_timer.configure(text=f"{self.format_time(curr_pos)} / {self.format_time(self.duration)}")
        # ------------------------------

        if curr_pos >= self.play_end_sec:
            next_idx = self.current_playing_chunk_idx + 1
            if next_idx < len(self.chunks):
                self.start_chunk_playback(next_idx)
                return 
            else:
                curr_pos = self.play_end_sec
                self.stop_playback()
            
        w = self.canvas.winfo_width()
        x = (curr_pos / self.duration) * w
        self.canvas.coords(self.playhead_line, x, 0, x, self.canvas.winfo_height())
        if self.is_playing: self.after(20, self.animate_playhead)

    def handle_right_click(self, event):
        w = self.canvas.winfo_width()
        t = (event.x / w) * self.duration
        idx = next((i for i, c in enumerate(self.chunks) if c['filtered_starts'] <= t <= c['filtered_ends']), None)
        if idx is not None:
            self.chunks[idx]['swapped'] = not self.chunks[idx].get('swapped', False)
            self.draw_all()

    def save_json(self):
        if not self.folder_path: return
        with open(os.path.join(self.folder_path, "chunks.json"), 'w') as f: json.dump(self.chunks, f, indent=4)
        messagebox.showinfo("成功", "修改已儲存！")

    def export_final_wav(self):
        if not self.folder_path or self.audio_data[0] is None: return
        if not messagebox.askyesno("匯出", "確定要根據 Swap 標記導出最終音軌嗎？"): return
        self.label_info.configure(text="正在匯出...", text_color="orange")
        def run_stitch():
            try:
                s1_final, s2_final = self.audio_data[0].copy(), self.audio_data[1].copy()
                for item in self.chunks:
                    if item.get("swapped", False):
                        s, e = int(item['filtered_starts']*self.sr), int(item['filtered_ends']*self.sr)
                        s1_final[s:e], s2_final[s:e] = s2_final[s:e].copy(), s1_final[s:e].copy()
                sf.write(os.path.join(self.folder_path, "spk1_final.wav"), s1_final, self.sr)
                sf.write(os.path.join(self.folder_path, "spk2_final.wav"), s2_final, self.sr)
                self.after(0, lambda: messagebox.showinfo("完成", "匯出成功！"))
                self.after(0, lambda: self.label_info.configure(text="匯出成功", text_color="green"))
            except Exception as e: self.after(0, lambda: messagebox.showerror("失敗", str(e)))
        threading.Thread(target=run_stitch).start()

if __name__ == "__main__":
    app = WaveformSequencePlayer()
    app.mainloop()