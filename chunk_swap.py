import os
import json
import numpy as np
import soundfile as sf
import sounddevice as sd
import customtkinter as ctk
from tkinter import Canvas, filedialog, messagebox
import time
import threading
import torch
from PIL import Image, ImageDraw, ImageTk

class WaveformSequencePlayer(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Look2Hear - Pro Waveform Editor (Mix at Bottom)")
        self.after(0, lambda: self.state('zoomed'))

        # 顏色設定
        self.WAVE_COLOR = "#3B9969"   # SPK 波形 (綠色)
        self.MIX_COLOR = "#A9D0F5"    # Mixture 波形 (淡藍色)
        self.BG_COLOR = "#1A1A1A"     # 背景色
        self.TEXT_COLOR = "#D3D3D3"   # 文字顏色 (淺灰色)
        self.SWAP_BG = "#3D2B1F"      # Swap 區域背景
        self.PLAYHEAD_COLOR = "#3498db" 
        self.BTN_BLUE = "#1f538d"    

        # 數據變數
        self.sr = 24000
        self.audio_data = [None, None, None] # [Mixture, SPK1, SPK2]
        self.chunks = []
        self.duration = 0
        self.folder_path = ""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.audio_tensors = [None, None, None]
        
        # 縮放與視野變數
        self.view_start_time = 0.0
        self.view_end_time = 0.0
        self.min_view_dur = 0.5
        
        # 播放狀態
        self.is_playing = False
        self.current_playing_chunk_idx = None
        self.current_playing_visual_lane = 0 # 0: SPK1, 1: SPK2, 2: MIX
        
        self.play_real_start_time = 0
        self.play_start_sec = 0
        self.play_end_sec = 0
        self.playhead_line = None

        # Edit mode variables
        self.edit_mode = False
        self.split_chunk_idx = None
        self.split_time = None

        self.setup_ui()
        self.check_default_output_dir()

    def format_time(self, seconds):
        m, s = int(seconds // 60), int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{m:02d}:{s:02d}.{ms:03d}"

    def update_chunk_info(self, idx):
        if idx is None or idx < 0 or idx >= len(self.chunks):
            self.label_chunk_info.configure(text="目前無選取 Chunk")
            return
        chunk = self.chunks[idx]
        chunk_id = chunk.get('chunks_id', idx)
        t_s = chunk['filtered_starts']
        t_e = chunk['filtered_ends']
        dur = t_e - t_s
        info_str = f"選取/播放 Chunk {chunk_id} | 起始: {self.format_time(t_s)} | 結束: {self.format_time(t_e)} | 長度: {self.format_time(dur)}"
        self.label_chunk_info.configure(text=info_str)

    def select_mode(self, mode_name):
        if self.is_playing:
            self.stop_playback()
            
        if mode_name == "分割模式":
            self.edit_mode = True
            self.label_chunk_info.configure(text="進入編輯模式：請左鍵點擊波形選擇分割點")
        else:
            self.edit_mode = False
            self.split_chunk_idx = None
            self.split_time = None
            self.label_chunk_info.configure(text="目前無選取 Chunk")
            
        self.draw_all()

    def update_edit_status(self):
        if self.split_chunk_idx is None or self.split_time is None:
            self.label_chunk_info.configure(text="進入編輯模式：請左鍵點擊波形選擇分割點")
            return
        chunk = self.chunks[self.split_chunk_idx]
        chunk_id = chunk.get('chunks_id', self.split_chunk_idx)
        self.label_chunk_info.configure(
            text=f"[編輯分割] Chunk {chunk_id} | 分割點: {self.format_time(self.split_time)} (←/→鍵微調，Space預覽，Enter確定)"
        )

    def handle_arrow_left(self, event):
        if not self.edit_mode or self.split_chunk_idx is None: return "break"
        chunk = self.chunks[self.split_chunk_idx]
        self.split_time = max(chunk['filtered_starts'] + 0.05, self.split_time - 0.1)
        self.draw_all()
        self.update_edit_status()
        return "break"

    def handle_arrow_right(self, event):
        if not self.edit_mode or self.split_chunk_idx is None: return "break"
        chunk = self.chunks[self.split_chunk_idx]
        self.split_time = min(chunk['filtered_ends'] - 0.05, self.split_time + 0.1)
        self.draw_all()
        self.update_edit_status()
        return "break"

    def handle_space(self, event):
        if not self.edit_mode: return
        if self.split_chunk_idx is None or self.split_time is None: return "break"
        
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_edit_preview()
        return "break"

    def handle_enter(self, event):
        if not self.edit_mode: return
        if self.split_chunk_idx is None or self.split_time is None: return "break"
        self.confirm_split()
        return "break"

    def start_edit_preview(self):
        sd.stop()
        chunk = self.chunks[self.split_chunk_idx]
        is_swapped = chunk.get('swapped', False)
        
        t_s = chunk['filtered_starts']
        t_e = chunk['filtered_ends']
        t_split = self.split_time
        
        s_start = int(t_s * self.sr)
        s_split = int(t_split * self.sr)
        s_end = int(t_e * self.sr)
        
        # lane0_spk and lane0_swapped
        lane0_spk = 2 if is_swapped else 1
        lane0_swapped = 1 if is_swapped else 2
        
        # lane1_spk and lane1_swapped
        lane1_spk = 1 if is_swapped else 2
        lane1_swapped = 2 if is_swapped else 1
        
        # Create preview buffers
        spk1_part1 = self.audio_data[lane0_spk][s_start : s_split]
        spk1_part2 = self.audio_data[lane0_swapped][s_split : s_end]
        spk1_preview = np.concatenate([spk1_part1, spk1_part2])
        
        spk2_part1 = self.audio_data[lane1_spk][s_start : s_split]
        spk2_part2 = self.audio_data[lane1_swapped][s_split : s_end]
        spk2_preview = np.concatenate([spk2_part1, spk2_part2])
        
        silence = np.zeros(int(0.5 * self.sr))
        full_preview = np.concatenate([spk1_preview, silence, spk2_preview])
        
        sd.play(full_preview, self.sr)
        self.play_real_start_time = time.time()
        self.is_playing = True
        self.animate_edit_playhead()

    def animate_edit_playhead(self):
        if not self.is_playing or not self.edit_mode: return
        elapsed = time.time() - self.play_real_start_time
        chunk = self.chunks[self.split_chunk_idx]
        dur = chunk['filtered_ends'] - chunk['filtered_starts']
        silence_dur = 0.5
        
        if elapsed < dur:
            # Playing SPK1
            curr_pos = chunk['filtered_starts'] + elapsed
            self.label_timer.configure(text=f"預覽 SPK1: {self.format_time(curr_pos)}")
            px = self.get_x_at_time(curr_pos)
            self.canvas.coords(self.playhead_line, px, 0, px, self.canvas.winfo_height())
        elif elapsed < dur + silence_dur:
            # Silence
            self.label_timer.configure(text="預覽間歇 (0.5s)...")
        elif elapsed < dur * 2 + silence_dur:
            # Playing SPK2
            curr_pos = chunk['filtered_starts'] + (elapsed - dur - silence_dur)
            self.label_timer.configure(text=f"預覽 SPK2: {self.format_time(curr_pos)}")
            px = self.get_x_at_time(curr_pos)
            self.canvas.coords(self.playhead_line, px, 0, px, self.canvas.winfo_height())
        else:
            self.stop_playback()
            return
            
        if self.is_playing:
            self.after(20, self.animate_edit_playhead)

    def confirm_split(self):
        if self.split_chunk_idx is None or self.split_time is None: return
        
        idx = self.split_chunk_idx
        chunk = self.chunks[idx]
        t_s = chunk['filtered_starts']
        t_e = chunk['filtered_ends']
        is_swapped = chunk.get('swapped', False)
        
        # Create two sub-chunks
        chunk_a = {
            "filtered_starts": t_s,
            "filtered_ends": self.split_time,
            "swapped": is_swapped
        }
        chunk_b = {
            "filtered_starts": self.split_time,
            "filtered_ends": t_e,
            "swapped": not is_swapped # Automatically invert swap of the second half
        }
        
        # Replace original chunk with chunk_a and chunk_b
        self.chunks.pop(idx)
        self.chunks.insert(idx, chunk_b)
        self.chunks.insert(idx, chunk_a)
        
        # Re-sequence chunks_id
        for i, c in enumerate(self.chunks):
            c['chunks_id'] = i
            
        if self.is_playing:
            self.stop_playback()
            
        # Reset edit state and exit edit mode
        self.split_chunk_idx = None
        self.split_time = None
        self.mode_button.set("一般模式")
        self.select_mode("一般模式")
        
        messagebox.showinfo("成功", "已成功分割 Chunk，並自動翻轉後半段音軌！")

    def setup_ui(self):
        self.header = ctk.CTkLabel(self, text="wave player", font=("Arial", 32, "bold"), text_color="red")
        self.header.pack(pady=5)

        self.top_bar = ctk.CTkFrame(self)
        self.top_bar.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkButton(self.top_bar, text="載入資料夾", width=100, fg_color=self.BTN_BLUE, hover=False, command=self.manual_load_folder).pack(side="left", padx=10)
        self.mode_button = ctk.CTkSegmentedButton(self.top_bar, values=["一般模式", "分割模式"], command=self.select_mode)
        self.mode_button.pack(side="left", padx=10)
        self.mode_button.set("一般模式")
        self.label_info = ctk.CTkLabel(self.top_bar, text="滾輪: 縮放 | 中鍵/Shift+左鍵: 平移", text_color="gray")
        self.label_info.pack(side="left", padx=10)

        self.label_timer = ctk.CTkLabel(self.top_bar, text="00:00.000 / 00:00.000", font=("Consolas", 20, "bold"), text_color="#ECf0F1")
        self.label_timer.pack(side="left", expand=True)

        ctk.CTkButton(self.top_bar, text="匯出 (Export)", width=100, fg_color=self.BTN_BLUE, hover=False, command=self.export_final_wav).pack(side="right", padx=10)
        ctk.CTkButton(self.top_bar, text="儲存 (JSON)", width=100, fg_color=self.BTN_BLUE, hover=False, command=self.save_json).pack(side="right", padx=10)

        self.canvas_frame = ctk.CTkFrame(self, fg_color=self.BG_COLOR)
        self.canvas_frame.pack(fill="both", expand=True, padx=20, pady=20)

        self.canvas = Canvas(self.canvas_frame, bg=self.BG_COLOR, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Status Bar
        self.status_bar = ctk.CTkFrame(self, fg_color="#1E1E1E", height=30)
        self.status_bar.pack(fill="x", padx=20, pady=(0, 20))
        self.label_chunk_info = ctk.CTkLabel(self.status_bar, text="目前無選取 Chunk", font=("Consolas", 14), text_color="#A9D0F5")
        self.label_chunk_info.pack(side="left", padx=15, pady=5)
        
        self.canvas.bind("<Button-1>", self.handle_left_click)
        self.canvas.bind("<Button-3>", self.handle_right_click)
        self.canvas.bind("<Button-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.do_pan)
        self.canvas.bind("<MouseWheel>", self.handle_zoom)
        self.canvas.bind("<Shift-Button-1>", self.start_pan)
        self.canvas.bind("<Shift-B1-Motion>", self.do_pan)

        # Keyboard Bindings for Edit Mode
        self.bind("<Left>", self.handle_arrow_left)
        self.bind("<Right>", self.handle_arrow_right)
        self.bind("<space>", self.handle_space)
        self.bind("<Return>", self.handle_enter)

    def check_default_output_dir(self):
        if os.path.exists("outputdir.txt"):
            with open("outputdir.txt", 'r', encoding='utf-8') as f:
                path = f.read().strip()
                if os.path.isdir(path): self.perform_load(path)

    def manual_load_folder(self):
        path = filedialog.askdirectory()
        if path: self.perform_load(path)

    def perform_load(self, path):
        try:
            self.folder_path = path
            with open(os.path.join(path, "chunks.json"), 'r') as f: self.chunks = json.load(f)
            mix, sr = sf.read(os.path.join(path, "mixture.wav"))
            s1, _ = sf.read(os.path.join(path, "spk1.wav"))
            s2, _ = sf.read(os.path.join(path, "spk2.wav"))
            self.audio_data = [mix, s1, s2]
            self.audio_tensors = [torch.tensor(x, dtype=torch.float32, device=self.device) for x in self.audio_data]
            self.sr, self.duration = sr, len(mix) / sr
            self.view_start_time, self.view_end_time = 0.0, self.duration
            self.label_info.configure(text="載入成功")
            self.label_timer.configure(text=f"00:00.000 / {self.format_time(self.duration)}")
            self.label_chunk_info.configure(text="目前無選取 Chunk")
            self.after(200, self.draw_all)
        except Exception as e: messagebox.showerror("錯誤", f"載入失敗: {e}")

    def get_time_at_x(self, x):
        w = self.canvas.winfo_width()
        return self.view_start_time + (x / w) * (self.view_end_time - self.view_start_time)

    def get_x_at_time(self, t):
        w = self.canvas.winfo_width()
        view_dur = self.view_end_time - self.view_start_time
        return ((t - self.view_start_time) / view_dur) * w if view_dur > 0 else 0

    def draw_all(self):
        self.canvas.delete("all")
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w <= 1 or h <= 1: return
        lane_h = h // 3

        if self.audio_tensors[0] is None:
            self.canvas.create_line(0, lane_h, w, lane_h, fill="#444444")
            self.canvas.create_line(0, lane_h*2, w, lane_h*2, fill="#444444")
            self.canvas.create_text(5, 5, text="SPK1", fill=self.TEXT_COLOR, anchor="nw")
            self.canvas.create_text(5, lane_h + 5, text="SPK2", fill=self.TEXT_COLOR, anchor="nw")
            self.canvas.create_text(5, lane_h*2 + 5, text="MIXTURE", fill=self.TEXT_COLOR, anchor="nw")
            return

        def hex_to_rgb(hex_str):
            hex_str = hex_str.lstrip('#')
            return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

        bg_rgb = hex_to_rgb(self.BG_COLOR)
        img = Image.new("RGB", (w, h), bg_rgb)
        draw = ImageDraw.Draw(img)

        # Draw Swap backgrounds
        swap_bg_rgb = hex_to_rgb(self.SWAP_BG)
        for idx, chunk in enumerate(self.chunks):
            t_s, t_e = chunk['filtered_starts'], chunk['filtered_ends']
            if t_e < self.view_start_time or t_s > self.view_end_time: continue
            x_s, x_e = self.get_x_at_time(t_s), self.get_x_at_time(t_e)
            if chunk.get('swapped', False):
                draw.rectangle([x_s, 0, x_e, lane_h * 2], fill=swap_bg_rgb)

        # Draw Waveforms
        wave_color_rgb = hex_to_rgb(self.WAVE_COLOR)
        mix_color_rgb = hex_to_rgb(self.MIX_COLOR)

        for idx, chunk in enumerate(self.chunks):
            t_s, t_e = chunk['filtered_starts'], chunk['filtered_ends']
            if t_e < self.view_start_time or t_s > self.view_end_time: continue
            x_s, x_e = self.get_x_at_time(t_s), self.get_x_at_time(t_e)
            is_swapped = chunk.get('swapped', False)

            self.draw_waveform_lane_segment(self.audio_tensors[2 if is_swapped else 1], t_s, t_e, x_s, x_e, 0, lane_h, wave_color_rgb, draw)
            self.draw_waveform_lane_segment(self.audio_tensors[1 if is_swapped else 2], t_s, t_e, x_s, x_e, lane_h, lane_h*2, wave_color_rgb, draw)
            self.draw_waveform_lane_segment(self.audio_tensors[0], t_s, t_e, x_s, x_e, lane_h*2, h, mix_color_rgb, draw)

        # Draw Grid boundaries
        line_color_rgb = hex_to_rgb("#444444")
        draw.line([0, lane_h, w, lane_h], fill=line_color_rgb)
        draw.line([0, lane_h*2, w, lane_h*2], fill=line_color_rgb)

        self.bg_photoimage = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, image=self.bg_photoimage, anchor="nw")

        # Draw Interactive Canvas items
        for idx, chunk in enumerate(self.chunks):
            t_s, t_e = chunk['filtered_starts'], chunk['filtered_ends']
            if t_e < self.view_start_time or t_s > self.view_end_time: continue
            x_s = self.get_x_at_time(t_s)
            chunk_id = chunk.get('chunks_id', idx)
            self.canvas.create_line(x_s, 0, x_s, h, fill="#555555", dash=(2, 2))
            self.canvas.create_text(x_s + 5, 20, text=f"CK {chunk_id}", fill=self.TEXT_COLOR, anchor="nw", font=("Arial", 8))

        self.canvas.create_text(5, 5, text="SPK1", fill=self.TEXT_COLOR, anchor="nw")
        self.canvas.create_text(5, lane_h + 5, text="SPK2", fill=self.TEXT_COLOR, anchor="nw")
        self.canvas.create_text(5, lane_h*2 + 5, text="MIXTURE", fill=self.TEXT_COLOR, anchor="nw")

        px = self.get_x_at_time(self.play_start_sec + (time.time() - self.play_real_start_time if self.is_playing else 0))
        self.playhead_line = self.canvas.create_line(px, 0, px, h, fill=self.PLAYHEAD_COLOR, width=2)

        # Draw split line if in edit mode
        if self.edit_mode and self.split_time is not None:
            sx = self.get_x_at_time(self.split_time)
            self.canvas.create_line(sx, 0, sx, h, fill="#E74C3C", width=3, dash=(4, 4))
            self.canvas.create_text(sx + 5, h - 30, text=f"分割點: {self.format_time(self.split_time)}", fill="#E74C3C", anchor="sw", font=("Arial", 10, "bold"))

    def draw_waveform_lane_segment(self, tensor_data, t_start, t_end, x_start, x_end, y_top, y_btm, color_rgb, draw):
        w_px = int(x_end - x_start)
        if w_px <= 0: return
        mid_y, max_h = (y_top + y_btm) // 2, (y_btm - y_top) // 2.2
        start_samp, end_samp = int(t_start * self.sr), int(t_end * self.sr)
        seg = tensor_data[start_samp:end_samp]
        if len(seg) == 0: return
        
        with torch.no_grad():
            seg_abs = torch.abs(seg).unsqueeze(0).unsqueeze(0)
            amps_tensor = torch.nn.functional.adaptive_max_pool1d(seg_abs, w_px)
            amps = (amps_tensor.view(-1) * max_h).cpu().numpy()
            
        for i in range(w_px):
            amp = amps[i]
            x = int(x_start + i)
            draw.line([x, mid_y - amp, x, mid_y + amp], fill=color_rgb)

    def handle_zoom(self, event):
        cursor_t = self.get_time_at_x(event.x)
        scale = 0.9 if event.delta > 0 else 1.1
        cur_dur = self.view_end_time - self.view_start_time
        new_dur = max(self.min_view_dur, min(self.duration, cur_dur * scale))
        ratio = (event.x / self.canvas.winfo_width())
        self.view_start_time = max(0, min(self.duration - new_dur, cursor_t - new_dur * ratio))
        self.view_end_time = self.view_start_time + new_dur
        self.draw_all()

    def start_pan(self, event): self.pan_last_x = event.x
    def do_pan(self, event):
        w = self.canvas.winfo_width()
        dx = event.x - self.pan_last_x
        dur = self.view_end_time - self.view_start_time
        dt = (dx / w) * dur
        self.view_start_time = max(0, min(self.duration - dur, self.view_start_time - dt))
        self.view_end_time = self.view_start_time + dur
        self.pan_last_x = event.x
        self.draw_all()

    def handle_left_click(self, event):
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        t = self.get_time_at_x(event.x)
        clicked_idx, chunk = next(((i, c) for i, c in enumerate(self.chunks) if c['filtered_starts'] <= t <= c['filtered_ends']), (None, None))
        
        if self.edit_mode:
            if chunk is None: return
            if self.is_playing: self.stop_playback()
            # Clamp split time to at least 0.05s from boundary
            self.split_chunk_idx = clicked_idx
            self.split_time = max(chunk['filtered_starts'] + 0.05, min(chunk['filtered_ends'] - 0.05, t))
            self.draw_all()
            self.update_edit_status()
            self.canvas.focus_set() # Enable canvas focus to receive key events
            return

        if chunk is None:
            if self.is_playing: self.stop_playback()
            return
        
        self.update_chunk_info(clicked_idx)
        
        lane_idx = event.y // (h // 3)
        lane_idx = min(lane_idx, 2) 

        if self.is_playing and clicked_idx == self.current_playing_chunk_idx and lane_idx == self.current_playing_visual_lane:
            self.stop_playback(); return

        self.current_playing_visual_lane = lane_idx
        self.start_chunk_playback(clicked_idx)

    def start_chunk_playback(self, chunk_idx):
        sd.stop()
        self.current_playing_chunk_idx = chunk_idx
        self.update_chunk_info(chunk_idx)
        chunk = self.chunks[chunk_idx]
        is_swapped = chunk.get('swapped', False)
        
        if self.current_playing_visual_lane == 0: 
            audio_idx = 2 if is_swapped else 1
        elif self.current_playing_visual_lane == 1: 
            audio_idx = 1 if is_swapped else 2
        else: 
            audio_idx = 0
        
        self.play_start_sec, self.play_end_sec = chunk['filtered_starts'], chunk['filtered_ends']
        data = self.audio_data[audio_idx][int(self.play_start_sec*self.sr):int(self.play_end_sec*self.sr)]
        sd.play(data, self.sr)
        self.play_real_start_time, self.is_playing = time.time(), True
        self.animate_playhead()

    def stop_playback(self):
        sd.stop(); self.is_playing = False; self.current_playing_chunk_idx = None
        self.label_timer.configure(text=f"00:00.000 / {self.format_time(self.duration)}")

    def animate_playhead(self):
        if not self.is_playing: return
        elapsed = time.time() - self.play_real_start_time
        curr_pos = self.play_start_sec + elapsed
        self.label_timer.configure(text=f"{self.format_time(curr_pos)} / {self.format_time(self.duration)}")
        if curr_pos >= self.play_end_sec:
            next_idx = self.current_playing_chunk_idx + 1
            if next_idx < len(self.chunks): self.start_chunk_playback(next_idx); return
            else: curr_pos = self.play_end_sec; self.stop_playback()
        px = self.get_x_at_time(curr_pos)
        self.canvas.coords(self.playhead_line, px, 0, px, self.canvas.winfo_height())
        if self.is_playing: self.after(20, self.animate_playhead)

    def handle_right_click(self, event):
        if self.edit_mode: return # Disable swap in edit mode
        t = self.get_time_at_x(event.x)
        idx = next((i for i, c in enumerate(self.chunks) if c['filtered_starts'] <= t <= c['filtered_ends']), None)
        if idx is not None:
            self.update_chunk_info(idx)
            self.chunks[idx]['swapped'] = not self.chunks[idx].get('swapped', False)
            self.draw_all()

    def save_json(self):
        if not self.folder_path: return
        with open(os.path.join(self.folder_path, "chunks.json"), 'w') as f: json.dump(self.chunks, f, indent=4)
        messagebox.showinfo("成功", "JSON 已儲存")

    def export_final_wav(self):
        if not self.folder_path or self.audio_data[1] is None: return
        if not messagebox.askyesno("匯出", "確定匯出最終音軌？"): return
        self.label_info.configure(text="匯出中...", text_color="orange")
        def run():
            try:
                s1, s2 = self.audio_data[1].copy(), self.audio_data[2].copy()
                for c in self.chunks:
                    if c.get("swapped"):
                        s, e = int(c['filtered_starts']*self.sr), int(c['filtered_ends']*self.sr)
                        tmp = s1[s:e].copy()
                        s1[s:e] = s2[s:e]
                        s2[s:e] = tmp
                sf.write(os.path.join(self.folder_path, "spk1_final.wav"), s1, self.sr)
                sf.write(os.path.join(self.folder_path, "spk2_final.wav"), s2, self.sr)
                self.after(0, lambda: messagebox.showinfo("完成", "匯出成功")); self.after(0, lambda: self.label_info.configure(text="匯出成功", text_color="green"))
            except Exception as e: self.after(0, lambda: messagebox.showerror("錯誤", str(e)))
        threading.Thread(target=run).start()

if __name__ == "__main__":
    app = WaveformSequencePlayer()
    app.mainloop()