import torch
import torch.nn.functional as F
from asteroid.dsp.overlap_add import LambdaOverlapAdd
import librosa
import json
from pathlib import Path
from speechbrain.pretrained import EncoderClassifier

 
from look2hear.utils.logging import AverageMeter
from functions.silence_split import magspec_vad, webrtc_vad
from functions.overlapadd_w2v import PITLossWrapper_Out_BatchIndices


class LambdaOverlapAdd_Chunkwise_SpectralFeatures(LambdaOverlapAdd):
    """
    Code for Chunk-wise processing, assignment is perfomed by Spectral features (here, we used mfcc or spectral centroid)
    """

    def __init__(
        self,
        nnet,
        n_src,
        window_size,
        hop_size=None,
        window="hanning",
        reorder_chunks=True,
        enable_grad=False,
        device="cpu",
        sr=24000,
        vad_method="spec",
        spectral_features="mfcc",
        output_dir=None,
    ):
        super().__init__(
            nnet, n_src, window_size, hop_size, window, reorder_chunks, enable_grad
        )
        self.nnet = self.nnet.to(device)
        if spectral_features == "deep_embedding":
            self.embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                # 重點：將 self.device 轉為字串 str(self.device)
                run_opts={"device": str(device)}, 
                savedir="pretrained_models/spkrec-ecapa-voxceleb" # 建議指定目錄，避免重複下載
            )
        else:
            self.embedding_model = None
            
        self.device = device
        self.sr = sr
        self.vad_method = vad_method
        self.spectral_features = spectral_features
        self.output_dir = output_dir
        self.last_sf = None

    def ola_forward(self, x):
        """Heart of the class: segment signal, apply func, combine with OLA."""
        self.sc_avg = AverageMeter()  # to cumulate previous spectral centroids
        chunks_data = []

        assert x.ndim == 3

        batch, channels, n_frames = x.size()

        # 1. 執行 VAD (修正：改用固定且精確的參數，不跟隨 self.window_size)
        vad_n_fft = 1024   # 約 42ms (24kHz 下)
        vad_hop = 256      # 約 10ms
        
        if self.vad_method == "spec":
            starts, ends = magspec_vad(
                x.cpu().numpy()[0, 0, :],
                n_fft=vad_n_fft,
                hop_length=vad_hop,
            )
            # 更新 Debug 用的參數顯示
            win_ms = (vad_n_fft / self.sr) * 1000
            hop_ms = (vad_hop / self.sr) * 1000
            overlap_ratio = (1 - vad_hop / vad_n_fft) * 100
            repeat_times = vad_n_fft // vad_hop
            
        elif self.vad_method == "webrtc":
            starts, ends = webrtc_vad(
                x.cpu().numpy()[0, 0, :], self.sr, vad_mode=3, frame_size=0.03
            )
            # WebRTC 固定參數
            win_ms = 30.0
            hop_ms = 30.0
            overlap_ratio = 0
            repeat_times = 1
        
         # --- 新增口語化 Debug 訊息 ---
        def format_time(n_samples, sr):
            seconds = n_samples / sr
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = seconds % 60
            return f"{h:02d}:{m:02d}:{s:05.2f}"

        print(f"\n{'='*50}")
        print(f"【VAD 語音偵測回報】")
        print(f"  偵測模式：{self.vad_method}")
        
        if self.vad_method == "spec":
            print(f"  分析細節：我每次抓 {win_ms:.1f}ms 的聲音來看，然後往後移 {hop_ms:.1f}ms。")
            print(f"            這表示每個點都被我重複檢查了 {repeat_times} 次 (重疊率 {overlap_ratio:.0f}%)，抓邊界會很精準喔！")
        else:
            print(f"  分析細節：我每隔 {win_ms:.1f}ms 跳著檢查一次有沒有人聲。")

        filtered_starts = []
        filtered_ends = []
        if len(starts) == 0:
            print(f"  偵測結果：掃描整段音訊後... 竟然連一點聲音都沒發現！")
        else:
            print(f"  偵測結果：嘿！我發現了 {len(starts)} 段有聲音的地方：")
            for i, (s, e) in enumerate(zip(starts, ends)):
                duration = (e - s) / self.sr
                if duration < 0.5:
                    print(f"    ❌ 段落 {i+1:02d}: {format_time(s, self.sr)} ~ {format_time(e, self.sr)} (太短 {duration:.2f}s, 已丟棄)")
                else:
                    print(f"    👉 段落 {i+1:02d}: {format_time(s, self.sr)} ~ {format_time(e, self.sr)}")
                    filtered_starts.append(s)
                    filtered_ends.append(e)
                    
        # chunk頭尾各加 0.5sec, 增加模型的分離能力
        context_pad = 0.5
        pad_samples = int(context_pad * self.sr)
        starts = [max(0, s - pad_samples) for s in filtered_starts]
        ends = [min(n_frames, e + pad_samples) for e in filtered_ends]
        print(f"{'='*50}\n")
        # --- Debug 訊息結束 ---
 
        # First, make the output tensor. divide by n_src will make sum of the output be consistent with input
        # except the regions where voice activity detected.
        out = (x / self.n_src).repeat(1, self.n_src, 1)  # [batch, n_src, n_frames]
        assert len(starts) == len(ends)

        for frame_idx in range(len(starts)):  # for loop to spare memory
            frame_length = ends[frame_idx] - starts[frame_idx]
            if (
                frame_length <= self.window_size // 2
            ):  # if input frames are too short, an error occurs.
                pad_each_side = int((self.window_size // 2 - frame_length) / 2) + 1
                segment = F.pad(
                    x[..., starts[frame_idx] : ends[frame_idx]],
                    (pad_each_side, pad_each_side),
                )
                frame = self.nnet(segment)
            else:
                segment = x[..., starts[frame_idx] : ends[frame_idx]]
                frame = self.nnet(segment)
            
            if frame_idx == 0:
                assert frame.ndim == 3, "nnet should return (batch, n_src, time)"
                if self.n_src is not None:
                    assert (
                        frame.shape[1] == self.n_src
                    ), "nnet should return (batch, n_src, time)"
                n_src = frame.shape[1]
                sf_output_list = []
                for src in range(n_src):
                    if self.spectral_features == "deep_embedding":
                        spec_feat_output = self._extract_deep_embedding(frame[0, src, :])
                    elif self.spectral_features == "mfcc":
                        spec_feat_output = torch.as_tensor(
                            librosa.feature.mfcc(
                                y=frame[0, src, :].cpu().numpy(),
                                sr=self.sr,
                                n_mfcc=20,
                                n_fft=self.window_size,
                                hop_length=self.hop_size,
                            )[1:, :]
                            .mean(1, keepdims=True)
                            .T,
                            device=self.device,
                        ).unsqueeze(0)
                    elif self.spectral_features == "spectral_centroid":
                        spec_feat_output = torch.as_tensor(
                            librosa.feature.spectral_centroid(
                                y=frame[0, src, :].cpu().numpy(),
                                sr=self.sr,
                                n_fft=self.window_size,
                                hop_length=self.hop_size,
                            ).mean(1, keepdims=True),
                            device=self.device,
                        ).unsqueeze(0)

                    sf_output_list.append(spec_feat_output)
                sf_output_list = torch.cat(
                    sf_output_list, dim=1
                )  # [batch, n_src, feature_dim]
                self.sc_avg.update(sf_output_list)
                self.last_sf = sf_output_list

            if frame_idx != 0 and self.reorder_chunks:
                # we determine best perm based on xcorr with previous sources
                # Use the last chunk's spectral features for better local continuity
                # and to solve the permutation problem.
                ref_sf = self.last_sf if self.last_sf is not None else self.sc_avg.avg
                frame, sc_out = self._reorder_sources_with_sf_and_non_overlapped_seg(
                    frame,
                    out[..., starts[frame_idx - 1] : ends[frame_idx - 1]],
                    ref_sf,
                    n_src,
                )
                self.sc_avg.update(sc_out)
                self.last_sf = sc_out
            if frame_length <= self.window_size // 2:
                frame = frame[..., pad_each_side:-pad_each_side]

            frame = self._suppress_ghosts(frame, frame_idx)
            out[..., starts[frame_idx] : ends[frame_idx]] = frame
            
            # Collect chunk data
            chunks_data.append({
                "chunks_id": frame_idx,
                "filtered_starts": filtered_starts[frame_idx] / self.sr if frame_idx < len(filtered_starts) else 0,
                "filtered_ends": filtered_ends[frame_idx] / self.sr if frame_idx < len(filtered_ends) else 0,
                "swapped": False
            })

        output_path = Path(self.output_dir) / "chunks.json" if self.output_dir else Path("chunks.json")
        with open(output_path, "w") as f:
            json.dump(chunks_data, f, indent=4)
            
        return out

    def forward(self, x):
        """Forward module: segment signal, apply func, combine with OLA.

        Args:
            x (:class:`torch.Tensor`): waveform signal of shape (batch, 1, time).

        Returns:
            :class:`torch.Tensor`: The output of the lambda OLA.
        """
        # Here we can do the reshaping
        with torch.autograd.set_grad_enabled(self.enable_grad):
            olad = self.ola_forward(x)
            return olad

    def _reorder_sources_with_sf_and_non_overlapped_seg(
        self,
        current: torch.FloatTensor,
        previous: torch.FloatTensor,
        previous_sf: torch.FloatTensor,
        n_src: int,
    ):
        """
        Reorder sources in current chunk to maximize correlation with previous chunk.
        Used for Continuous Source Separation. Wav2Vec2.0-based correlation is used
        for reordering.

        Args:
            current (:class:`torch.Tensor`): current chunk, tensor
                                            of shape (batch, n_src, window_size)
            previous (:class:`torch.Tensor`): previous chunk, tensor
                                            of shape (batch, n_src, window_size)
            n_src (:class:`int`): number of sources.
            window_size (:class:`int`): window_size, equal to last dimension of
                                        both current and previous.
            hop_size (:class:`int`): hop_size between current and previous tensors.

        """
        # batch, frames = current.size()
        batch, n_src, frames = current.size()

        def reorder_func_sf(x):
            sf_output_list = self._extract_spectral_features(x)
            return (
                -F.cosine_similarity(
                    sf_output_list.unsqueeze(1), previous_sf.unsqueeze(2), dim=-1
                ),
                sf_output_list,
            )

        # We maximize correlation-like between previous and current.
        pit = PITLossWrapper_Out_BatchIndices(
            reorder_func_sf
        )  # So, reorder_func is a loss_function in PITLossWrapper

        _, current, current_sf = pit(current, previous)
        return (
            current,
            current_sf,
        )

    def _extract_spectral_features(self, x):
        """Extract spectral features for all sources in the frame.
        x: [batch, n_src, time]
        Returns: [batch, n_src, feature_dim]
        """
        batch, n_src, _ = x.size()
        sf_output_list = []
        for src in range(n_src):
            # Use the first batch element for feature extraction as per original implementation
            waveform = x[0, src, :]
            if self.spectral_features == "deep_embedding":
                spec_feat_output = self._extract_deep_embedding(waveform)
            elif self.spectral_features == "mfcc":
                spec_feat_output = torch.as_tensor(
                    librosa.feature.mfcc(
                        y=waveform.cpu().numpy(),
                        sr=self.sr,
                        n_mfcc=20,
                        n_fft=self.window_size,
                        hop_length=self.hop_size,
                    )[1:, :]
                    .mean(1, keepdims=True)
                    .T,
                    device=self.device,
                ).unsqueeze(0)
            elif self.spectral_features == "spectral_centroid":
                spec_feat_output = torch.as_tensor(
                    librosa.feature.spectral_centroid(
                        y=waveform.cpu().numpy(),
                        sr=self.sr,
                        n_fft=self.window_size,
                        hop_length=self.hop_size,
                    ).mean(1, keepdims=True),
                    device=self.device,
                ).unsqueeze(0)
            else:
                spec_feat_output = torch.zeros((1, 1, 1), device=self.device)

            sf_output_list.append(spec_feat_output)
        
        sf_output_list = torch.cat(sf_output_list, dim=1) # [batch, n_src, feature_dim]
        return sf_output_list

    def _suppress_ghosts(self, frame, frame_idx=None):
        """Suppress 'ghost' sources in the harmony channel if they are low energy and similar to the lead."""
        # frame: [batch, n_src, time]
        batch, n_src, _ = frame.size()
        if n_src < 2:
            return frame

        # 1. Calculate energy for each source
        energies = torch.norm(frame, dim=-1) # [batch, n_src]
        
        # 2. Extract spectral features
        sf = self._extract_spectral_features(frame) # [batch, n_src, feature_dim]
        
        # Process each batch
        for b in range(batch):
            # Dynamically identify stronger and weaker channels among the first two
            e0, e1 = energies[b, 0], energies[b, 1]
            weaker_idx = 0 if e0 < e1 else 1
            stronger_idx = 1 - weaker_idx
            
            lead_energy = energies[b, stronger_idx]
            harm_energy = energies[b, weaker_idx]
            lead_sf = sf[b, stronger_idx]
            harm_sf = sf[b, weaker_idx]
            
            # Ghost criteria: weaker is low energy AND highly similar to stronger
            energy_ratio = harm_energy / (lead_energy + 1e-6)
            similarity = F.cosine_similarity(harm_sf.unsqueeze(0), lead_sf.unsqueeze(0)).item()
            
            is_ghost = energy_ratio < 0.2 and similarity > 0.9
            if is_ghost:
                frame[b, weaker_idx, :] = 0
            
            # Debug message
            idx_str = f"Chunk {frame_idx}" if frame_idx is not None else "Unknown Chunk"
            result_str = "❌ SUPPRESSED" if is_ghost else "✅ KEPT"
            print(f"[{idx_str} Batch {b}] Ratio: {energy_ratio:.4f}, Sim: {similarity:.4f} -> {result_str} (Weaker: Ch{weaker_idx})")
                    
        return frame

    def _extract_deep_embedding(self, waveform):
        """Extract speaker embedding using pre-trained ECAPA-TDNN model."""
        if self.embedding_model is None:
            raise RuntimeError("Embedding model not initialized. Set spectral_features='deep_embedding' in __init__.")

        # waveform: [time]
        if waveform.ndim > 1:
            waveform = waveform.flatten()

        # SpeechBrain expects [batch, time]
        waveform = waveform.unsqueeze(0).to(self.device)

        with torch.no_grad():
            embeddings = self.embedding_model.encode_batch(waveform)
            # embeddings: [batch, 1, embedding_dim]
            embedding = embeddings.squeeze(0).squeeze(0)  # [embedding_dim]

        return embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, embedding_dim]
