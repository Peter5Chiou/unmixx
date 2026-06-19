import torch
import torch.nn.functional as F
from asteroid.dsp.overlap_add import LambdaOverlapAdd
import librosa


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
        vad_threshold=0.2,
        n_mfcc=40,
        debug_silence_dur=0.0,
        chunk_factor=1.0,
    ):
        super().__init__(
            nnet, n_src, window_size, hop_size, window, reorder_chunks, enable_grad
        )
        self.nnet = self.nnet.to(device)
        self.device = device
        self.sr = sr
        self.vad_method = vad_method
        self.spectral_features = spectral_features
        self.vad_threshold = vad_threshold
        self.n_mfcc = n_mfcc
        self.debug_silence_dur = debug_silence_dur
        self.chunk_factor = chunk_factor

    def ola_forward(self, x):
        """Heart of the class: segment signal, apply func, combine with OLA."""
        self.sc_avg = AverageMeter()  # to cumulate previous spectral centroids

        assert x.ndim == 3

        batch, channels, n_frames = x.size()

        if self.vad_method == "spec":
            starts, ends = magspec_vad(
                x.cpu().numpy()[0, 0, :],
                n_fft=self.window_size,
                hop_length=self.hop_size,
                threshold=self.vad_threshold,
            )
        elif self.vad_method == "webrtc":
            starts, ends = webrtc_vad(
                x.cpu().numpy()[0, 0, :], self.sr, vad_mode=3, frame_size=0.03
            )
        
        output_chunks = []
        last_pos = 0
        
        first_chunk = True
        previous_chunk = None

        for frame_idx in range(len(starts)):  # for loop to spare memory
            curr_start = starts[frame_idx]
            curr_end = ends[frame_idx]
            
            # Add non-voice region before this segment
            if curr_start > last_pos:
                non_voice = (x[..., last_pos : curr_start] / self.n_src).repeat(1, self.n_src, 1)
                output_chunks.append(non_voice)
            
            # Split long segments into chunks to prevent VRAM OOM
            temp_start = curr_start
            if self.chunk_factor > 0:
                max_chunk_size = int(self.window_size * self.chunk_factor)
                while curr_end - temp_start > max_chunk_size:
                    chunk_end = temp_start + max_chunk_size
                    segment = x[..., temp_start : chunk_end]
                    frame = self.nnet(segment)
                    
                    # Reorder and update spectral features
                    if first_chunk:
                        sf = self._extract_spectral_features(frame)
                        self.sc_avg.update(sf)
                        first_chunk = False
                    elif self.reorder_chunks:
                        frame, sf = self._reorder_sources_with_sf_and_non_overlapped_seg(
                            frame, previous_chunk, self.sc_avg.avg, frame.shape[1]
                        )
                        self.sc_avg.update(sf)
                    
                    output_chunks.append(frame)
                    
                    # Insert debug silence
                    if self.debug_silence_dur > 0:
                        silence_len = int(self.debug_silence_dur * self.sr)
                        silence = torch.zeros((batch, self.n_src, silence_len), device=self.device)
                        output_chunks.append(silence)
                    
                    previous_chunk = frame
                    temp_start = chunk_end
            
            # Process the remaining part (or the whole segment if it was short enough)
            frame_length = curr_end - temp_start
            if frame_length > 0:
                if frame_length <= self.window_size // 2:
                    pad_each_side = int((self.window_size // 2 - frame_length) / 2) + 1
                    segment = F.pad(
                        x[..., temp_start : curr_end],
                        (pad_each_side, pad_each_side),
                    )
                    frame = self.nnet(segment)
                    
                    # Reorder and update spectral features
                    if first_chunk:
                        sf = self._extract_spectral_features(frame)
                        self.sc_avg.update(sf)
                        first_chunk = False
                    elif self.reorder_chunks:
                        frame, sf = self._reorder_sources_with_sf_and_non_overlapped_seg(
                            frame, previous_chunk, self.sc_avg.avg, frame.shape[1]
                        )
                        self.sc_avg.update(sf)
                    
                    # Use original length for output
                    output_chunks.append(frame[..., :frame_length])
                    
                    if self.debug_silence_dur > 0:
                        silence_len = int(self.debug_silence_dur * self.sr)
                        silence = torch.zeros((batch, self.n_src, silence_len), device=self.device)
                        output_chunks.append(silence)
                    
                    previous_chunk = frame
                else:
                    segment = x[..., temp_start : curr_end]
                    frame = self.nnet(segment)
                    
                    # Reorder and update spectral features
                    if first_chunk:
                        sf = self._extract_spectral_features(frame)
                        self.sc_avg.update(sf)
                        first_chunk = False
                    elif self.reorder_chunks:
                        frame, sf = self._reorder_sources_with_sf_and_non_overlapped_seg(
                            frame, previous_chunk, self.sc_avg.avg, frame.shape[1]
                        )
                        self.sc_avg.update(sf)
                    
                    output_chunks.append(frame)
                    
                    if self.debug_silence_dur > 0:
                        silence_len = int(self.debug_silence_dur * self.sr)
                        silence = torch.zeros((batch, self.n_src, silence_len), device=self.device)
                        output_chunks.append(silence)
                    
                    previous_chunk = frame
            
            last_pos = curr_end

        # Add final non-voice region
        if last_pos < n_frames:
            non_voice = (x[..., last_pos : n_frames] / self.n_src).repeat(1, self.n_src, 1)
            output_chunks.append(non_voice)

        return torch.cat(output_chunks, dim=-1)

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

    def _extract_spectral_features(self, frame):
        """Extracts spectral features (MFCC or Spectral Centroid) from a frame."""
        batch, n_src, frames = frame.shape
        
        # Pad frame to at least window_size to avoid librosa warning when frames < n_fft
        if frames < self.window_size:
            frame = F.pad(frame, (0, self.window_size - frames))
            
        sf_output_list = []
        for src in range(n_src):
            if self.spectral_features == "mfcc":
                spec_feat_output = torch.as_tensor(
                    librosa.feature.mfcc(
                        y=frame[0, src, :].cpu().numpy(),
                        sr=self.sr,
                        n_mfcc=self.n_mfcc,
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
        
        sf_output_list = torch.cat(sf_output_list, dim=1)  # [batch, n_src, feature_dim]
        return sf_output_list

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
