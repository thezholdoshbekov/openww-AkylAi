"""DTLN-aec ONNX runner — streaming neural acoustic echo canceller.

Wraps the two ONNX models from breizhn/DTLN-aec (Apache-2.0) in a stateful
streaming class. Block-by-block call: `clean = aec.process(mic_block, ref_block)`.

Architecture (per block, 8 ms = 128 samples @ 16 kHz):
  mic ring buffer (512 samp) ──> FFT magnitude ──┐
                                                  ├──> model_1 ──> mask
  ref ring buffer (512 samp) ──> FFT magnitude ──┘
  mic magnitude * mask, combined with mic phase ──> iFFT ──> model_2 ──> clean block
  overlap-add into output ring buffer; emit shift samples.

LSTM states for both models are persisted across blocks (stateful streaming).

Get the models from https://github.com/breizhn/DTLN-aec (pretrained_models/):
  dtln_aec_512_1.onnx
  dtln_aec_512_2.onnx
Place them under aec/models/ (or pass --m1/--m2).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import onnxruntime as ort


SR = 16000
BLOCK_LEN = 512        # FFT size / analysis window
BLOCK_SHIFT = 128      # hop = 8 ms; pipeline emits one block_shift per process() call


class DTLNAec:
    def __init__(self, model1_path: str, model2_path: str,
                 providers=("CPUExecutionProvider",)):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1            # one-thread per session is faster on small ARM cores
        opts.inter_op_num_threads = 1
        self.s1 = ort.InferenceSession(model1_path, sess_options=opts, providers=list(providers))
        self.s2 = ort.InferenceSession(model2_path, sess_options=opts, providers=list(providers))

        # Discover input/output ordering. Each DTLN-aec sub-model has 2 inputs
        # (audio-like tensor + LSTM state) and 2 outputs (result + new LSTM state).
        self.s1_in_audio, self.s1_in_state = (i.name for i in self.s1.get_inputs())
        self.s1_out_mask, self.s1_out_state = (o.name for o in self.s1.get_outputs())
        self.s2_in_audio, self.s2_in_state = (i.name for i in self.s2.get_inputs())
        self.s2_out_block, self.s2_out_state = (o.name for o in self.s2.get_outputs())

        # LSTM states (shapes come from the model itself — usually [1, 2, N, 2])
        s1_state_shape = next(i.shape for i in self.s1.get_inputs() if i.name == self.s1_in_state)
        s2_state_shape = next(i.shape for i in self.s2.get_inputs() if i.name == self.s2_in_state)
        self.state1 = np.zeros(s1_state_shape, dtype=np.float32)
        self.state2 = np.zeros(s2_state_shape, dtype=np.float32)

        # Streaming ring buffers
        self.mic_buf = np.zeros(BLOCK_LEN, dtype=np.float32)
        self.ref_buf = np.zeros(BLOCK_LEN, dtype=np.float32)
        self.out_buf = np.zeros(BLOCK_LEN, dtype=np.float32)

    def reset(self):
        self.state1.fill(0); self.state2.fill(0)
        self.mic_buf.fill(0); self.ref_buf.fill(0); self.out_buf.fill(0)

    def process(self, mic_block: np.ndarray, ref_block: np.ndarray) -> np.ndarray:
        """Process one BLOCK_SHIFT-sized chunk (128 samp = 8 ms). Returns 128 cleaned samp."""
        assert mic_block.shape[0] == BLOCK_SHIFT and ref_block.shape[0] == BLOCK_SHIFT, \
            f"need {BLOCK_SHIFT}-sample blocks, got mic={mic_block.shape[0]} ref={ref_block.shape[0]}"
        m = mic_block.astype(np.float32)
        r = ref_block.astype(np.float32)

        # shift-and-append (overlap-add windows)
        self.mic_buf = np.concatenate([self.mic_buf[BLOCK_SHIFT:], m])
        self.ref_buf = np.concatenate([self.ref_buf[BLOCK_SHIFT:], r])

        # rFFT → magnitude + phase (mic) and magnitude (ref)
        mic_fft = np.fft.rfft(self.mic_buf)
        ref_fft = np.fft.rfft(self.ref_buf)
        mic_mag = np.abs(mic_fft).astype(np.float32)
        mic_phase = np.angle(mic_fft)
        ref_mag = np.abs(ref_fft).astype(np.float32)

        # model_1 input: concatenated [mic_mag | ref_mag], shape (1, 1, 2*F)
        in1 = np.concatenate([mic_mag, ref_mag])[None, None, :]
        mask, self.state1 = self.s1.run(
            [self.s1_out_mask, self.s1_out_state],
            {self.s1_in_audio: in1, self.s1_in_state: self.state1},
        )

        # apply mask in magnitude domain, restore phase, iFFT
        est_mag = mic_mag * mask.reshape(-1)
        est_block = np.fft.irfft(est_mag * np.exp(1j * mic_phase)).astype(np.float32)
        in2 = est_block[None, None, :]

        # model_2 refines time-domain block + advances its LSTM
        out_block, self.state2 = self.s2.run(
            [self.s2_out_block, self.s2_out_state],
            {self.s2_in_audio: in2, self.s2_in_state: self.state2},
        )
        out_block = out_block.reshape(-1)

        # overlap-add into the output ring; emit the leftmost BLOCK_SHIFT
        self.out_buf = np.concatenate([self.out_buf[BLOCK_SHIFT:], np.zeros(BLOCK_SHIFT, dtype=np.float32)])
        self.out_buf += out_block
        return self.out_buf[:BLOCK_SHIFT].copy()


def process_file(mic_wav: str, ref_wav: str, model1: str, model2: str, out_wav: str) -> None:
    """Offline convenience: feed two synchronous wavs through the AEC and save the result."""
    import soundfile as sf
    mic, sr_m = sf.read(mic_wav, dtype="int16", always_2d=False)
    ref, sr_r = sf.read(ref_wav, dtype="int16", always_2d=False)
    assert sr_m == SR and sr_r == SR, f"need {SR} Hz wavs (got mic={sr_m}, ref={sr_r})"

    aec = DTLNAec(model1, model2)
    n = min(len(mic), len(ref)) // BLOCK_SHIFT
    out = np.zeros(n * BLOCK_SHIFT, dtype=np.float32)
    mic_f = mic.astype(np.float32) / 32768.0
    ref_f = ref.astype(np.float32) / 32768.0
    for i in range(n):
        s = slice(i * BLOCK_SHIFT, (i + 1) * BLOCK_SHIFT)
        out[s] = aec.process(mic_f[s], ref_f[s])
    sf.write(out_wav, np.clip(out, -1, 1), SR, subtype="PCM_16")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mic"); ap.add_argument("ref"); ap.add_argument("out")
    ap.add_argument("--m1", default=str(Path(__file__).parent / "models" / "dtln_aec_512_1.onnx"))
    ap.add_argument("--m2", default=str(Path(__file__).parent / "models" / "dtln_aec_512_2.onnx"))
    a = ap.parse_args()
    process_file(a.mic, a.ref, a.m1, a.m2, a.out)
    print(f"wrote {a.out}")
