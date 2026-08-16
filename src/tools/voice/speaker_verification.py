"""
声纹验证模块 - 使用 ONNX Runtime 实现低延迟声纹验证
默认使用在 20 万中文说话人数据上训练的 ERes2NetV2，也可回退 WeSpeaker ResNet34_LM
依赖: onnxruntime, numpy, soundfile, scipy
内存占用 ~200MB（对比 SpeechBrain 方案 ~1.5GB）
支持多样本注册提高准确率
支持声纹持久化保存/加载
"""
import numpy as np
import os
import time
from typing import Optional, Tuple, List, Dict
from pathlib import Path

# 声纹数据保存目录
SPEAKER_DATA_DIR = Path("data/speakers")

# ============================================================
#  纯 numpy/scipy 实现 Kaldi 兼容 Fbank 特征提取（替代 torchaudio）
# ============================================================

def _load_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """加载音频文件并重采样到 target_sr，返回 float32 单声道"""
    import soundfile as sf
    data, sr = sf.read(audio_path, dtype='float32')
    # 转单声道
    if data.ndim > 1:
        data = data.mean(axis=1)
    # 重采样
    if sr != target_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, target_sr)
        data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
    return data


def _compute_fbank(signal: np.ndarray,
                   sample_rate: int = 16000,
                   num_mel_bins: int = 80,
                   frame_length_ms: float = 25.0,
                   frame_shift_ms: float = 10.0,
                   preemph: float = 0.97) -> np.ndarray:
    """
    纯 numpy/scipy Fbank 特征提取（Kaldi 兼容）
    返回 shape = (num_frames, num_mel_bins)，已做 CMN
    """
    from scipy.fftpack import fft
    
    # 预加重
    emphasized = np.append(signal[0], signal[1:] - preemph * signal[:-1])
    # 乘以 2^15 与 WeSpeaker 保持一致
    emphasized = emphasized * (1 << 15)

    # 帧参数
    frame_length = int(sample_rate * frame_length_ms / 1000)
    frame_shift = int(sample_rate * frame_shift_ms / 1000)
    n_fft = 1
    while n_fft < frame_length:
        n_fft <<= 1  # 下一个 2 的幂

    num_frames = max(1, 1 + (len(emphasized) - frame_length) // frame_shift)

    # Hamming 窗
    window = np.hamming(frame_length).astype(np.float32)

    # 分帧 + 加窗 + FFT
    frames = np.zeros((num_frames, n_fft), dtype=np.float32)
    for i in range(num_frames):
        start = i * frame_shift
        end = start + frame_length
        if end <= len(emphasized):
            frames[i, :frame_length] = emphasized[start:end] * window
        else:
            seg = emphasized[start:]
            frames[i, :len(seg)] = seg[:frame_length] * window[:len(seg)]

    # 功率谱
    spectrum = np.abs(fft(frames, n=n_fft, axis=1)[:, :n_fft // 2 + 1]) ** 2

    # Mel 滤波器组
    mel_filters = _mel_filterbank(num_mel_bins, n_fft, sample_rate)
    mel_spec = np.dot(spectrum, mel_filters.T)
    mel_spec = np.maximum(mel_spec, 1e-10)
    log_mel = np.log(mel_spec)

    # CMN（句子级别均值归一化）
    log_mel = log_mel - np.mean(log_mel, axis=0, keepdims=True)

    return log_mel.astype(np.float32)


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def _mel_filterbank(num_mel_bins: int, n_fft: int, sample_rate: int) -> np.ndarray:
    """生成 Mel 滤波器组，shape = (num_mel_bins, n_fft//2+1)"""
    low_freq_mel = 0
    high_freq_mel = _hz_to_mel(sample_rate / 2)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_mel_bins + 2)
    hz_points = _mel_to_hz(mel_points)

    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    n_freqs = n_fft // 2 + 1
    filters = np.zeros((num_mel_bins, n_freqs), dtype=np.float32)

    for m in range(num_mel_bins):
        f_left = bin_points[m]
        f_center = bin_points[m + 1]
        f_right = bin_points[m + 2]

        for k in range(f_left, f_center):
            if f_center != f_left:
                filters[m, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right != f_center:
                filters[m, k] = (f_right - k) / (f_right - f_center)

    return filters


# ============================================================
#  SpeakerVerifier - ONNX Runtime 版本
# ============================================================

class SpeakerVerifier:
    def __init__(self, threshold: Optional[float] = None, device: str = None, model_name: Optional[str] = None):
        """
        初始化声纹验证器（ONNX Runtime 轻量版）
        
        Args:
            threshold: 相似度阈值；不传时使用模型对应默认值
            device: 忽略（仅 CPU 推理）
        """
        requested_model = (model_name or os.getenv("SPEAKER_VERIFIER_MODEL", "eres2netv2")).strip().lower()
        aliases = {
            "eres2netv2": "eres2netv2",
            "eres2net-v2": "eres2netv2",
            "resnet34": "resnet34",
            "wespeaker": "resnet34",
        }
        if requested_model not in aliases:
            raise ValueError(f"不支持的声纹模型: {requested_model}")
        self.model_name = aliases[requested_model]
        self.model_id = (
            "iic/speech_eres2netv2_sv_zh-cn_16k-common@v1.0.2"
            if self.model_name == "eres2netv2"
            else "wespeaker/cnceleb-resnet34-LM"
        )
        default_threshold = 0.36 if self.model_name == "eres2netv2" else 0.25
        configured_threshold = os.getenv("SPEAKER_VERIFIER_THRESHOLD", "").strip()
        if threshold is not None:
            self.threshold = float(threshold)
        elif configured_threshold:
            self.threshold = float(configured_threshold)
        else:
            self.threshold = default_threshold
        self.target_embedding = None
        self.embeddings: List[np.ndarray] = []  # 存储多个样本的embedding
        self.is_enrolled = False
        self.min_samples = 8  # 需要至少8个样本才算注册完成
        self.speaker_name = None  # 说话人名称
        self.session = None
        self.model_output_dim: Optional[int] = None
        self.last_error: Optional[str] = None
        self.last_inference_ms: Optional[float] = None
        self.cpu_threads = max(1, int(os.getenv("SPEAKER_VERIFIER_CPU_THREADS", "4")))
        
        print(f"[SpeakerVerifier] 🎤 初始化 {self.model_name} ONNX 声纹模型 (CPU)...")
        self._load_onnx_model()
        self._warmup()
        print(f"[SpeakerVerifier] ✅ {self.model_name} ONNX 声纹模型加载完成（阈值 {self.threshold:.2f}）")
    
    def _load_onnx_model(self):
        """加载 WeSpeaker ONNX 模型"""
        import onnxruntime as ort
        
        project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        if self.model_name == "eres2netv2":
            possible_paths = [
                os.path.join(project_root, 'models', 'eres2netv2-cn', 'eres2netv2_cn_common.onnx'),
                "/opt/voice-server/models/eres2netv2-cn/eres2netv2_cn_common.onnx",
            ]
        else:
            possible_paths = [
                os.path.join(project_root, 'models', 'wespeaker-cnceleb', 'cnceleb_resnet34_LM.onnx'),
                "/opt/voice-server/models/wespeaker-cnceleb/cnceleb_resnet34_LM.onnx",
            ]
        
        model_path = None
        for path in possible_paths:
            path = os.path.normpath(path)
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError(
                f"WeSpeaker ONNX 模型未找到，请将模型放在以下路径之一:\n"
                + "\n".join(f"  - {os.path.normpath(p)}" for p in possible_paths)
            )
        
        print(f"[SpeakerVerifier] 📦 加载模型: {model_path} ({os.path.getsize(model_path) / 1024 / 1024:.1f}MB)")
        
        # 配置 ONNX Runtime（CPU 优化）
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = self.cpu_threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, sess_options=so, providers=['CPUExecutionProvider'])
        
        # 验证模型 IO
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        print(f"[SpeakerVerifier] 📐 模型输入: {inputs[0].name} {inputs[0].shape}")
        print(f"[SpeakerVerifier] 📐 模型输出: {outputs[0].name} {outputs[0].shape}")
        output_shape = outputs[0].shape
        if output_shape and isinstance(output_shape[-1], (int, np.integer)):
            self.model_output_dim = int(output_shape[-1])
        
        SPEAKER_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _compute_model_features(self, signal: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """按模型训练时使用的前端提取 80 维 Fbank。"""
        if self.model_name == "eres2netv2":
            import kaldi_native_fbank as knf

            options = knf.FbankOptions()
            options.frame_opts.samp_freq = sample_rate
            options.frame_opts.dither = 0
            options.mel_opts.num_bins = 80
            fbank = knf.OnlineFbank(options)
            fbank.accept_waveform(sample_rate, np.asarray(signal, dtype=np.float32).tolist())
            fbank.input_finished()
            feats = np.asarray(
                [fbank.get_frame(index) for index in range(fbank.num_frames_ready)],
                dtype=np.float32,
            )
            feats -= feats.mean(axis=0, keepdims=True)
            return feats
        return _compute_fbank(signal, sample_rate=sample_rate, num_mel_bins=80)

    def _run_embedding(self, feats: np.ndarray) -> np.ndarray:
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        started = time.perf_counter()
        embeddings = self.session.run([output_name], {input_name: feats[np.newaxis, :, :]})
        self.last_inference_ms = (time.perf_counter() - started) * 1000
        return embeddings[0].squeeze()

    def _warmup(self) -> None:
        """启动时完成特征库导入和一次推理，避免首个真实请求承受冷启动。"""
        started = time.perf_counter()
        silence = np.zeros(16000, dtype=np.float32)
        feats = self._compute_model_features(silence, sample_rate=16000)
        self._run_embedding(feats)
        print(
            f"[SpeakerVerifier] ⚡ 预热完成: {(time.perf_counter() - started) * 1000:.1f}ms "
            f"(CPU threads={self.cpu_threads})"
        )
    
    def _extract_embedding(self, audio_path: str) -> np.ndarray:
        """从音频文件提取 speaker embedding（纯 numpy，无 PyTorch）"""
        # 加载音频
        signal = _load_audio(audio_path, target_sr=16000)
        
        # 提取 Fbank 特征
        feats = self._compute_model_features(signal, sample_rate=16000)
        return self._run_embedding(feats)
    
    def _extract_embedding_from_numpy(self, audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """从 numpy array 直接提取 embedding（避免磁盘 IO）"""
        # 确保 float32 单声道
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        
        # 重采样
        if sample_rate != 16000:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sample_rate, 16000)
            audio_data = resample_poly(audio_data, 16000 // g, sample_rate // g).astype(np.float32)
        
        # 使用与模型训练一致的 Fbank 前端，再执行 ONNX 推理。
        feats = self._compute_model_features(audio_data, sample_rate=16000)
        return self._run_embedding(feats)

    def _get_embedding_dim(self, embedding: Optional[np.ndarray]) -> Optional[int]:
        if embedding is None or getattr(embedding, 'ndim', 0) == 0:
            return None
        return int(embedding.shape[-1])

    def _disable_enrollment(self, message: str):
        self.last_error = message
        print(f"[SpeakerVerifier] ⚠️ enrollment_disabled error_chars={len(message)}")
        self.target_embedding = None
        self.embeddings = []
        self.is_enrolled = False
        self.speaker_name = None
    
    def add_sample(self, audio_path: str) -> Tuple[bool, int, int]:
        """
        添加一个声纹样本
        Returns: (是否成功, 当前样本数, 需要样本数)
        """
        try:
            embedding = self._extract_embedding(audio_path)
            self.last_error = None
            self.embeddings.append(embedding)
            
            current = len(self.embeddings)
            needed = self.min_samples
            
            print(f"[SpeakerVerifier] 📝 添加样本 {current}/{needed}")
            
            # 达到最小样本数，计算平均embedding
            if current >= needed:
                self.target_embedding = np.mean(self.embeddings, axis=0)
                self.is_enrolled = True
                print(f"[SpeakerVerifier] ✅ 声纹注册完成 ({current}个样本)")
            
            return True, current, needed
        except Exception as e:
            print(f"[SpeakerVerifier] ❌ 添加样本失败: {type(e).__name__}")
            return False, len(self.embeddings), self.min_samples
    
    def enroll(self, audio_path: str) -> bool:
        """兼容旧接口：单样本注册"""
        success, current, needed = self.add_sample(audio_path)
        return success and self.is_enrolled
    
    def verify(self, audio_path: str) -> Tuple[bool, float]:
        """
        验证音频是否属于已注册的说话人
        
        Returns:
            (是否匹配, 相似度分数)
        """
        if not self.is_enrolled:
            self.last_error = "声纹未注册或未加载"
            return False, float('nan')
        try:
            self.last_error = None
            test_emb = self._extract_embedding(audio_path)
            target_dim = self._get_embedding_dim(self.target_embedding)
            test_dim = self._get_embedding_dim(test_emb)
            if target_dim is not None and test_dim is not None and target_dim != test_dim:
                self._disable_enrollment(f"已加载声纹维度 {target_dim} 与当前模型输出维度 {test_dim} 不一致，请重新注册或重新加载声纹")
                return False, float('nan')
            
            # 计算余弦相似度
            sim = float(np.dot(self.target_embedding, test_emb) / 
                       (np.linalg.norm(self.target_embedding) * np.linalg.norm(test_emb)))
            
            is_target = sim >= self.threshold
            print(f"[SpeakerVerifier] {'✅' if is_target else '❌'} 相似度: {sim:.3f} (阈值: {self.threshold})")
            return is_target, sim
        except Exception as e:
            self.last_error = type(e).__name__
            print(f"[SpeakerVerifier] ⚠️ 验证出错: {self.last_error}")
            return False, float('nan')
    
    def verify_from_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Tuple[bool, float]:
        """从 numpy array 直接验证（零磁盘 IO，更快）"""
        if not self.is_enrolled:
            self.last_error = "声纹未注册或未加载"
            return False, float('nan')
        try:
            self.last_error = None
            test_emb = self._extract_embedding_from_numpy(audio_data, sample_rate)
            target_dim = self._get_embedding_dim(self.target_embedding)
            test_dim = self._get_embedding_dim(test_emb)
            if target_dim is not None and test_dim is not None and target_dim != test_dim:
                self._disable_enrollment(f"已加载声纹维度 {target_dim} 与当前模型输出维度 {test_dim} 不一致，请重新注册或重新加载声纹")
                return False, float('nan')
            sim = float(np.dot(self.target_embedding, test_emb) /
                       (np.linalg.norm(self.target_embedding) * np.linalg.norm(test_emb)))
            is_target = sim >= self.threshold
            print(f"[SpeakerVerifier] {'✅' if is_target else '❌'} 相似度: {sim:.3f} (阈值: {self.threshold})")
            return is_target, sim
        except Exception as e:
            self.last_error = type(e).__name__
            print(f"[SpeakerVerifier] ⚠️ 验证出错: {self.last_error}")
            return False, float('nan')
    
    def get_status(self) -> dict:
        """获取注册状态"""
        return {
            "is_enrolled": self.is_enrolled,
            "current_samples": len(self.embeddings),
            "min_samples": self.min_samples,
            "threshold": self.threshold,
            "model_name": self.model_name,
            "model_id": self.model_id,
            "cpu_threads": self.cpu_threads,
            "model_output_dim": self.model_output_dim,
            "last_inference_ms": self.last_inference_ms,
            "last_error": self.last_error,
        }
    
    def reset(self):
        self.target_embedding = None
        self.embeddings = []
        self.is_enrolled = False
        self.speaker_name = None
        print("[SpeakerVerifier] 🔄 声纹已重置")
    
    def save(self, speaker_name: str) -> bool:
        """
        保存声纹数据到文件
        
        Args:
            speaker_name: 说话人名称
            
        Returns:
            是否保存成功
        """
        if not self.is_enrolled or self.target_embedding is None:
            print("[SpeakerVerifier] ⚠️ 未注册声纹，无法保存")
            return False
        
        try:
            self.speaker_name = speaker_name
            # 文件名使用说话人名称
            safe_name = speaker_name.replace("/", "_").replace("\\", "_")
            file_path = SPEAKER_DATA_DIR / f"{safe_name}.npz"
            
            # 保存embedding和元数据
            np.savez(
                file_path,
                target_embedding=self.target_embedding,
                embeddings=np.array(self.embeddings),
                embedding_dim=self._get_embedding_dim(self.target_embedding),
                model_id=self.model_id,
                threshold=self.threshold,
                speaker_name=speaker_name
            )
            
            print(f"[SpeakerVerifier] 💾 声纹已保存 samples={len(self.embeddings)}")
            return True
        except Exception as e:
            print(f"[SpeakerVerifier] ❌ 保存失败: {type(e).__name__}")
            return False
    
    def load(self, speaker_name: str) -> bool:
        """
        从文件加载声纹数据
        
        Args:
            speaker_name: 说话人名称
            
        Returns:
            是否加载成功
        """
        try:
            safe_name = speaker_name.replace("/", "_").replace("\\", "_")
            file_path = SPEAKER_DATA_DIR / f"{safe_name}.npz"
            
            if not file_path.exists():
                self._disable_enrollment("声纹文件不存在")
                return False
            
            data = np.load(file_path, allow_pickle=True)
            target_embedding = data['target_embedding']
            saved_model_id = str(data['model_id']) if 'model_id' in data.files else ""
            if saved_model_id != self.model_id:
                previous = saved_model_id or "旧版未知模型"
                self._disable_enrollment(
                    f"声纹由 {previous} 生成，与当前 {self.model_id} 不兼容，请重新注册声纹"
                )
                return False
            loaded_dim = self._get_embedding_dim(target_embedding)
            if self.model_output_dim is not None and loaded_dim is not None and loaded_dim != self.model_output_dim:
                self._disable_enrollment(
                    f"声纹文件维度 {loaded_dim} 与当前模型维度 {self.model_output_dim} 不一致，请重新注册声纹"
                )
                return False
            self.target_embedding = target_embedding
            self.embeddings = list(data['embeddings'])
            # 注意：不覆盖当前阈值，让用户可以动态调节
            # self.threshold = float(data['threshold'])
            self.speaker_name = str(data['speaker_name'])
            self.is_enrolled = True
            self.last_error = None
            
            print(f"[SpeakerVerifier] ✅ 已加载声纹 samples={len(self.embeddings)}, 阈值: {self.threshold}")
            return True
        except Exception as e:
            self._disable_enrollment(f"加载声纹失败: {type(e).__name__}")
            return False
    
    @staticmethod
    def list_saved_speakers() -> List[str]:
        """获取所有已保存的说话人列表"""
        SPEAKER_DATA_DIR.mkdir(parents=True, exist_ok=True)
        speakers = []
        for f in SPEAKER_DATA_DIR.glob("*.npz"):
            try:
                data = np.load(f, allow_pickle=True)
                name = str(data['speaker_name'])
                speakers.append(name)
            except:
                pass
        return speakers


_verifier = None

def get_speaker_verifier(threshold: Optional[float] = None) -> SpeakerVerifier:
    """
    获取声纹验证器单例
    
    Args:
        threshold: 相似度阈值；不传时读取环境变量或使用模型默认值
    """
    global _verifier
    if _verifier is None:
        _verifier = SpeakerVerifier(threshold=threshold)
    return _verifier
