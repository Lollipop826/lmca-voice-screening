"""
ZipVoice 高质量语音合成工具
基于 Flow Matching 的快速、高质量零样本 TTS

特点：
- 仅 123M 参数，速度快
- 高质量语音克隆 (SOTA 性能)
- 支持中英文
- 原生 24kHz 采样率
- 蒸馏版本仅需 4-8 步推理
"""

import asyncio
import os
import sys
import numpy as np
from pathlib import Path
import hashlib
import time
try:
    import torch
    _inference_mode = torch.inference_mode()
except ImportError:
    torch = None
    _inference_mode = lambda f: f  # no-op decorator when torch unavailable
from typing import Optional, Generator, AsyncGenerator

# 添加 ZipVoice 到路径
_zipvoice_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ZipVoice-master'))
if _zipvoice_root not in sys.path:
    sys.path.insert(0, _zipvoice_root)

# 尝试导入 ZipVoice 依赖
ZIPVOICE_AVAILABLE = False
try:
    from huggingface_hub import hf_hub_download
    from lhotse.utils import fix_random_seed
    from vocos import Vocos
    import json
    ZIPVOICE_AVAILABLE = True
except ImportError as e:
    print(f"[ZipVoice] ⚠️ 部分依赖缺失: {type(e).__name__}")
    print("[ZipVoice] 请运行: pip install vocos lhotse huggingface_hub safetensors")

# 延迟导入 ZipVoice 模块
ZIPVOICE_MODEL_AVAILABLE = False
try:
    from zipvoice.models.zipvoice import ZipVoice as ZipVoiceModel
    from zipvoice.models.zipvoice_distill import ZipVoiceDistill
    from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
    from zipvoice.utils.checkpoint import load_checkpoint
    from zipvoice.utils.feature import VocosFbank
    from zipvoice.utils.infer import (
        add_punctuation,
        batchify_tokens,
        chunk_tokens_punctuation,
        cross_fade_concat,
        load_prompt_wav,
        remove_silence,
        rms_norm,
    )
    ZIPVOICE_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"[ZipVoice] ⚠️ ZipVoice 模块导入失败: {type(e).__name__}")


HUGGINGFACE_REPO = "k2-fsa/ZipVoice"
MODEL_DIR = {
    "zipvoice": "zipvoice",
    "zipvoice_distill": "zipvoice_distill",
}


class ZipVoiceTTS:
    """
    ZipVoice 语音合成
    
    与 CosyVoiceTTS 接口兼容，可直接替换
    """
    
    # 情感映射（ZipVoice 暂不支持情感，但保持接口兼容）
    SPEAKERS = {
        "neutral": "default",
        "gentle": "default",
        "professional": "default",
        "happy": "default",
        "sad": "default",
        "angry": "default",
        "fearful": "default"
    }
    
    def __init__(
        self,
        model_dir: str = None,
        device: str = None,
        model_name: str = "zipvoice_distill",
        num_step: int = 4,
        prompt_wav: str = None,
        prompt_text: str = None,
    ):
        """
        初始化 ZipVoice TTS
        
        Args:
            model_dir: 本地模型目录（如果为 None，从 HuggingFace 下载）
            device: 设备，例如 'cuda:0' 或 'cpu'
            model_name: 模型版本 ('zipvoice' 或 'zipvoice_distill')
            num_step: 推理步数（蒸馏版推荐 4-8）
            prompt_wav: 参考音频路径（用于语音克隆）
            prompt_text: 参考音频的转录文本
        """
        if not ZIPVOICE_AVAILABLE:
            raise ImportError("[ZipVoice] 依赖未安装，请运行: pip install vocos lhotse huggingface_hub safetensors")
        
        if not ZIPVOICE_MODEL_AVAILABLE:
            raise ImportError("[ZipVoice] 模块未找到，请确保 ZipVoice-master 在正确路径")
        
        print(f"[ZipVoice] 🚀 初始化 {model_name}...")
        
        self.model_name = model_name
        self.sample_rate = 24000  # ZipVoice 原生 24kHz
        self.num_step = num_step
        self.feat_scale = 0.1
        self.t_shift = 0.5
        self.target_rms = 0.1
        
        # 设置默认参数
        model_defaults = {
            "zipvoice": {"num_step": 16, "guidance_scale": 1.0},
            "zipvoice_distill": {"num_step": 8, "guidance_scale": 3.0},
        }
        defaults = model_defaults.get(model_name, model_defaults["zipvoice_distill"])
        self.num_step = num_step if num_step is not None else defaults["num_step"]
        self.guidance_scale = defaults["guidance_scale"]
        
        # 设置设备
        if device:
            self.device = torch.device(device.replace('gpu:', 'cuda:'))
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        print(f"[ZipVoice] 设备: {self.device}")
        
        # 加载模型
        self._load_model(model_name, model_dir)
        
        # 加载 vocoder
        self._load_vocoder()
        
        # 初始化特征提取器
        self.feature_extractor = VocosFbank()
        
        # 设置参考音频
        self.prompt_wav_path = prompt_wav
        self.prompt_text = prompt_text or "你好。"
        self.prompt_features = None
        self.prompt_rms = None
        self.prompt_tokens = None
        
        # 如果提供了参考音频，预处理它
        if prompt_wav and os.path.exists(prompt_wav):
            self._prepare_prompt(prompt_wav, prompt_text)
        else:
            # 使用默认参考
            self._create_default_prompt()
        
        # 预热模型
        self._warmup()
        
        print(f"[ZipVoice] ✅ 初始化完成")
        print(f"  - 模型: {model_name}")
        print(f"  - 推理步数: {self.num_step}")
        print(f"  - 采样率: {self.sample_rate}Hz")
    
    def _load_model(self, model_name: str, model_dir: Optional[str]):
        """加载 ZipVoice 模型"""
        if model_dir and os.path.isdir(model_dir):
            # 从本地加载
            model_ckpt = os.path.join(model_dir, "model.pt")
            model_config = os.path.join(model_dir, "model.json")
            token_file = os.path.join(model_dir, "tokens.txt")
            print(f"[ZipVoice] 从本地加载: {model_dir}")
        else:
            # 从 HuggingFace 下载
            print(f"[ZipVoice] 📥 从 HuggingFace 下载模型...")
            model_ckpt = hf_hub_download(
                HUGGINGFACE_REPO, 
                filename=f"{MODEL_DIR[model_name]}/model.pt"
            )
            model_config = hf_hub_download(
                HUGGINGFACE_REPO, 
                filename=f"{MODEL_DIR[model_name]}/model.json"
            )
            token_file = hf_hub_download(
                HUGGINGFACE_REPO, 
                filename=f"{MODEL_DIR[model_name]}/tokens.txt"
            )
            print(f"[ZipVoice] ✅ 模型下载完成")
        
        # 初始化 tokenizer
        self.tokenizer = EmiliaTokenizer(token_file=token_file)
        tokenizer_config = {
            "vocab_size": self.tokenizer.vocab_size,
            "pad_id": self.tokenizer.pad_id
        }
        
        # 加载模型配置
        with open(model_config, "r") as f:
            config = json.load(f)
        
        # 创建模型
        if model_name == "zipvoice":
            self.model = ZipVoiceModel(**config["model"], **tokenizer_config)
        else:
            self.model = ZipVoiceDistill(**config["model"], **tokenizer_config)
        
        # 加载权重
        load_checkpoint(filename=model_ckpt, model=self.model, strict=True)
        
        # 移到设备
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"[ZipVoice] ✅ 模型加载完成")
    
    def _load_vocoder(self):
        """加载 Vocos vocoder"""
        print(f"[ZipVoice] 📥 加载 Vocos vocoder...")
        # 尝试从本地加载，如果不存在则从 HuggingFace 下载
        local_vocoder_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'vocos-mel-24khz')
        local_vocoder_path = os.path.normpath(local_vocoder_path)
        
        if os.path.exists(local_vocoder_path):
            print(f"[ZipVoice] 从本地加载 Vocoder: {local_vocoder_path}")
            # 使用 from_hparams 从本地配置加载
            import yaml
            config_path = os.path.join(local_vocoder_path, "config.yaml")
            checkpoint_path = os.path.join(local_vocoder_path, "pytorch_model.bin")
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.vocoder = Vocos.from_hparams(config_path)
            # 加载权重
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.vocoder.load_state_dict(state_dict)
        else:
            print(f"[ZipVoice] 从 HuggingFace 下载 Vocoder...")
            self.vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        
        self.vocoder = self.vocoder.to(self.device)
        self.vocoder.eval()
        print(f"[ZipVoice] ✅ Vocoder 加载完成")
    
    def _create_default_prompt(self):
        """创建默认参考音频 - 使用固定的参考确保音色一致"""
        print(f"[ZipVoice] 🎙️ 加载默认参考音频...")
        
        # 默认参考音频路径列表（按优先级）
        configured_prompt = os.getenv("ZIPVOICE_PROMPT_WAV", "").strip()
        default_prompts = [
            path
            for path in (
                configured_prompt,
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "..",
                    "static",
                    "audio",
                    "参考音频.wav",
                ),
            )
            if path
        ]
        
        # 尝试加载参考音频
        for prompt_path in default_prompts:
            prompt_path = os.path.normpath(prompt_path)
            if os.path.exists(prompt_path):
                print(f"[ZipVoice] 使用参考音频: {prompt_path}")
                
                # 尝试读取同名的 .txt 文件作为转录文本
                txt_path = os.path.splitext(prompt_path)[0] + '.txt'
                prompt_text = "你好。"  # 默认
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        prompt_text_from_file = f.read().strip()
                    if prompt_text_from_file:
                        prompt_text = prompt_text_from_file
                        print(
                            "[ZipVoice] 使用转录文本 "
                            f"text_chars={len(prompt_text)}"
                        )
                
                try:
                    self._prepare_prompt(prompt_path, prompt_text)
                    print(f"[ZipVoice] ✅ 参考音频加载成功，音色将保持一致")
                    return
                except Exception as e:
                    print(f"[ZipVoice] ⚠️ 加载参考音频失败: {type(e).__name__}")
                    continue
        
        # 如果所有参考音频都不可用，使用备用方案
        print(f"[ZipVoice] ⚠️ 未找到参考音频，使用备用方案")
        print(f"[ZipVoice] 建议将参考音频放在: static/audio/参考音频.wav")
        print(f"[ZipVoice] 并创建对应的转录文件: static/audio/参考音频.txt")
        
        # 使用固定的随机种子确保每次生成相同的"噪声"
        torch.manual_seed(42)
        dummy_wav = torch.randn(self.sample_rate) * 0.01
        
        # 提取特征
        features = self.feature_extractor.extract(dummy_wav, sampling_rate=self.sample_rate)
        self.prompt_features = features.unsqueeze(0).to(self.device) * self.feat_scale
        
        # 处理参考文本
        self.prompt_text = "你好。"
        self.prompt_tokens = self.tokenizer.texts_to_token_ids([self.prompt_text])
        self.prompt_rms = self.target_rms
    
    def _prepare_prompt(self, prompt_wav: str, prompt_text: str):
        """预处理参考音频"""
        print(f"[ZipVoice] 预处理参考音频: {prompt_wav}")
        
        # 加载音频
        wav = load_prompt_wav(prompt_wav, sampling_rate=self.sample_rate)
        
        # 移除静音
        wav = remove_silence(wav, self.sample_rate, only_edge=False, trail_sil=200)
        
        # 音量归一化
        wav, self.prompt_rms = rms_norm(wav, self.target_rms)
        
        # 提取特征
        features = self.feature_extractor.extract(wav, sampling_rate=self.sample_rate)
        self.prompt_features = features.unsqueeze(0).to(self.device) * self.feat_scale
        
        # 处理参考文本
        self.prompt_text = add_punctuation(prompt_text) if prompt_text else "你好。"
        self.prompt_tokens = self.tokenizer.texts_to_token_ids([self.prompt_text])
        
        prompt_duration = wav.shape[-1] / self.sample_rate
        print(f"[ZipVoice] 参考音频时长: {prompt_duration:.2f}s")
        print(f"[ZipVoice] 参考文本 text_chars={len(self.prompt_text)}")
    
    def _warmup(self):
        """预热模型"""
        print(f"[ZipVoice] 🔥 预热模型...")
        try:
            # 生成一小段测试
            _ = self._generate_speech_sync("你好", speed=1.0)
            print(f"[ZipVoice] ✅ 预热完成")
        except Exception as e:
            print(f"[ZipVoice] ⚠️ 预热失败（不影响使用）: {type(e).__name__}")
    
    def _get_cache_path(self, text: str, emotion: str = "neutral") -> str:
        """基于文本生成缓存文件路径"""
        version_tag = f"zipvoice_{self.model_name}_v1"
        key = f"{version_tag}|{emotion}|{text}".encode("utf-8")
        digest = hashlib.sha1(key).hexdigest()
        temp_dir = Path(os.path.dirname(__file__)).parent.parent.parent / "tmp" / "ad_screening_voice"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return str(temp_dir / f"tts_{digest}.wav")
    
    @_inference_mode
    def _generate_speech_sync(self, text: str, speed: float = 1.0) -> np.ndarray:
        """同步生成语音"""
        if self.prompt_features is None:
            self._create_default_prompt()
        
        # 添加标点
        text = add_punctuation(text)
        
        # 分词
        tokens_str = self.tokenizer.texts_to_tokens([text])[0]
        prompt_tokens_str = self.tokenizer.texts_to_tokens([self.prompt_text])[0]
        
        # 计算 token 时长
        prompt_duration = self.prompt_features.shape[1] * 256 / self.sample_rate
        token_duration = max(0.01, prompt_duration / (len(prompt_tokens_str) * speed + 1e-6))
        
        # 分块
        max_tokens = max(10, int((25 - prompt_duration) / (token_duration + 1e-6)))
        chunked_tokens_str = chunk_tokens_punctuation(tokens_str, max_tokens=max_tokens)
        
        # 转为 token IDs
        chunked_tokens = self.tokenizer.tokens_to_token_ids(chunked_tokens_str)
        
        # 批处理
        tokens_batches, chunked_index = batchify_tokens(
            chunked_tokens, 100, prompt_duration, token_duration
        )
        
        # 生成
        all_wavs = []
        for batch_tokens in tokens_batches:
            batch_size = len(batch_tokens)
            
            batch_prompt_tokens = self.prompt_tokens * batch_size
            batch_prompt_features = self.prompt_features.repeat(batch_size, 1, 1)
            batch_prompt_features_lens = torch.full(
                (batch_size,), self.prompt_features.size(1), device=self.device
            )
            
            # 生成特征
            (
                pred_features,
                pred_features_lens,
                _,
                _,
            ) = self.model.sample(
                tokens=batch_tokens,
                prompt_tokens=batch_prompt_tokens,
                prompt_features=batch_prompt_features,
                prompt_features_lens=batch_prompt_features_lens,
                speed=speed,
                t_shift=self.t_shift,
                duration="predict",
                num_step=self.num_step,
                guidance_scale=self.guidance_scale,
            )
            
            # 后处理
            pred_features = pred_features.permute(0, 2, 1) / self.feat_scale
            
            # Vocoder 解码
            for i in range(batch_size):
                wav = (
                    self.vocoder.decode(pred_features[i:i+1, :, :pred_features_lens[i]])
                    .squeeze(1)
                    .clamp(-1, 1)
                )
                
                if self.prompt_rms and self.prompt_rms < self.target_rms:
                    wav = wav * self.prompt_rms / self.target_rms
                
                wav_np = wav.squeeze(0).cpu().numpy()
                all_wavs.append((chunked_index[len(all_wavs)], wav_np))
        
        # 排序并拼接
        all_wavs.sort(key=lambda x: x[0])
        wav_list = [w[1] for w in all_wavs]
        
        if len(wav_list) > 1:
            # 使用 cross_fade 拼接
            wav_tensors = [torch.from_numpy(w).unsqueeze(0) for w in wav_list]
            final_wav = cross_fade_concat(wav_tensors, fade_duration=0.1, sample_rate=self.sample_rate)
            final_wav = final_wav.squeeze(0).numpy()
        elif len(wav_list) == 1:
            final_wav = wav_list[0]
        else:
            final_wav = np.array([], dtype=np.float32)
        
        return final_wav
    
    def _split_and_merge_sentences(self, text: str, min_chars: int = 15) -> list:
        """智能分句+合并短句，减少推理次数
        
        策略：
        1. 按标点分句
        2. 将过短的句子与相邻句子合并，确保每个块 >= min_chars
        3. 目标：将 5 个短句合并为 2-3 个适中长度的块
        """
        import re
        # 按句子分割（保留标点）
        sentences = re.split(r'([。！？；;!?])', text)
        raw = []
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                raw.append(sentences[i] + sentences[i + 1])
            elif sentences[i].strip():
                raw.append(sentences[i])
        
        # 过滤空句
        raw = [s for s in raw if s.strip()]
        if not raw:
            return [text] if text.strip() else []
        
        # 合并短句
        merged = []
        buf = ""
        for s in raw:
            if buf:
                buf += s
                if len(buf) >= min_chars:
                    merged.append(buf)
                    buf = ""
            elif len(s) < min_chars:
                buf = s
            else:
                merged.append(s)
        if buf:
            # 剩余短句：如果 merged 非空就追加到最后一个，否则独立成块
            if merged:
                merged[-1] += buf
            else:
                merged.append(buf)
        
        return merged
    
    async def text_to_speech_streaming(
        self,
        text: str,
        emotion: str = "neutral"
    ) -> AsyncGenerator[np.ndarray, None]:
        """流式文本转语音（首句优先 + 后续并行生成）
        
        优化策略：
        1. 智能合并短句，减少推理次数（5句→2-3句）
        2. 首句立即生成并 yield，降低首包延迟
        3. 剩余句子并行生成，按顺序 yield
        
        Args:
            text: 要合成的文本
            emotion: 情感类型（暂不支持，保持接口兼容）
            
        Yields:
            音频数据块（numpy数组, float32, 24kHz）
        """
        t0 = time.time()
        print(f"[ZipVoice] 🎵 流式生成语音...")
        print(f"[ZipVoice] 文本 text_chars={len(text)}")
        
        try:
            # 🔥 智能分句 + 合并短句
            merged = self._split_and_merge_sentences(text, min_chars=15)
            
            if not merged:
                return
            
            print(f"[ZipVoice] 分块: {len(merged)} 块 ({[len(s) for s in merged]})")
            
            # ---- 首句立即生成 & yield（降低首包延迟） ----
            first = merged[0]
            print(f"[ZipVoice] 块 1/{len(merged)} text_chars={len(first)}")
            t1 = time.time()
            audio_first = await asyncio.to_thread(
                self._generate_speech_sync, first, 1.0
            )
            print(f"[ZipVoice] ✅ 块 1 完成 ({time.time()-t1:.2f}s)")
            yield audio_first.astype(np.float32)
            
            # ---- 剩余句子并行生成 ----
            rest = merged[1:]
            if rest:
                async def _gen(idx, sentence):
                    """在线程池中生成单句音频"""
                    t_s = time.time()
                    audio = await asyncio.to_thread(
                        self._generate_speech_sync, sentence, 1.0
                    )
                    print(
                        f"[ZipVoice] ✅ 块 {idx+2}/{len(merged)} 完成 "
                        f"({time.time()-t_s:.2f}s) text_chars={len(sentence)}"
                    )
                    return audio.astype(np.float32)
                
                # 并行启动所有剩余块
                tasks = [
                    asyncio.create_task(_gen(i, s))
                    for i, s in enumerate(rest)
                ]
                
                # 按原始顺序依次 yield（保证播放顺序正确）
                for task in tasks:
                    audio = await task
                    yield audio
            
            print(f"[ZipVoice] 🏁 全部完成，总耗时 {time.time()-t0:.2f}s")
                
        except Exception as e:
            print(f"[ZipVoice] ❌ 流式生成失败: {type(e).__name__}")
    
    async def text_to_speech_async(
        self,
        text: str,
        emotion: str = "neutral",
        speed: float = 1.0
    ) -> str:
        """异步文本转语音
        
        Args:
            text: 要合成的文本
            emotion: 情感类型
            speed: 语速
        
        Returns:
            生成的音频文件路径
        """
        # 检查缓存
        cache_path = self._get_cache_path(text, emotion)
        if os.path.exists(cache_path):
            print("[ZipVoice] 💾 使用缓存")
            return cache_path
        
        start_time = time.time()
        print(f"[ZipVoice] 🎙️ 正在生成语音...")
        print(f"[ZipVoice] 文本 text_chars={len(text)}")
        
        try:
            # 在线程池中执行同步生成
            audio = await asyncio.to_thread(
                self._generate_speech_sync,
                text,
                speed
            )
            
            # 保存音频
            import soundfile as sf
            sf.write(cache_path, audio, samplerate=self.sample_rate)
            
            elapsed = (time.time() - start_time) * 1000
            print(f"[ZipVoice] ✅ 生成完成，耗时: {elapsed:.1f}ms")
            
            return cache_path
            
        except Exception as e:
            print(f"[ZipVoice] ❌ 生成失败: {type(e).__name__}")
            raise
    
    def text_to_speech(self, text: str, emotion: str = "neutral", speed: float = 1.0) -> str:
        """同步文本转语音（兼容旧接口）
        
        Args:
            text: 要合成的文本
            emotion: 情感类型
            speed: 语速
        
        Returns:
            生成的音频文件路径
        """
        return asyncio.run(self.text_to_speech_async(text, emotion, speed))


# 兼容旧代码的别名
VoiceTTS = ZipVoiceTTS


# 测试函数
def test_zipvoice():
    """快速测试"""
    print("=" * 60)
    print("ZipVoice TTS 测试")
    print("=" * 60)
    
    tts = ZipVoiceTTS(
        model_name="zipvoice_distill",
        num_step=6
    )
    
    text = "你好，我是ZipVoice，一个快速高质量的语音合成系统。"
    print(f"\n测试文本 text_chars={len(text)}")
    
    start = time.time()
    output_path = tts.text_to_speech(text)
    elapsed = time.time() - start
    
    print(f"\n结果:")
    print(f"  - 输出文件: {output_path}")
    print(f"  - 生成耗时: {elapsed:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    test_zipvoice()
