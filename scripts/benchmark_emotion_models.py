"""
中文语音情绪识别模型对比实验
目标：为心理健康对话系统选择最优情绪识别模型
"""

import sys
sys.path.insert(0, '.')

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import psutil
import os


@dataclass
class BenchmarkResult:
    """单个模型的评估结果"""
    model_name: str
    accuracy: float
    f1_weighted: float
    f1_macro: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    model_size_mb: float
    memory_usage_mb: float
    classification_report: dict
    confusion_matrix: list
    per_age_group_accuracy: dict  # 新增：分年龄段准确率


class EmotionModelBenchmark:
    """情绪模型对比实验框架"""

    EMOTION_LABELS = ['joy', 'sadness', 'anger', 'fear', 'anxiety', 'calm', 'confusion']

    # 9类到7类的映射（Emotion2Vec+）
    EMOTION9_TO_7 = {
        'angry': 'anger',
        'disgusted': 'anger',
        'fearful': 'fear',
        'happy': 'joy',
        'neutral': 'calm',
        'sad': 'sadness',
        'surprised': 'confusion',
        'worried': 'anxiety',
        'anxious': 'anxiety',
    }

    def __init__(self, testset_dir: str):
        self.testset_dir = Path(testset_dir)
        self.test_samples = []
        self.results = {}

    def load_testset(self):
        """加载测试集"""
        manifest_path = self.testset_dir / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"测试集清单文件不存在: {manifest_path}\n"
                f"请先运行: python scripts/prepare_emotion_testset.py"
            )

        manifest = json.load(open(manifest_path, encoding='utf-8'))
        self.test_samples = manifest['samples']

        print(f"✅ 加载测试集: {len(self.test_samples)} 条样本")

        # 统计年龄分布
        age_dist = {}
        for s in self.test_samples:
            age = s.get('age_group', 'unknown')
            age_dist[age] = age_dist.get(age, 0) + 1

        print(f"   年龄分布: {age_dist}")

        # 统计情绪分布
        emotion_dist = {}
        for s in self.test_samples:
            emo = s['true_emotion']
            emotion_dist[emo] = emotion_dist.get(emo, 0) + 1

        print(f"   情绪分布: {emotion_dist}\n")

    def _get_memory_usage_mb(self) -> float:
        """获取当前进程内存占用（MB）"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def _get_model_size_mb(self, model_path: str) -> float:
        """获取模型文件大小（MB）"""
        if not os.path.exists(model_path):
            return 0.0

        total_size = 0
        for root, dirs, files in os.walk(model_path):
            for f in files:
                fp = os.path.join(root, f)
                total_size += os.path.getsize(fp)

        return total_size / 1024 / 1024

    def benchmark_emotion2vec(self) -> BenchmarkResult:
        """
        测试 Emotion2Vec+ Large
        - 模型：iic/emotion2vec_plus_large
        - 9类情绪 → 映射到7类
        """
        print("=" * 70)
        print("【模型1】Emotion2Vec+ Large")
        print("=" * 70)

        from src.tools.emotion import get_audio_emotion_classifier

        print("加载模型...")
        mem_before = self._get_memory_usage_mb()
        start_load = time.time()

        classifier = get_audio_emotion_classifier()

        warmup_audio = next(
            (
                sample["audio_path"]
                for sample in self.test_samples
                if os.path.isfile(sample.get("audio_path", ""))
            ),
            None,
        )
        if warmup_audio is None:
            raise FileNotFoundError("测试集没有可用的音频文件")
        classifier.classify_audio(warmup_audio)
        if not classifier.model_available:
            raise RuntimeError(
                "Emotion2Vec+ 模型不可用: "
                f"{classifier.last_error or '请检查 USE_MODELSCOPE 和模型依赖'}"
            )

        load_time = time.time() - start_load
        mem_after = self._get_memory_usage_mb()
        mem_usage = mem_after - mem_before

        model_path = os.getenv('EMOTION2VEC_MODEL', 'iic/emotion2vec_plus_large')
        model_size = self._get_model_size_mb(model_path)

        print(f"✅ 模型加载完成")
        print(f"   耗时: {load_time:.2f}秒")
        print(f"   内存增量: {mem_usage:.1f} MB")
        print(f"   模型大小: {model_size:.1f} MB\n")

        predictions = []
        true_labels = []
        latencies = []
        age_group_results = {}  # {age_group: {'correct': N, 'total': N}}

        print("开始推理...")
        for i, sample in enumerate(self.test_samples):
            audio_path = sample['audio_path']
            true_emotion = sample['true_emotion']
            age_group = sample.get('age_group', 'unknown')

            if not audio_path or not os.path.isfile(audio_path):
                print(f"   ⚠️ 样本 {i+1} 音频不存在，跳过")
                continue

            # 推理
            start = time.perf_counter()
            try:
                emotion_scores = classifier.classify_audio(audio_path)
                latency = (time.perf_counter() - start) * 1000

                if not classifier.model_available:
                    raise RuntimeError(
                        classifier.last_error or "Emotion2Vec+ 推理后不可用"
                    )

                predicted = max(emotion_scores, key=emotion_scores.get)
                predictions.append(predicted)
                true_labels.append(true_emotion)
                latencies.append(latency)

                # 统计分年龄段准确率
                if age_group not in age_group_results:
                    age_group_results[age_group] = {'correct': 0, 'total': 0}

                age_group_results[age_group]['total'] += 1
                if predicted == true_emotion:
                    age_group_results[age_group]['correct'] += 1

                if (i + 1) % 10 == 0:
                    print(f"   进度: {i+1}/{len(self.test_samples)}")

            except Exception as e:
                print(f"   ⚠️ 样本 {i+1} 失败: {e}")
                continue

        print(f"✅ 推理完成\n")

        if not predictions:
            raise RuntimeError("没有成功完成的 Emotion2Vec+ 推理")

        # 计算指标
        accuracy = accuracy_score(true_labels, predictions)
        f1_weighted = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0)

        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)

        # 分年龄段准确率
        per_age_accuracy = {}
        for age, stats in age_group_results.items():
            if stats['total'] > 0:
                per_age_accuracy[age] = stats['correct'] / stats['total']

        result = BenchmarkResult(
            model_name="Emotion2Vec+ Large",
            accuracy=accuracy,
            f1_weighted=f1_weighted,
            f1_macro=f1_macro,
            avg_latency_ms=sum(latencies) / len(latencies),
            p50_latency_ms=latencies_sorted[int(n * 0.5)],
            p95_latency_ms=latencies_sorted[int(n * 0.95)],
            p99_latency_ms=latencies_sorted[int(n * 0.99)],
            model_size_mb=model_size,
            memory_usage_mb=mem_usage,
            classification_report=classification_report(
                true_labels, predictions,
                labels=self.EMOTION_LABELS,
                target_names=self.EMOTION_LABELS,
                output_dict=True,
                zero_division=0
            ),
            confusion_matrix=confusion_matrix(
                true_labels, predictions,
                labels=self.EMOTION_LABELS
            ).tolist(),
            per_age_group_accuracy=per_age_accuracy
        )

        self._print_result(result)
        return result

    def benchmark_wav2vec2_cn(self) -> BenchmarkResult:
        """
        测试 Wav2Vec2-Emotion-CN
        （备选方案，如果 Emotion2Vec 不满足需求）
        """
        print("=" * 70)
        print("【模型2】Wav2Vec2-Emotion-CN")
        print("=" * 70)
        print("⚠️ 未实现，需要安装额外依赖\n")

        # TODO: 实现 Wav2Vec2 测试
        # from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

        return None

    def benchmark_mfcc_svm(self) -> BenchmarkResult:
        """
        测试传统方法：MFCC + SVM
        （baseline，用于对比）
        """
        print("=" * 70)
        print("【模型3】MFCC + SVM (Baseline)")
        print("=" * 70)
        print("⚠️ 未实现，需要训练 SVM 分类器\n")

        # TODO: 实现传统方法
        # 1. 提取 MFCC 特征
        # 2. 加载预训练 SVM
        # 3. 预测

        return None

    def _print_result(self, result: BenchmarkResult):
        """打印单个模型的结果"""
        print(f"📊 {result.model_name} - 评估结果")
        print("-" * 70)
        print(f"准确率 (Accuracy):     {result.accuracy*100:.2f}%")
        print(f"F1-Score (Weighted):   {result.f1_weighted:.4f}")
        print(f"F1-Score (Macro):      {result.f1_macro:.4f}")
        print()
        print(f"平均延迟:              {result.avg_latency_ms:.2f} ms")
        print(f"P50 延迟:              {result.p50_latency_ms:.2f} ms")
        print(f"P95 延迟:              {result.p95_latency_ms:.2f} ms")
        print(f"P99 延迟:              {result.p99_latency_ms:.2f} ms")
        print()
        print(f"模型大小:              {result.model_size_mb:.1f} MB")
        print(f"内存占用:              {result.memory_usage_mb:.1f} MB")
        print()

        # 分年龄段准确率
        if result.per_age_group_accuracy:
            print("分年龄段准确率:")
            for age, acc in result.per_age_group_accuracy.items():
                print(f"  {age:15s}: {acc*100:.2f}%")
            print()

        # 每类情绪的 F1
        print("各情绪类别 F1-Score:")
        for emotion in self.EMOTION_LABELS:
            if emotion in result.classification_report:
                f1 = result.classification_report[emotion]['f1-score']
                support = int(result.classification_report[emotion]['support'])
                print(f"  {emotion:12s}: {f1:.4f}  (样本数: {support})")

        print("=" * 70)
        print()

    def run_all(self, models: List[str] = None):
        """
        运行所有对比实验

        Args:
            models: 要测试的模型列表，默认全部
                    ['emotion2vec', 'wav2vec2', 'mfcc_svm']
        """
        if models is None:
            models = ['emotion2vec']  # 默认只测 emotion2vec

        results = {}

        if 'emotion2vec' in models:
            try:
                results['emotion2vec'] = self.benchmark_emotion2vec()
            except Exception as e:
                print(f"❌ Emotion2Vec 测试失败: {e}\n")

        if 'wav2vec2' in models:
            results['wav2vec2'] = self.benchmark_wav2vec2_cn()

        if 'mfcc_svm' in models:
            results['mfcc_svm'] = self.benchmark_mfcc_svm()

        self.results = results
        return results

    def save_results(self, output_path: str):
        """保存结果到 JSON"""
        output = {}

        for model_name, result in self.results.items():
            if result is None:
                continue

            output[model_name] = {
                'model_name': result.model_name,
                'accuracy': result.accuracy,
                'f1_weighted': result.f1_weighted,
                'f1_macro': result.f1_macro,
                'avg_latency_ms': result.avg_latency_ms,
                'p50_latency_ms': result.p50_latency_ms,
                'p95_latency_ms': result.p95_latency_ms,
                'p99_latency_ms': result.p99_latency_ms,
                'model_size_mb': result.model_size_mb,
                'memory_usage_mb': result.memory_usage_mb,
                'classification_report': result.classification_report,
                'confusion_matrix': result.confusion_matrix,
                'per_age_group_accuracy': result.per_age_group_accuracy
            }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"✅ 结果已保存: {output_path}")

    def generate_comparison_table(self) -> str:
        """生成对比表格（Markdown）"""
        valid_results = {k: v for k, v in self.results.items() if v is not None}

        if not valid_results:
            return "无有效结果"

        table = "| 指标 | " + " | ".join([r.model_name for r in valid_results.values()]) + " |\n"
        table += "|" + "---|" * (len(valid_results) + 1) + "\n"

        # 准确率
        table += "| 准确率 | " + " | ".join([
            f"{r.accuracy*100:.2f}%" for r in valid_results.values()
        ]) + " |\n"

        # F1 (Weighted)
        table += "| F1-Score (Weighted) | " + " | ".join([
            f"{r.f1_weighted:.4f}" for r in valid_results.values()
        ]) + " |\n"

        # 平均延迟
        table += "| 平均延迟 (ms) | " + " | ".join([
            f"{r.avg_latency_ms:.2f}" for r in valid_results.values()
        ]) + " |\n"

        # P95 延迟
        table += "| P95 延迟 (ms) | " + " | ".join([
            f"{r.p95_latency_ms:.2f}" for r in valid_results.values()
        ]) + " |\n"

        # 模型大小
        table += "| 模型大小 (MB) | " + " | ".join([
            f"{r.model_size_mb:.1f}" for r in valid_results.values()
        ]) + " |\n"

        # 内存占用
        table += "| 内存占用 (MB) | " + " | ".join([
            f"{r.memory_usage_mb:.1f}" for r in valid_results.values()
        ]) + " |\n"

        return table


def main():
    import argparse

    parser = argparse.ArgumentParser(description="中文语音情绪识别模型对比实验")
    parser.add_argument(
        '--testset',
        type=str,
        default='tests/emotion_benchmark/testset',
        help='测试集目录'
    )
    parser.add_argument(
        '--models',
        type=str,
        default='emotion2vec',
        help='要测试的模型（逗号分隔），如: emotion2vec,wav2vec2,mfcc_svm'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='tests/emotion_benchmark/results.json',
        help='结果输出路径'
    )

    args = parser.parse_args()

    models_to_test = args.models.split(',')

    print("=" * 70)
    print("中文语音情绪识别模型对比实验")
    print("=" * 70)
    print(f"测试集: {args.testset}")
    print(f"模型: {', '.join(models_to_test)}")
    print(f"输出: {args.output}")
    print("=" * 70)
    print()

    benchmark = EmotionModelBenchmark(args.testset)
    benchmark.load_testset()

    benchmark.run_all(models=models_to_test)

    benchmark.save_results(args.output)

    print("\n" + "=" * 70)
    print("对比总结")
    print("=" * 70)
    print(benchmark.generate_comparison_table())
    print()


if __name__ == "__main__":
    main()
