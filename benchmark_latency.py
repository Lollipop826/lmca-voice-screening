"""
测量关键路径延迟
"""
import time
import asyncio
import numpy as np
from pathlib import Path

# Emotion2Vec 推理延迟测试
def benchmark_emotion2vec():
    """测量不同音频长度下的推理时间"""
    print("=== Emotion2Vec 推理延迟测试 ===")

    try:
        from funasr import AutoModel

        # 初始化模型
        model = AutoModel(
            model="iic/emotion2vec_plus_large",
            device="cpu"  # 如果有GPU改成cuda:0
        )

        # 模拟不同长度的音频（采样率16000）
        durations = [1.0, 2.0, 3.0, 4.0, 5.0]  # 秒
        results = []

        for duration in durations:
            # 生成随机音频数据
            samples = int(16000 * duration)
            audio = np.random.randn(samples).astype(np.float32)

            # 预热
            model.generate(audio, granularity="utterance")

            # 正式测试（10次取平均）
            latencies = []
            for _ in range(10):
                start = time.perf_counter()
                model.generate(audio, granularity="utterance")
                latencies.append(time.perf_counter() - start)

            avg = np.mean(latencies)
            p95 = np.percentile(latencies, 95)
            results.append({
                'duration': duration,
                'avg_ms': avg * 1000,
                'p95_ms': p95 * 1000
            })

            print(f"{duration}s音频: 平均 {avg*1000:.1f}ms, P95 {p95*1000:.1f}ms")

        # 关键判断：3秒音频能否在800ms内完成
        target_duration = 3.0
        target_result = [r for r in results if r['duration'] == target_duration][0]

        print(f"\n滚动分析可行性判断:")
        print(f"3秒音频 P95延迟: {target_result['p95_ms']:.1f}ms")
        if target_result['p95_ms'] < 800:
            print("✓ 可以每0.8秒触发一次分析")
        else:
            recommended_interval = target_result['p95_ms'] / 1000 * 1.2
            print(f"✗ 建议采样间隔改为 {recommended_interval:.1f}秒")

        return results

    except ImportError:
        print("未安装 funasr，跳过 Emotion2Vec 测试")
        print("安装命令: pip install funasr")
        return None
    except Exception as e:
        print(f"测试失败: {e}")
        return None


# Memobase 检索延迟测试
async def benchmark_memobase():
    """测量长期记忆检索延迟"""
    print("\n=== Memobase 检索延迟测试 ===")

    try:
        from app.services.memory.memobase_service import MemobaseService

        service = MemobaseService()

        # 测试查询（不同长度）
        queries = [
            "我很难过",  # 短
            "我最近总是失眠，感觉压力很大",  # 中
            "我和家人的关系一直不太好，总是会因为一些小事吵架，我不知道该怎么办"  # 长
        ]

        results = []

        for query in queries:
            latencies = []

            # 每个查询测5次
            for _ in range(5):
                start = time.perf_counter()
                await service.get_relevant_memories(
                    patient_id="test_patient",
                    query=query,
                    top_k=5
                )
                latencies.append(time.perf_counter() - start)

            avg = np.mean(latencies)
            p95 = np.percentile(latencies, 95)
            results.append({
                'query_len': len(query),
                'avg_ms': avg * 1000,
                'p95_ms': p95 * 1000
            })

            print(f"{len(query)}字查询: 平均 {avg*1000:.1f}ms, P95 {p95*1000:.1f}ms")

        # 判断提前检索价值
        typical_latency = results[1]['p95_ms']  # 用中等长度的P95
        print(f"\n提前检索价值判断:")
        print(f"典型检索 P95延迟: {typical_latency:.1f}ms")
        if typical_latency > 1000:
            print(f"✓ 提前检索能节省 ~{typical_latency/1000:.1f}秒，值得做")
        else:
            print(f"? 只能节省 {typical_latency:.1f}ms，收益有限")

        return results

    except ImportError as e:
        print(f"无法导入 MemobaseService: {e}")
        print("需要确保项目依赖已安装")
        return None
    except Exception as e:
        print(f"测试失败: {e}")
        return None


async def main():
    print("开始性能基准测试\n")
    print("=" * 50)

    # 测试1: Emotion2Vec
    emotion_results = benchmark_emotion2vec()

    # 测试2: Memobase
    memobase_results = await benchmark_memobase()

    print("\n" + "=" * 50)
    print("\n总结:")
    print("1. 根据Emotion2Vec延迟决定滚动分析的采样间隔")
    print("2. 根据Memobase延迟决定是否实施提前检索")
    print("3. 如果两个延迟都小，整体响应会很快")
    print("4. 如果任一延迟大，需要调整设计")


if __name__ == "__main__":
    asyncio.run(main())
