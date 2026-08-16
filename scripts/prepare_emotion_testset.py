"""
准备情绪识别测试集
- 从 SQLite 导出带音频的对话
- 人工标注真实情绪（或使用现有标注）
- 生成 manifest.json
"""

import sys
sys.path.insert(0, '.')

import json
import sqlite3
import shutil
from pathlib import Path
from typing import List, Dict
import argparse


class EmotionTestsetPreparer:
    """测试集准备工具"""

    EMOTION_LABELS = ['joy', 'sadness', 'anger', 'fear', 'anxiety', 'calm', 'confusion']

    def __init__(self, sqlite_db: str, output_dir: str):
        self.sqlite_db = sqlite_db
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.audio_dir = self.output_dir / "audio"
        self.audio_dir.mkdir(exist_ok=True)

        self.samples = []

    def extract_from_sqlite(self, num_samples: int = 100):
        """从 SQLite 提取样本"""
        print(f"从 SQLite 提取样本...")

        if not Path(self.sqlite_db).exists():
            print(f"⚠️  数据库不存在: {self.sqlite_db}")
            print(f"   将创建模拟测试集")
            self._create_mock_testset(num_samples)
            return

        conn = sqlite3.connect(self.sqlite_db)
        cursor = conn.cursor()

        # 查询带音频路径的对话
        query = """
            SELECT
                turn_id,
                user_id,
                user_text,
                audio_path,
                dominant_emotion,
                emotion_joy,
                emotion_sadness,
                emotion_anger,
                emotion_fear,
                emotion_anxiety,
                emotion_calm,
                emotion_confusion,
                timestamp
            FROM turns
            WHERE audio_path IS NOT NULL
            ORDER BY RANDOM()
            LIMIT ?
        """

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            print(f"⚠️  未找到带音频的对话记录")
            print(f"   将创建模拟测试集")
            self._create_mock_testset(num_samples)
            return

        print(f"✅ 提取到 {len(rows)} 条样本")

        for row in rows:
            turn_id, user_id, text, audio_path, emotion, *emotion_scores, timestamp = row

            # 复制音频文件
            if audio_path and Path(audio_path).exists():
                new_audio_path = self.audio_dir / f"sample_{turn_id}.wav"
                shutil.copy2(audio_path, new_audio_path)
                audio_path_rel = str(new_audio_path.relative_to(self.output_dir))
            else:
                print(f"   ⚠️  音频文件不存在: {audio_path}")
                continue

            # 推断年龄段（从 user_id 或默认）
            age_group = self._infer_age_group(user_id)

            self.samples.append({
                'sample_id': f"sample_{turn_id}",
                'audio_path': str(self.output_dir / audio_path_rel),
                'text': text,
                'true_emotion': emotion,  # 使用系统识别的情绪作为标注
                'emotion_scores': {
                    'joy': emotion_scores[0],
                    'sadness': emotion_scores[1],
                    'anger': emotion_scores[2],
                    'fear': emotion_scores[3],
                    'anxiety': emotion_scores[4],
                    'calm': emotion_scores[5],
                    'confusion': emotion_scores[6]
                },
                'age_group': age_group,
                'timestamp': timestamp,
                'source': 'sqlite'
            })

    def _create_mock_testset(self, num_samples: int):
        """创建模拟测试集（当没有真实数据时）"""
        print(f"创建模拟测试集: {num_samples} 条")

        # 模拟样本模板
        mock_templates = [
            {
                'text': '我最近睡眠不好，总是半夜醒来',
                'emotion': 'anxiety',
                'age_group': 'elderly'
            },
            {
                'text': '今天天气真好，心情不错',
                'emotion': 'joy',
                'age_group': 'middle_aged'
            },
            {
                'text': '我感觉记忆力下降了，很担心',
                'emotion': 'anxiety',
                'age_group': 'elderly'
            },
            {
                'text': '工作压力太大，感觉喘不过气',
                'emotion': 'anxiety',
                'age_group': 'young'
            },
            {
                'text': '我今天很平静，没什么特别的',
                'emotion': 'calm',
                'age_group': 'middle_aged'
            },
            {
                'text': '孩子不听话，我很生气',
                'emotion': 'anger',
                'age_group': 'middle_aged'
            },
            {
                'text': '我有点搞不清楚状况',
                'emotion': 'confusion',
                'age_group': 'elderly'
            },
            {
                'text': '感觉很难过，提不起精神',
                'emotion': 'sadness',
                'age_group': 'young'
            },
        ]

        for i in range(num_samples):
            template = mock_templates[i % len(mock_templates)]

            # 注意：这里不创建实际音频文件，只记录路径
            # 实际测试时需要用户提供真实音频
            audio_path = self.audio_dir / f"mock_sample_{i}.wav"

            self.samples.append({
                'sample_id': f"mock_sample_{i}",
                'audio_path': str(audio_path),
                'text': template['text'],
                'true_emotion': template['emotion'],
                'emotion_scores': self._generate_mock_scores(template['emotion']),
                'age_group': template['age_group'],
                'timestamp': None,
                'source': 'mock'
            })

        print(f"⚠️  注意：模拟测试集不包含实际音频文件")
        print(f"   请手动准备音频并替换 {self.audio_dir}/ 下的文件")

    def _generate_mock_scores(self, dominant: str) -> dict:
        """生成模拟的情绪得分"""
        scores = {e: 0.05 for e in self.EMOTION_LABELS}
        scores[dominant] = 0.65
        return scores

    def _infer_age_group(self, user_id: str) -> str:
        """推断年龄段（简单规则）"""
        # 实际应用中应从用户画像表读取
        if 'elderly' in user_id.lower() or 'old' in user_id.lower():
            return 'elderly'
        elif 'young' in user_id.lower():
            return 'young'
        else:
            return 'middle_aged'

    def add_manual_annotations(self, annotation_file: str):
        """
        添加人工标注（可选）

        annotation_file 格式（JSON）:
        [
            {"sample_id": "sample_1", "true_emotion": "anxiety"},
            {"sample_id": "sample_2", "true_emotion": "joy"},
            ...
        ]
        """
        if not Path(annotation_file).exists():
            print(f"⚠️  标注文件不存在: {annotation_file}")
            return

        annotations = json.load(open(annotation_file, encoding='utf-8'))
        annotation_map = {a['sample_id']: a['true_emotion'] for a in annotations}

        updated = 0
        for sample in self.samples:
            if sample['sample_id'] in annotation_map:
                old_emotion = sample['true_emotion']
                new_emotion = annotation_map[sample['sample_id']]
                if old_emotion != new_emotion:
                    sample['true_emotion'] = new_emotion
                    sample['manually_annotated'] = True
                    updated += 1

        print(f"✅ 更新了 {updated} 条人工标注")

    def save_manifest(self):
        """保存测试集清单"""
        manifest = {
            'version': '1.0',
            'total_samples': len(self.samples),
            'emotion_labels': self.EMOTION_LABELS,
            'samples': self.samples
        }

        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"✅ 测试集清单已保存: {manifest_path}")

        # 统计信息
        emotion_dist = {}
        age_dist = {}

        for s in self.samples:
            emo = s['true_emotion']
            age = s['age_group']

            emotion_dist[emo] = emotion_dist.get(emo, 0) + 1
            age_dist[age] = age_dist.get(age, 0) + 1

        print(f"\n测试集统计:")
        print(f"  总样本数: {len(self.samples)}")
        print(f"  情绪分布: {emotion_dist}")
        print(f"  年龄分布: {age_dist}")
        print(f"  输出目录: {self.output_dir}")

    def validate(self) -> bool:
        """验证测试集完整性"""
        print(f"\n验证测试集...")

        valid = True
        missing_audio = 0

        for sample in self.samples:
            audio_path = sample['audio_path']

            if not Path(audio_path).exists():
                missing_audio += 1
                if missing_audio <= 3:  # 只打印前3个
                    print(f"   ⚠️  音频文件缺失: {audio_path}")

            if sample['true_emotion'] not in self.EMOTION_LABELS:
                print(f"   ❌ 无效情绪标签: {sample['true_emotion']}")
                valid = False

        if missing_audio > 0:
            print(f"   ⚠️  共 {missing_audio} 个音频文件缺失")
            if sample.get('source') == 'mock':
                print(f"   这是模拟测试集，需要手动准备音频文件")
                valid = False

        if valid:
            print(f"✅ 测试集验证通过")

        return valid


def main():
    parser = argparse.ArgumentParser(description="准备情绪识别测试集")
    parser.add_argument(
        '--sqlite_db',
        type=str,
        default='data/patient_memory.db',
        help='SQLite 数据库路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='tests/emotion_benchmark/testset',
        help='输出目录'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=100,
        help='样本数量'
    )
    parser.add_argument(
        '--annotation_file',
        type=str,
        default=None,
        help='人工标注文件（可选）'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("准备情绪识别测试集")
    print("=" * 70)
    print(f"SQLite 数据库: {args.sqlite_db}")
    print(f"输出目录: {args.output_dir}")
    print(f"样本数量: {args.num_samples}")
    print("=" * 70)
    print()

    preparer = EmotionTestsetPreparer(args.sqlite_db, args.output_dir)

    # 1. 从 SQLite 提取
    preparer.extract_from_sqlite(args.num_samples)

    # 2. 添加人工标注（可选）
    if args.annotation_file:
        preparer.add_manual_annotations(args.annotation_file)

    # 3. 保存清单
    preparer.save_manifest()

    # 4. 验证
    preparer.validate()

    print("\n" + "=" * 70)
    print("下一步:")
    print(f"  1. 检查测试集: {args.output_dir}/")
    print(f"  2. 运行对比实验: python scripts/benchmark_emotion_models.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
