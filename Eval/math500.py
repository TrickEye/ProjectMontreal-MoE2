"""
MATH500 评测脚本

数据集来源（HuggingFace）：HuggingFaceH4/MATH-500, split="test"
https://huggingface.co/datasets/HuggingFaceH4/MATH-500

字段说明：
    problem  - 题目
    solution - 完整解答过程（含 \\boxed{} 最终答案）
    answer   - 已提取好的标准最终答案
    subject  - 学科分类
    level    - 难度等级
    unique_id- 唯一 id

用法：
    python math500_eval.py --limit 50
    python math500_eval.py
    python math500_eval.py --output result.json

模型生成部分使用 utils.generate_response 占位函数，
请在接入自己的模型时替换该函数的实现。
"""

import argparse
import json
import time

from datasets import load_dataset
from tqdm import tqdm

from utils import generate_response, math_is_equiv

PROMPT_TEMPLATE = """请一步步推理并解答下面的数学题，并将最终答案放在 \\boxed{{}} 中。

题目: {problem}
解答:"""


def build_prompt(problem: str) -> str:
    return PROMPT_TEMPLATE.format(problem=problem)


def run_eval(limit: int = None, output_path: str = "math500_results.json"):
    print("正在加载 MATH500 数据集 (HuggingFaceH4/MATH-500, test)...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    total = len(dataset)
    correct = 0
    records = []
    by_level = {}
    by_subject = {}

    start_time = time.time()
    for idx, example in enumerate(tqdm(dataset, desc="MATH500 评测中")):
        problem = example["problem"]
        gold_answer = example["answer"]
        level = example.get("level")
        subject = example.get("subject")

        prompt = build_prompt(problem)

        try:
            model_output = generate_response(prompt)
        except NotImplementedError:
            raise
        except Exception as e:
            model_output = ""
            print(f"[警告] 第 {idx} 条样本生成失败: {e}")

        is_correct = math_is_equiv(model_output, gold_answer)
        correct += int(is_correct)

        # 按 level / subject 统计
        if level is not None:
            by_level.setdefault(level, {"total": 0, "correct": 0})
            by_level[level]["total"] += 1
            by_level[level]["correct"] += int(is_correct)
        if subject is not None:
            by_subject.setdefault(subject, {"total": 0, "correct": 0})
            by_subject[subject]["total"] += 1
            by_subject[subject]["correct"] += int(is_correct)

        records.append({
            "index": idx,
            "problem": problem,
            "gold_answer": gold_answer,
            "level": level,
            "subject": subject,
            "model_output": model_output,
            "correct": is_correct,
        })

    elapsed = time.time() - start_time
    accuracy = correct / total if total > 0 else 0.0

    def _with_acc(d):
        return {
            k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] else 0.0}
            for k, v in d.items()
        }

    summary = {
        "benchmark": "MATH500",
        "dataset": "HuggingFaceH4/MATH-500 (test)",
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_seconds": round(elapsed, 2),
        "by_level": _with_acc(by_level),
        "by_subject": _with_acc(by_subject),
    }

    result = {"summary": summary, "records": records}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n===== MATH500 评测结果 =====")
    print(f"样本数: {total}")
    print(f"正确数: {correct}")
    print(f"准确率: {accuracy:.4f}")
    print(f"结果已保存至: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="MATH500 评测脚本")
    parser.add_argument("--limit", type=int, default=None, help="仅评测前 N 条样本，默认全量")
    parser.add_argument("--output", type=str, default="math500_results.json", help="结果输出路径")
    args = parser.parse_args()

    run_eval(limit=args.limit, output_path=args.output)


if __name__ == "__main__":
    main()