"""
GSM8K 评测脚本

数据集来源（HuggingFace）：openai/gsm8k, config="main", split="test"
https://huggingface.co/datasets/openai/gsm8k

用法：
    python gsm8k_eval.py --limit 50          # 只测前 50 条，方便快速跑通
    python gsm8k_eval.py                     # 跑全量 test 集（1319 条）
    python gsm8k_eval.py --output result.json

模型生成部分使用 utils.generate_response 占位函数，
请在接入自己的模型时替换该函数的实现。
"""

import argparse
import json
import time

from datasets import load_dataset
from tqdm import tqdm

from utils import generate_response, gsm8k_is_correct

FEW_SHOT_PROMPT = """请一步步推理并解答下面的数学题，最后一行以 "#### <数字答案>" 的格式给出最终答案。

题目: {question}
解答:"""


def build_prompt(question: str) -> str:
    return FEW_SHOT_PROMPT.format(question=question)


def run_eval(limit: int = None, output_path: str = "gsm8k_results.json"):
    print("正在加载 GSM8K 数据集 (openai/gsm8k, main, test)...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    total = len(dataset)
    correct = 0
    records = []

    start_time = time.time()
    for idx, example in enumerate(tqdm(dataset, desc="GSM8K 评测中")):
        question = example["question"]
        gold_answer = example["answer"]  # 含推理过程 + "#### 答案"

        prompt = build_prompt(question)

        try:
            model_output = generate_response(prompt)
        except NotImplementedError:
            raise
        except Exception as e:
            model_output = ""
            print(f"[警告] 第 {idx} 条样本生成失败: {e}")

        is_correct = gsm8k_is_correct(model_output, gold_answer)
        correct += int(is_correct)

        records.append({
            "index": idx,
            "question": question,
            "gold_answer": gold_answer,
            "model_output": model_output,
            "correct": is_correct,
        })

    elapsed = time.time() - start_time
    accuracy = correct / total if total > 0 else 0.0

    summary = {
        "benchmark": "GSM8K",
        "dataset": "openai/gsm8k (main, test)",
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_seconds": round(elapsed, 2),
    }

    result = {"summary": summary, "records": records}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n===== GSM8K 评测结果 =====")
    print(f"样本数: {total}")
    print(f"正确数: {correct}")
    print(f"准确率: {accuracy:.4f}")
    print(f"结果已保存至: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="GSM8K 评测脚本")
    parser.add_argument("--limit", type=int, default=None, help="仅评测前 N 条样本，默认全量")
    parser.add_argument("--output", type=str, default="gsm8k_results.json", help="结果输出路径")
    args = parser.parse_args()

    run_eval(limit=args.limit, output_path=args.output)


if __name__ == "__main__":
    main()