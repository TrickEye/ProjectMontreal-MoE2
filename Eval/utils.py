"""
共享工具函数：
- GSM8K / MATH 的标准答案与模型输出中答案的提取
- 数值 / 表达式的等价性比较
- 占位的模型生成函数 generate_response（请替换成你自己的模型推理逻辑）
"""

import re
from typing import Optional


# ============================================================
# 1. 模型生成占位函数
# ============================================================
def generate_response(prompt: str, **kwargs) -> str:
    """
    占位函数：代表"模型生成"这一步。

    在真正接入你自己的模型时，只需要替换这个函数的实现即可，
    其余评测逻辑（数据加载、答案提取、打分、统计）都不需要改动。

    要求：
        - 输入：拼接好的 prompt（题目 + few-shot 示例等）
        - 输出：模型生成的字符串（通常应包含最终答案，
                GSM8K 建议以 "#### <answer>" 结尾，
                MATH 建议以 \\boxed{<answer>} 给出最终答案）

    示例（接入 HuggingFace transformers 模型）：
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("your-model")
        model = AutoModelForCausalLM.from_pretrained("your-model")

        def generate_response(prompt, **kwargs):
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=512)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

    示例（接入 OpenAI 兼容 API）：
        import openai
        def generate_response(prompt, **kwargs):
            resp = openai.chat.completions.create(
                model="your-model",
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content
    """
    raise NotImplementedError(
        "请将 generate_response 替换为你自己模型的推理逻辑，"
        "当前仅为占位实现，用于跑通评测流程的示例可以临时 return '#### 0' 之类的假输出。"
    )


# ============================================================
# 2. GSM8K 答案提取与比对
# ============================================================
GSM8K_ANS_RE = re.compile(r"####\s*(-?[0-9\.\,]+)")
GSM8K_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """从模型输出或标准答案文本中提取最终数值答案。"""
    if text is None:
        return None
    match = GSM8K_ANS_RE.search(text)
    if match:
        return match.group(1).replace(",", "").strip()

    # 兜底：如果没有 "####" 格式，尝试取文本中最后一个数字
    nums = GSM8K_NUM_RE.findall(text)
    if nums:
        return nums[-1].replace(",", "").strip()
    return None


def gsm8k_is_correct(pred_text: str, gold_text: str) -> bool:
    pred = extract_gsm8k_answer(pred_text)
    gold = extract_gsm8k_answer(gold_text)
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        return pred.strip() == gold.strip()


# ============================================================
# 3. MATH500 答案提取与比对
# ============================================================
def extract_boxed_answer(text: str) -> Optional[str]:
    """提取 \\boxed{...} 中的内容（支持嵌套花括号）。"""
    if text is None:
        return None
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    idx += len("\\boxed")
    # 跳过空格
    while idx < len(text) and text[idx] == " ":
        idx += 1
    if idx >= len(text) or text[idx] != "{":
        return None

    depth = 0
    start = idx
    for i in range(idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
    return None


def _normalize_math_answer(ans: str) -> str:
    """对答案字符串做基础归一化，便于比较。"""
    if ans is None:
        return ""
    s = ans.strip()
    # 去掉常见的多余符号
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    s = s.replace(" ", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = s.rstrip(".")
    # 去掉外层 $ 符号
    s = s.strip("$")
    return s


def math_is_equiv(pred_text: str, gold_answer: str) -> bool:
    """
    比较模型输出与标准答案是否等价。

    注：这里使用基础的字符串归一化比较，覆盖大多数简单情况
    （整数、小数、简单分数、简单表达式等）。
    如果需要更严格的数学等价判断（如分数化简、代数等价等），
    建议引入 `math_verify` 或 sympy 做符号化比较，
    可在此函数中替换实现，不影响其余评测流程。
    """
    pred = extract_boxed_answer(pred_text)
    if pred is None:
        pred = pred_text  # 兜底：直接用全文比较

    pred_norm = _normalize_math_answer(pred)
    gold_norm = _normalize_math_answer(gold_answer)

    if pred_norm == gold_norm:
        return True

    # 尝试数值比较
    try:
        return abs(float(pred_norm) - float(gold_norm)) < 1e-4
    except ValueError:
        pass

    return False