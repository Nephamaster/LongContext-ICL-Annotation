#!/usr/bin/env python3
"""计算标准答案 JSON 与预测 JSONL 的准确率。

默认文件：
- data_100/task1_testanswer.json
- data_100/tem.jsonl

用法：
    python src/calc_accuracy.py
    python src/calc_accuracy.py --answer data_100/task1_testanswer.json --pred data_100/tem.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_answer_map(answer_path: Path) -> Dict[str, str]:
    data = json.loads(answer_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"答案文件格式错误：期望顶层为 list，实际为 {type(data)}")

    answer_map: Dict[str, str] = {}
    for idx, item in enumerate(data):
        try:
            sample_id = item["id"]
            output = item["output"]
            answer = str(output[0])
        except Exception as e:
            raise ValueError(f"答案文件第 {idx} 条数据格式异常: {item}") from e
        answer_map[sample_id] = answer
    return answer_map


def load_prediction_map(pred_path: Path) -> Dict[str, str]:
    pred_map: Dict[str, str] = {}
    for line_no, line in enumerate(pred_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            sample_id = obj["test_sample_id"]
            pred = str(obj["prediction"])
        except Exception as e:
            raise ValueError(f"预测文件第 {line_no} 行格式异常: {line}") from e
        pred_map[sample_id] = pred
    return pred_map


def compare(answer_map: Dict[str, str], pred_map: Dict[str, str]) -> Tuple[int, int, int, int, List[Tuple[str, str, str]]]:
    answer_ids = set(answer_map)
    pred_ids = set(pred_map)

    overlap_ids = sorted(answer_ids & pred_ids)
    missing_pred_count = len(answer_ids - pred_ids)
    extra_pred_count = len(pred_ids - answer_ids)

    wrong_cases: List[Tuple[str, str, str]] = []
    correct = 0
    for sample_id in overlap_ids:
        gold = answer_map[sample_id]
        pred = pred_map[sample_id]
        if gold == pred:
            correct += 1
        else:
            wrong_cases.append((sample_id, gold, pred))

    return len(answer_map), len(pred_map), len(overlap_ids), correct, missing_pred_count, extra_pred_count, wrong_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="计算答案 JSON 与预测 JSONL 的正确率")
    parser.add_argument(
        "--answer",
        type=Path,
        default=Path("/mnt/disk4t/huangyishuo/Race/LongContext-ICL-Annotation/data_100/task7_testanswer.json"),
        help="标准答案 JSON 文件路径",
    )
    parser.add_argument(
        "--pred",
        type=Path,
        default=Path("/mnt/disk4t/huangyishuo/Race/LongContext-ICL-Annotation/outputs/openseek-7-v1.jsonl"),
        help="预测结果 JSONL 文件路径",
    )
    parser.add_argument(
        "--show-wrong",
        type=int,
        default=10,
        help="最多展示多少条错误样本",
    )

    args = parser.parse_args()

    answer_map = load_answer_map(args.answer)
    pred_map = load_prediction_map(args.pred)

    total_ans, total_pred, overlap, correct, missing_pred, extra_pred, wrong_cases = compare(answer_map, pred_map)

    acc_on_overlap = (correct / overlap) if overlap else 0.0
    acc_on_all_answers = (correct / total_ans) if total_ans else 0.0

    print("=== 评估结果 ===")
    print(f"答案总数: {total_ans}")
    print(f"预测总数: {total_pred}")
    print(f"对齐样本数(交集): {overlap}")
    print(f"正确数: {correct}")
    print(f"缺失预测数(答案有/预测无): {missing_pred}")
    print(f"额外预测数(预测有/答案无): {extra_pred}")
    print(f"正确率(在交集上): {acc_on_overlap:.2%}")
    print(f"正确率(相对全部答案): {acc_on_all_answers:.2%}")

    if wrong_cases and args.show_wrong > 0:
        print("\n=== 错误样本(部分) ===")
        for sample_id, gold, pred in wrong_cases[: args.show_wrong]:
            print(f"id={sample_id}  gold={gold}  pred={pred}")


if __name__ == "__main__":
    main()
