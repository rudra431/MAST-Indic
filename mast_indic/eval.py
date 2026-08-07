"""Evaluate MAST Indic run files against gold answers using an LLM judge.

There are no qrels/relevance-judgment files for this track, so this only
computes judge-based metrics (accuracy, calibration error) plus SQuAD-style
exact-match/F1 string metrics -- no retrieval recall and no citation
precision/recall.

Expected inputs:
  --gold        JSONL with fields: task_id, question, gold_answer
  --predictions One or more run JSONL files (as written by runner.py), or
                directories containing them. Each line has fields:
                query_id, language, retriever, llm, tool_call_counts,
                retrieved_docids, result (list of reasoning/tool_call/
                output_text steps -- the last output_text step is the
                model's final answer).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import string
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from openai import OpenAI
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from .config import config

GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response].

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


def load_ground_truth(jsonl_path: Path) -> Dict[str, Dict[str, str]]:
    gt: Dict[str, Dict[str, str]] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            gt[str(obj["task_id"])] = {
                "question": obj["question"],
                "answer": obj["gold_answer"],
            }
    return gt


def resolve_prediction_files(paths: List[str]) -> List[Path]:
    files: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.glob("*.jsonl")))
        elif p.is_file():
            files.append(p)
        else:
            raise ValueError(f"Predictions path {p} does not exist")
    return files


def load_predictions(jsonl_path: Path) -> List[dict]:
    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_response(record: dict) -> str:
    result = record.get("result") or []
    if result and result[-1].get("type") == "output_text":
        return result[-1].get("output", "") or ""
    return ""


def extract_exact_answer(response_text: str) -> str:
    """Pull out the short answer the agent was told to give after 'Exact Answer:'."""
    match = re.search(r"Exact Answer:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text.strip()


# SQuAD-style string normalization/metrics:
# https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v2.0.py
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return float(pred_tokens == gt_tokens)

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_string_metrics(result: dict) -> dict:
    response = result.get("response", "") or ""
    if not response:
        return {"predicted_answer": "", "exact_match": False, "f1": 0.0}

    judge_result = result.get("judge_result", {}) or {}
    predicted = judge_result.get("extracted_final_answer") or extract_exact_answer(response)
    correct_answer = result.get("correct_answer", "") or ""

    return {
        "predicted_answer": predicted,
        "exact_match": exact_match_score(predicted, correct_answer),
        "f1": f1_score(predicted, correct_answer),
    }


def create_judge_prompt(question: str, response: str, correct_answer: str) -> str:
    return GRADER_TEMPLATE.format(
        question=question, response=response, correct_answer=correct_answer
    )


def parse_judge_response(judge_response: str) -> dict:
    result = {
        "extracted_final_answer": None,
        "reasoning": None,
        "correct": None,
        "confidence": None,
        "parse_error": False,
    }

    if not judge_response:
        result["parse_error"] = True
        return result

    # Extract extracted_final_answer (try bold formats first, then regular)
    answer_match = re.search(
        r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)",
        judge_response,
        re.IGNORECASE | re.DOTALL,
    )
    if not answer_match:
        answer_match = re.search(
            r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)",
            judge_response,
            re.IGNORECASE | re.DOTALL,
        )
    if not answer_match:
        answer_match = re.search(
            r"extracted_final_answer:\s*(.*?)(?=\n|$)",
            judge_response,
            re.IGNORECASE | re.DOTALL,
        )
    if answer_match:
        result["extracted_final_answer"] = answer_match.group(1).strip()

    # Extract reasoning/explanation
    reasoning_match = re.search(
        r"\*\*reasoning:\*\*\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
        judge_response,
        re.IGNORECASE | re.DOTALL,
    )
    if not reasoning_match:
        reasoning_match = re.search(
            r"\*\*reasoning\*\*:\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
            judge_response,
            re.IGNORECASE | re.DOTALL,
        )
    if not reasoning_match:
        reasoning_match = re.search(
            r"reasoning:\s*(.*?)(?=\ncorrect:|$)",
            judge_response,
            re.IGNORECASE | re.DOTALL,
        )
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    # Extract correct (yes/no)
    correct_match = re.search(
        r"\*\*correct:\*\*\s*(yes|no)", judge_response, re.IGNORECASE
    )
    if not correct_match:
        correct_match = re.search(
            r"\*\*correct\*\*:\s*(yes|no)", judge_response, re.IGNORECASE
        )
    if not correct_match:
        correct_match = re.search(r"correct:\s*(yes|no)", judge_response, re.IGNORECASE)
    if correct_match:
        result["correct"] = correct_match.group(1).lower() == "yes"

    # Extract confidence (percentage)
    confidence_match = re.search(
        r"\*\*confidence:\*\*\s*(\d+(?:\.\d+)?)\s*%?", judge_response, re.IGNORECASE
    )
    if not confidence_match:
        confidence_match = re.search(
            r"\*\*confidence\*\*:\s*(\d+(?:\.\d+)?)\s*%?", judge_response, re.IGNORECASE
        )
    if not confidence_match:
        confidence_match = re.search(
            r"confidence:\s*(\d+(?:\.\d+)?)\s*%?", judge_response, re.IGNORECASE
        )
    if confidence_match:
        result["confidence"] = float(confidence_match.group(1))
        if result["confidence"] > 100:
            result["confidence"] = 100

    # Check if we got the essential fields
    if result["correct"] is None:
        result["parse_error"] = True

    return result


# source: https://github.com/hendrycks/outlier-exposure/blob/master/utils/calibration_tools.py
def calib_err(confidence, correct, p="2", beta=100):
    idxs = np.argsort(confidence)
    bins = [[bin_i * beta, (bin_i + 1) * beta] for bin_i in range(len(idxs) // beta)]
    bins[-1] = [bins[-1][0], len(idxs)]
    cerr = 0
    total_examples = len(confidence)
    for bin_start, bin_end in bins:
        bin_confidence = confidence[idxs[bin_start:bin_end]]
        bin_correct = correct[idxs[bin_start:bin_end]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))

            if p == "2":
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == "1":
                cerr += num_examples_in_bin / total_examples * difference
            elif p in ("infty", "infinity", "max"):
                cerr = np.maximum(cerr, difference)
            else:
                raise ValueError("p must be '1', '2', or 'infty'")

    if p == "2":
        cerr = np.sqrt(cerr)

    return cerr


def calculate_calibration_error(
    confidences: List[float], correctness: List[bool], p: str = "2", beta: int = 100
) -> float:
    confidence = np.array(confidences) / 100.0
    correct = np.array(correctness, dtype=float)
    if len(confidence) < beta:
        return 0.0
    return float(calib_err(confidence, correct, p=p, beta=beta) * 100)


def save_detailed_csv(all_results: List[dict], csv_path: Path):
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "query_id",
            "predicted_answer",
            "correct_answer",
            "judge_correct",
            "confidence",
            "exact_match",
            "f1",
            "is_completed",
            "parse_error",
            "source_file",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in all_results:
            judge_result = result.get("judge_result", {})
            string_metrics = compute_string_metrics(result)
            predicted_answer = string_metrics["predicted_answer"]

            # If we couldn't extract an exact answer, use the full response (truncated for readability)
            if not predicted_answer:
                full_response = result.get("response", "")
                predicted_answer = (
                    full_response[:200] + "..."
                    if len(full_response) > 200
                    else full_response
                )

            writer.writerow(
                {
                    "query_id": result.get("query_id", ""),
                    "predicted_answer": predicted_answer,
                    "correct_answer": result.get("correct_answer", ""),
                    "judge_correct": judge_result.get("correct", ""),
                    "confidence": judge_result.get("confidence", ""),
                    "exact_match": string_metrics["exact_match"],
                    "f1": round(string_metrics["f1"], 4),
                    "is_completed": result.get("is_completed", ""),
                    "parse_error": judge_result.get("parse_error", False),
                    "source_file": result.get("source_file", ""),
                }
            )

    print(f"Detailed CSV results saved to {csv_path}")


def grade_prediction_file(
    pred_path: Path,
    ground_truth: Dict[str, Dict[str, str]],
    eval_dir: Path,
    judge_client: OpenAI,
    args: argparse.Namespace,
) -> List[dict]:
    predictions = load_predictions(pred_path)
    if not predictions:
        print(f"No records found in {pred_path}")
        return []

    eval_path = eval_dir / f"{pred_path.stem}_eval.jsonl"
    existing_by_qid: Dict[str, dict] = {}
    if eval_path.exists() and not args.force:
        with eval_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                existing = json.loads(line)
                existing_by_qid[str(existing.get("query_id"))] = existing

    all_results: List[dict] = []
    pending_items: List[dict] = []

    for record in predictions:
        query_id = str(record.get("query_id"))
        if query_id in existing_by_qid:
            all_results.append(existing_by_qid[query_id])
            continue

        if query_id not in ground_truth:
            print(f"No ground truth for query_id {query_id} in {pred_path}")
            continue

        gt_question = ground_truth[query_id]["question"]
        correct_answer = ground_truth[query_id]["answer"]
        response = extract_response(record)

        if not response:
            result = {
                "source_file": str(pred_path),
                "query_id": query_id,
                "question": gt_question,
                "response": response,
                "correct_answer": correct_answer,
                "is_completed": False,
                "judge_prompt": None,
                "judge_response": None,
                "judge_result": {
                    "parse_error": True,
                    "correct": None,
                    "confidence": None,
                    "error": "Response missing or incomplete",
                },
                "tool_call_counts": record.get("tool_call_counts", {}),
                "model_info": {"judge_model": args.judge_model},
            }
            all_results.append(result)
            continue

        pending_items.append(
            {
                "source_file": pred_path,
                "query_id": query_id,
                "gt_question": gt_question,
                "correct_answer": correct_answer,
                "response": response,
                "tool_call_counts": record.get("tool_call_counts", {}),
                "judge_prompt": create_judge_prompt(gt_question, response, correct_answer),
            }
        )

    def _judge_one(item: dict) -> dict:
        judge_text = ""
        try:
            completion = judge_client.chat.completions.create(
                model=args.judge_model,
                messages=[{"role": "user", "content": item["judge_prompt"]}],
                temperature=args.temperature,
                max_tokens=args.max_output_tokens,
            )
            judge_text = completion.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 -- one bad judge call must not sink the batch
            print(f"Error judging query {item['query_id']}: {exc!r}")

        judge_result = parse_judge_response(judge_text)
        return {
            "source_file": str(item["source_file"]),
            "query_id": item["query_id"],
            "question": item["gt_question"],
            "response": item["response"],
            "correct_answer": item["correct_answer"],
            "is_completed": True,
            "judge_prompt": item["judge_prompt"],
            "judge_response": judge_text,
            "judge_result": judge_result,
            "tool_call_counts": item["tool_call_counts"],
            "model_info": {"judge_model": args.judge_model},
        }

    if pending_items:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(_judge_one, item) for item in pending_items]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Judging [{pred_path.name}]",
                unit="query",
            ):
                all_results.append(future.result())

    eval_dir.mkdir(parents=True, exist_ok=True)
    with eval_path.open("w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_results)} judged records -> {eval_path}")

    return all_results


def summarize(all_results: List[dict], pred_path: Path, eval_dir: Path, args: argparse.Namespace) -> dict:
    all_tool_counts = defaultdict(int)
    for result in all_results:
        for tool_name, count in result.get("tool_call_counts", {}).items():
            all_tool_counts[tool_name] += count
    for tool_name, count in all_tool_counts.items():
        all_tool_counts[tool_name] = count / len(all_results)

    confidences = []
    correctness = []
    missing_judge_confidence_count = 0

    for result in all_results:
        judge_result = result.get("judge_result", {})
        judge_conf = judge_result.get("confidence")

        if (
            not judge_result.get("parse_error", False)
            and judge_result.get("correct") is not None
        ):
            if judge_conf is not None:
                confidences.append(judge_conf)
                correctness.append(judge_result.get("correct"))
            else:
                missing_judge_confidence_count += 1

    if missing_judge_confidence_count > 0:
        print(
            f"Warning: {missing_judge_confidence_count} of {len(all_results)} results are missing judge "
            "confidence scores, either because the original response was incomplete or because the judge "
            "model failed to judge the response"
        )

    if confidences and len(confidences) >= 100:
        calibration_error = calculate_calibration_error(confidences, correctness)
    else:
        print(
            f"Warning: {len(confidences)} confidences in total, not enough to calculate calibration "
            "error (need at least 100)"
        )
        calibration_error = None

    total = len(all_results)
    correct_count = sum(
        1 for r in all_results if r.get("judge_result", {}).get("correct", False)
    )
    accuracy_percent = round((correct_count / total) * 100.0, 2) if total else 0.0
    calibration_err_percent = (
        round(calibration_error, 2) if isinstance(calibration_error, (int, float)) else None
    )

    string_metrics_by_result = [compute_string_metrics(r) for r in all_results]
    em_percent = (
        round(
            sum(1 for m in string_metrics_by_result if m["exact_match"]) / total * 100.0, 2
        )
        if total
        else 0.0
    )
    f1_percent = (
        round(sum(m["f1"] for m in string_metrics_by_result) / total * 100.0, 2)
        if total
        else 0.0
    )

    per_query_metrics = [
        {
            "query_id": r.get("query_id"),
            "correct": bool(r.get("judge_result", {}).get("correct", False)),
            "confidence": r.get("judge_result", {}).get("confidence"),
            "exact_match": m["exact_match"],
            "f1": round(m["f1"], 4),
        }
        for r, m in zip(all_results, string_metrics_by_result)
    ]

    summary = {
        "run_file": str(pred_path),
        "Judge Model": args.judge_model,
        "Accuracy (%)": accuracy_percent,
        "Exact Match (%)": em_percent,
        "F1 (%)": f1_percent,
        "Calibration Error (%)": calibration_err_percent,
        "avg_tool_stats": dict(all_tool_counts),
        "num_evaluated": total,
        "Evaluation Date": datetime.now().date().isoformat(),
        "per_query_metrics": per_query_metrics,
    }

    print(f"\n[{pred_path.name}] Evaluated {total} responses:")
    print(f"Accuracy: {accuracy_percent:.2f}%")
    print(f"Exact Match: {em_percent:.2f}%")
    print(f"F1: {f1_percent:.2f}%")
    print(
        "Calibration Error: "
        + (f"{calibration_err_percent:.2f}%" if calibration_err_percent is not None else "N/A")
    )
    print(f"Average Tool Calls: {dict(all_tool_counts)}")

    summary_path = eval_dir / f"{pred_path.stem}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to {summary_path}")

    save_detailed_csv(all_results, eval_dir / f"{pred_path.stem}_detailed.csv")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MAST Indic run files with an LLM judge (no qrels required)."
    )
    parser.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help="Run JSONL file(s) (as written by runner.py), or directories containing them",
    )
    parser.add_argument(
        "--gold",
        default="data/hi_gold.jsonl",
        help="Path to gold JSONL dataset (expects fields: task_id, question, gold_answer)",
    )
    parser.add_argument(
        "--eval_dir", default="./evals", help="Directory to store evaluation results"
    )
    parser.add_argument(
        "--judge_model",
        default=config.judge_model,
        help="Model used to judge predicted vs. gold answers "
        "(default: MAST_JUDGE_MODEL, falling back to MAST_CHAT_MODEL)",
    )
    parser.add_argument(
        "--judge_base_url",
        default=config.judge_base_url,
        help="OpenAI-compatible base URL for the judge model "
        "(default: MAST_JUDGE_BASE_URL, falling back to OPENAI_BASE_URL)",
    )
    parser.add_argument(
        "--judge_api_key",
        default=config.judge_api_key,
        help="API key for the judge model endpoint "
        "(default: MAST_JUDGE_API_KEY, falling back to OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Judge decoding temperature"
    )
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=2048,
        help="Maximum output tokens for the judge model",
    )
    parser.add_argument(
        "--concurrency", type=int, default=8, help="Number of concurrent judge calls"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-evaluation of already-judged queries"
    )
    args = parser.parse_args()

    gt_path = Path(args.gold)
    if not gt_path.is_file():
        raise ValueError(f"Ground truth JSONL file {gt_path} does not exist")

    print(f"Loading ground truth from {gt_path}")
    ground_truth = load_ground_truth(gt_path)

    pred_files = resolve_prediction_files(args.predictions)
    if not pred_files:
        print("No prediction JSONL files found")
        return
    print(f"Found {len(pred_files)} prediction file(s) to evaluate")

    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    judge_client = OpenAI(base_url=args.judge_base_url, api_key=args.judge_api_key)

    summaries = []
    for pred_path in pred_files:
        all_results = grade_prediction_file(pred_path, ground_truth, eval_dir, judge_client, args)
        if not all_results:
            print(f"No results to analyze for {pred_path}")
            continue
        summaries.append(summarize(all_results, pred_path, eval_dir, args))

    if len(summaries) > 1:
        overview_path = eval_dir / "evaluation_overview.json"
        with overview_path.open("w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
        print(f"\nOverview of {len(summaries)} runs saved to {overview_path}")


if __name__ == "__main__":
    main()
