import argparse
import csv
from datetime import datetime
import json
import os
import re
from pathlib import Path

import run_llm_expert_survey as survey


DEFAULT_MODELS = ["claude-sonnet-4-5-20250929", "claude-sonnet-4-6"]


def write_jsonl_line(path, payload):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_unique_failed(path, limit):
    items = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            key = (payload["expert_id"], int(payload["pair_number"]))
            if key in seen:
                continue
            seen.add(key)
            items.append(payload)
            if len(items) >= limit:
                break
    return items


def result_rows_from_answers(model, item, answers):
    rows = []
    hypothesis_a_id = item["hypothesis_a_id"]
    hypothesis_b_id = item["hypothesis_b_id"]
    for key, dimension in survey.DIMENSIONS:
        rows.append(
            [
                item["expert_id"],
                hypothesis_a_id,
                hypothesis_b_id,
                dimension,
                survey.answer_to_result(answers[key], hypothesis_a_id, hypothesis_b_id),
                model,
                int(item["pair_number"]),
            ]
        )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Smoke-test selected failed LLM-as-Judge pairs on alternative Claude models."
    )
    parser.add_argument(
        "--failed-jsonl",
        default=(
            "LLMJudge_results/claude-sonnet-5/full_run_20260709_claude_sonnet_5/"
            "failed_model_calls.jsonl"
        ),
    )
    parser.add_argument("--output-root", default="LLMJudge_results/failed_pair_smoke_tests")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--base-url", default=os.environ.get("EXPERT_BASE_URL", "https://svip-ip.xty.app/v1"))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    api_key = re.sub(r"\s+", "", os.environ.get("EXPERT_API_KEY") or "")
    if not api_key:
        raise SystemExit("Please set EXPERT_API_KEY in the environment.")

    script_dir = Path(__file__).resolve().parent
    failed_path = Path(args.failed_jsonl)
    if not failed_path.is_absolute():
        failed_path = script_dir / failed_path
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = script_dir / output_root

    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_output = run_dir / "raw_smoke_results.jsonl"
    summary_output = run_dir / "summary.csv"
    comparison_output = run_dir / "comparison_rows.csv"
    metadata_output = run_dir / "metadata.json"

    failed_items = load_unique_failed(failed_path, args.limit)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "failed_jsonl": str(failed_path),
        "output_directory": str(run_dir),
        "models": args.models,
        "limit": args.limit,
        "loaded_failed_pairs": len(failed_items),
        "base_url": args.base_url,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
        "retry_sleep": args.retry_sleep,
        "hypothesis_text_modified": False,
    }
    metadata_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_rows = []
    comparison_rows = []
    for model in args.models:
        for item in failed_items:
            expert_id = item["expert_id"]
            pair_number = int(item["pair_number"])
            system_prompt = survey.build_system_prompt(expert_id)
            user_prompt = survey.build_user_prompt(
                item["hypothesis_a_content"],
                item["hypothesis_b_content"],
            )
            started_at = datetime.now().isoformat(timespec="seconds")
            record = {
                "started_at": started_at,
                "model_name": model,
                "expert_id": expert_id,
                "expert_label": survey.anonymized_expert_label(expert_id),
                "pair_number": pair_number,
                "hypothesis_a_id": item["hypothesis_a_id"],
                "hypothesis_b_id": item["hypothesis_b_id"],
                "hypothesis_a_content": item["hypothesis_a_content"],
                "hypothesis_b_content": item["hypothesis_b_content"],
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "hypothesis_text_modified": False,
            }
            try:
                content = None
                raw_data = None
                for attempt in range(args.max_retries + 1):
                    try:
                        content, raw_data = survey.call_model(
                            args.base_url,
                            api_key,
                            model,
                            args.timeout,
                            system_prompt,
                            user_prompt,
                            args.temperature,
                        )
                        parsed = survey.extract_json(content)
                        answers = survey.validate_parsed_answers(parsed)
                        break
                    except Exception:
                        if attempt >= args.max_retries:
                            raise
                        print(
                            f"API call or response validation failed; retrying in "
                            f"{args.retry_sleep:.1f}s ({attempt + 1}/{args.max_retries})..."
                        )
                        import time

                        time.sleep(args.retry_sleep)
                record.update(
                    {
                        "completed_at": datetime.now().isoformat(timespec="seconds"),
                        "status": "success",
                        "parsed_answers": answers,
                        "parsed_json": parsed,
                        "raw_message_content": content,
                        "raw_api_response": raw_data,
                    }
                )
                comparison_rows.extend(result_rows_from_answers(model, item, answers))
                print(f"SUCCESS {model} {expert_id} pair {pair_number}: {answers}")
            except Exception as exc:
                if raw_data is not None:
                    record["raw_api_response"] = raw_data
                    record["raw_message_content"] = content
                    record["finish_reason"] = raw_data.get("choices", [{}])[0].get("finish_reason")
                record.update(
                    {
                        "completed_at": datetime.now().isoformat(timespec="seconds"),
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                print(f"FAILED {model} {expert_id} pair {pair_number}: {type(exc).__name__}: {exc}")

            write_jsonl_line(raw_output, record)
            summary_rows.append(
                {
                    "model_name": model,
                    "expert_id": expert_id,
                    "pair_number": pair_number,
                    "hypothesis_a_id": item["hypothesis_a_id"],
                    "hypothesis_b_id": item["hypothesis_b_id"],
                    "status": record["status"],
                    "error_type": record.get("error_type", ""),
                    "error_message": record.get("error_message", ""),
                }
            )

    with summary_output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with comparison_output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Expert_ID",
                "Hypothesis_A_ID",
                "Hypothesis_B_ID",
                "Dimension",
                "Result",
                "Model",
                "Pair_Number",
            ]
        )
        writer.writerows(comparison_rows)

    success_count = sum(1 for row in summary_rows if row["status"] == "success")
    print(f"Smoke test complete: {success_count}/{len(summary_rows)} calls succeeded.")
    print(f"Output directory: {run_dir}")


if __name__ == "__main__":
    main()
