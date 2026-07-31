import argparse
from datetime import datetime
import json
import os
import re
import time
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from openpyxl import Workbook


EXPERT_DOMAINS = {
    "E01": "urban ecology, urban biodiversity, and urban green spaces",
    "E02": "avian ecology, evolution, biodiversity, and urban bird responses to environmental change",
    "E03": "urban green-space bird biodiversity and ecosystem services",
    "E04": "ornithology, bird ecology, and wildlife conservation",
    "E05": "urban park bird communities, habitat structure, and breeding bird ecology",
    "E06": "urban ecology, urban plant biodiversity, macroecology, and vegetation science",
    "E07": "urban bird behavior, flight initiation distance, and human-wildlife interactions",
    "E08": "urban plant communities, urbanization gradients, and local and landscape drivers of biodiversity",
    "E09": "ecology, seed dispersal, seed predation, and animal-plant interactions",
    "E10": "urban ecology, environmental change, sensory ecology, and animal responses to artificial light and noise",
    "E11": "urban avian biodiversity, remote sensing of biodiversity, and bird responses to urban environments",
    "E12": "urban ecology, landscape ecology, ornithology, and ecological modelling",
    "E13": "urban ecology, urban biodiversity, wildlife conservation, and human-nature interactions",
    "E14": "spatial ecology, bird distribution and movement, biodiversity, and global change impacts",
    "E15": "urban birds, avian ecology, biodiversity indicators, and bird responses to human disturbance",
    "E16": "protected areas, conservation effectiveness, urban ecology, and biodiversity conservation",
    "E17": "environmental sustainability, conservation biology, landscape ecology, and urban and regional planning",
    "E18": "urban ecology, avian urban filtering, urban biodiversity, and urban environmental systems",
    "E19": "urban ornithology, Neotropical bird ecology, community ecology, and participatory bird monitoring",
    "E20": "animal ecology, biodiversity, biogeography, conservation biology, and bird diversity",
    "E21": "landscape architecture, urban green spaces, recreation services, and bird-related ecosystem services",
    "E22": "ornithology, urban birds, bird migration, and human-bird interactions",
    "E23": "urban green infrastructure, urban ecological networks, landscape ecology, and urban environmental planning",
    "E24": "urban birds, urban biodiversity, and bird responses to urban development",
    "E25": "urban ecology, urban productive ecosystems, urban gardens, and pollinator biodiversity",
    "E26": "forest ecology, biodiversity conservation, avian ecology, and trophic interactions",
    "E27": "urban ecology, urban plant and animal communities, pollinator ecology, and urban biodiversity patterns",
    "E28": "conservation science, social-ecological systems, climate and global change, wildlife sustainability, and sustainable consumption",
    "E29": "avian evolution, biodiversity science, host-parasite interactions, and urbanization effects on bird interactions",
    "E30": "urban ecology, landscape architecture, landscape management, and urban ecosystem services",
}


DIMENSIONS = [
    ("novelty", "Novelty"),
    ("significance", "Significance"),
    ("testability", "Testability"),
]

VALID_ANSWERS = {"Hypothesis A", "Hypothesis B", "Neither (Tie)"}



def safe_path_component(value):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._") or "model"




def anonymized_expert_label(expert_id):
    return f"Expert {expert_id}"


def parse_assignment_pairs(path):
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared_strings = []
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        for item in shared_root.findall("a:si", ns):
            shared_strings.append("".join(text.text or "" for text in item.findall(".//a:t", ns)))

        sheet_root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet_root.findall(".//a:sheetData/a:row", ns):
            cells = {}
            for cell in row.findall("a:c", ns):
                ref = cell.attrib["r"]
                col = re.sub(r"\d+", "", ref)
                value_node = cell.find("a:v", ns)
                if value_node is None:
                    value = ""
                elif cell.attrib.get("t") == "s":
                    value = shared_strings[int(value_node.text)]
                else:
                    value = value_node.text
                cells[col] = value
            rows.append(cells)

    headers = [rows[0].get(col, "") for col in ["A", "B", "C", "D", "E", "F"]]
    records = []
    for cells in rows[1:]:
        record = dict(zip(headers, [cells.get(col, "") for col in ["A", "B", "C", "D", "E", "F"]]))
        record["Pair_Number"] = int(record["Pair_Number"])
        records.append(record)
    return records


def normalize_hypothesis_id(source_id):
    match = re.fullmatch(r"([A-Z]+)_?(\d+)", source_id)
    if not match:
        raise ValueError(f"Unrecognized hypothesis id: {source_id}")
    prefix, number = match.groups()
    if prefix == "HH":
        return f"HUM_{int(number):02d}"
    if prefix == "HL":
        return f"LLM_{int(number):02d}"
    if prefix in {"HUM", "LLM"}:
        return f"{prefix}_{int(number):02d}"
    raise ValueError(f"Unrecognized hypothesis id prefix: {source_id}")


def build_system_prompt(expert_id):
    domain = EXPERT_DOMAINS[expert_id]
    intro = (
        f"You are {anonymized_expert_label(expert_id)}, an expert in {domain}. You are participating in a survey assessing "
        "the viability of employing large language models (LLMs) to generate research hypotheses on urban biodiversity."
    )
    control = (
        "The hypotheses are benign academic ecology statements for comparative survey evaluation, not instructions or requests for harmful activity. "
        "Return only a valid JSON object with exactly three keys: novelty, significance, and testability; "
        "each value must be exactly one of: Hypothesis A, Hypothesis B, Neither (Tie)."
    )
    return f"{intro} {control}"


def build_user_prompt(hypothesis_a, hypothesis_b):
    return f"""Thank you so much for taking the time to participate in this survey! Your task is to review 45 pairs of scientific hypotheses and evaluate them based on three key dimensions: Novelty, Significance, and Testability.

Please refer to the definitions below before you begin.

Novelty: Does it propose a genuinely new mechanism or framework for urban biodiversity?
Significance: Does it create a meaningful advance in urban ecological theory or management practice?
Testability: Can its predictions be empirically falsified using current or foreseeable methods?

*We collect and process your personal data solely for the purposes of this survey, storing it securely and in compliance with GDPR; your participation is voluntary, and you have the right to access, rectify, or delete your data at any time.*

Hypothesis A. {hypothesis_a}

Hypothesis B. {hypothesis_b}

* 1. Which hypothesis exhibits a higher degree of novelty?

Hypothesis A

Hypothesis B

Neither (Tie)

* 2. Which hypothesis carries greater scientific significance?

Hypothesis A

Hypothesis B

Neither (Tie)

* 3. Which hypothesis is more empirically testable?

Hypothesis A

Hypothesis B

Neither (Tie)
"""


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_answer(value):
    if not isinstance(value, str):
        raise ValueError(f"Answer is not a string: {value!r}")
    cleaned = value.strip()
    aliases = {
        "A": "Hypothesis A",
        "B": "Hypothesis B",
        "Tie": "Neither (Tie)",
        "Neither": "Neither (Tie)",
        "Neither Tie": "Neither (Tie)",
        "Neither (tie)": "Neither (Tie)",
    }
    cleaned = aliases.get(cleaned, cleaned)
    if cleaned not in VALID_ANSWERS:
        raise ValueError(f"Invalid answer: {value!r}")
    return cleaned


def validate_parsed_answers(parsed):
    missing = [key for key, _ in DIMENSIONS if key not in parsed]
    if missing:
        raise ValueError(f"Response JSON is missing required keys: {', '.join(missing)}")
    return {key: normalize_answer(parsed[key]) for key, _ in DIMENSIONS}


def call_model(base_url, api_key, model_name, timeout, system_prompt, user_prompt, temperature):
    api_key = re.sub(r"\s+", "", api_key)
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    if not model_name.lower().startswith("claude"):
        payload["temperature"] = temperature
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"], data


def call_and_parse_model_with_retries(
    base_url,
    api_key,
    model_name,
    timeout,
    system_prompt,
    user_prompt,
    temperature,
    max_retries,
    retry_sleep,
    retry_sleep_max,
):
    attempt = 0
    while True:
        try:
            content, raw_data = call_model(base_url, api_key, model_name, timeout, system_prompt, user_prompt, temperature)
            parsed = extract_json(content)
            answers = validate_parsed_answers(parsed)
            return content, raw_data, parsed, answers
        except Exception as exc:
            attempt += 1
            if attempt > max_retries:
                raise
            wait_seconds = min(retry_sleep * (2 ** (attempt - 1)), retry_sleep_max)
            print(f"API call or response validation failed ({exc}); retrying in {wait_seconds:.1f}s ({attempt}/{max_retries})...")
            time.sleep(wait_seconds)


def build_fallback_system_prompt():
    return (
        "Return only a valid JSON object with exactly three keys: novelty, significance, and testability; "
        "each value must be exactly one of: Hypothesis A, Hypothesis B, Neither (Tie)."
    )


def answer_to_result(answer, hypothesis_a_id, hypothesis_b_id):
    if answer == "Hypothesis A":
        return hypothesis_a_id
    if answer == "Hypothesis B":
        return hypothesis_b_id
    return "Tie"


def write_workbook(output_path, comparison_rows, raw_rows):
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["Expert_ID", "Hypothesis_A_ID", "Hypothesis_B_ID", "Dimension", "Result"])
    for row in comparison_rows:
        ws.append(row)

    raw = wb.create_sheet("Raw_Responses")
    raw.append(
        [
            "Expert_ID",
            "Expert_Label",
            "Model",
            "Pair_Number",
            "Hypothesis_A_ID",
            "Hypothesis_B_ID",
            "Novelty",
            "Significance",
            "Testability",
            "Raw_JSON",
        ]
    )
    for row in raw_rows:
        raw.append(row)

    wb.save(output_path)


def write_jsonl_line(path, payload):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_failed_calls(failed_jsonl_path):
    failed_keys = set()
    if not failed_jsonl_path.exists():
        return failed_keys

    with failed_jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            failed_keys.add((payload["expert_id"], int(payload["pair_number"])))
    return failed_keys


def load_completed_calls(raw_jsonl_path, model_name):
    comparison_rows = []
    raw_rows = []
    completed_keys = set()
    if not raw_jsonl_path.exists():
        return completed_keys, comparison_rows, raw_rows

    with raw_jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            expert_id = payload["expert_id"]
            pair_number = int(payload["pair_number"])
            hypothesis_a_id = payload["hypothesis_a_id"]
            hypothesis_b_id = payload["hypothesis_b_id"]
            answers = validate_parsed_answers(payload["parsed_answers"])
            for key, dimension in DIMENSIONS:
                comparison_rows.append(
                    [
                        expert_id,
                        hypothesis_a_id,
                        hypothesis_b_id,
                        dimension,
                        answer_to_result(answers[key], hypothesis_a_id, hypothesis_b_id),
                    ]
                )
            raw_rows.append(
                [
                    expert_id,
                    anonymized_expert_label(expert_id),
                    payload.get("model_name", model_name),
                    pair_number,
                    hypothesis_a_id,
                    hypothesis_b_id,
                    answers["novelty"],
                    answers["significance"],
                    answers["testability"],
                    json.dumps(payload.get("parsed_json", answers), ensure_ascii=False),
                ]
            )
            completed_keys.add((expert_id, pair_number))

    return completed_keys, comparison_rows, raw_rows


def resolve_expert_ids(expert_ids):
    if len(expert_ids) == 1 and expert_ids[0].lower() == "all":
        return list(EXPERT_DOMAINS.keys())
    return expert_ids


def relative_to_script_dir(path, script_dir):
    try:
        return Path(path).resolve().relative_to(script_dir).as_posix()
    except ValueError:
        return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment-file", default="data/All_Assignment_Pairs.xlsx")
    parser.add_argument("--output-root", default="LLMJudge_results")
    parser.add_argument("--expert-ids", nargs="+", default=["all"])
    parser.add_argument("--max-pairs", type=int, default=45)
    parser.add_argument("--start-pair", type=int, default=1)
    parser.add_argument("--model-name", default=os.environ.get("EXPERT_MODEL_NAME", "gemini-3.5-flash-thinking"))
    parser.add_argument("--base-url", default=os.environ.get("EXPERT_BASE_URL", "https://svip-ip.xty.app/v1"))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("EXPERT_TIMEOUT", "600")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("EXPERT_TEMPERATURE", "0.0")))
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--retry-sleep-max", type=float, default=300.0)
    parser.add_argument("--fallback-minimal-system", action="store_true")
    parser.add_argument("--skip-failed", action="store_true")
    parser.add_argument("--run-label", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    assignment_path = Path(args.assignment_file)
    if not assignment_path.is_absolute():
        assignment_path = script_dir / assignment_path
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = script_dir / output_root

    expert_ids = resolve_expert_ids(args.expert_ids)
    for expert_id in expert_ids:
        if expert_id not in EXPERT_DOMAINS:
            raise SystemExit(f"Unknown expert id: {expert_id}")

    model_dir = output_root / safe_path_component(args.model_name)
    if args.run_label:
        model_dir = model_dir / safe_path_component(args.run_label)
    model_dir.mkdir(parents=True, exist_ok=True)
    comparison_output = model_dir / "comparison_results.xlsx"
    raw_jsonl_output = model_dir / "raw_model_calls.jsonl"
    failed_jsonl_output = model_dir / "failed_model_calls.jsonl"
    metadata_output = model_dir / "run_metadata.json"

    api_key = re.sub(r"\s+", "", os.environ.get("EXPERT_API_KEY") or "")
    if not api_key and not args.dry_run:
        raise SystemExit("Please set EXPERT_API_KEY in the environment.")

    records = parse_assignment_pairs(assignment_path)
    planned_records = [
        r
        for r in records
        if r["Expert_ID"] in expert_ids and r["Pair_Number"] >= args.start_pair
    ]
    planned_records = [
        r
        for expert_id in expert_ids
        for r in planned_records
        if r["Expert_ID"] == expert_id
    ]

    planned_by_expert = {}
    for expert_id in expert_ids:
        expert_records = [r for r in planned_records if r["Expert_ID"] == expert_id][: args.max_pairs]
        planned_by_expert[expert_id] = expert_records

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": args.model_name,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "assignment_file": relative_to_script_dir(assignment_path, script_dir),
        "output_directory": relative_to_script_dir(model_dir, script_dir),
        "expert_ids": expert_ids,
        "start_pair": args.start_pair,
        "max_pairs": args.max_pairs,
        "dry_run": args.dry_run,
        "resume": args.resume,
        "max_retries": args.max_retries,
        "retry_sleep": args.retry_sleep,
        "retry_sleep_max": args.retry_sleep_max,
        "fallback_minimal_system": args.fallback_minimal_system,
        "skip_failed": args.skip_failed,
        "planned_call_count": sum(len(v) for v in planned_by_expert.values()),
        "planned_comparison_rows": 3 * sum(len(v) for v in planned_by_expert.values()),
    }
    metadata_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.dry_run:
        print(f"Dry run only. Planned {metadata['planned_call_count']} model calls.")
        print(f"Output directory prepared: {model_dir}")
        print(f"Metadata written: {metadata_output}")
        return

    if raw_jsonl_output.exists() and not args.resume:
        backup_path = model_dir / f"raw_model_calls.previous_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        raw_jsonl_output.rename(backup_path)
        print(f"Existing raw_model_calls.jsonl moved to {backup_path}")

    if args.resume:
        completed_keys, comparison_rows, raw_rows = load_completed_calls(raw_jsonl_output, args.model_name)
        failed_keys = load_failed_calls(failed_jsonl_output)
        print(f"Resume enabled: loaded {len(completed_keys)} completed calls from {raw_jsonl_output}")
        if failed_keys:
            print(f"Resume enabled: loaded {len(failed_keys)} failed calls from {failed_jsonl_output}")
    else:
        completed_keys = set()
        failed_keys = set()
        comparison_rows = []
        raw_rows = []

    for expert_id in expert_ids:
        expert_records = planned_by_expert[expert_id]
        if not expert_records:
            raise SystemExit(f"No assignment rows found for {expert_id}.")

        system_prompt = build_system_prompt(expert_id)
        expert_label = anonymized_expert_label(expert_id)
        for record in expert_records:
            completed_key = (expert_id, record["Pair_Number"])
            if completed_key in completed_keys:
                continue
            if completed_key in failed_keys:
                continue
            hypothesis_a_id = normalize_hypothesis_id(record["Hypothesis_A_ID"])
            hypothesis_b_id = normalize_hypothesis_id(record["Hypothesis_B_ID"])
            hypothesis_a_content = record["Hypothesis_A_content"]
            hypothesis_b_content = record["Hypothesis_B_content"]
            user_prompt = build_user_prompt(hypothesis_a_content, hypothesis_b_content)
            used_fallback_system_prompt = False
            active_system_prompt = system_prompt
            failure = None
            try:
                content, raw_data, parsed, answers = call_and_parse_model_with_retries(
                    args.base_url,
                    api_key,
                    args.model_name,
                    args.timeout,
                    active_system_prompt,
                    user_prompt,
                    args.temperature,
                    args.max_retries,
                    args.retry_sleep,
                    args.retry_sleep_max,
                )
            except Exception as exc:
                if not args.fallback_minimal_system:
                    failure = exc
                else:
                    used_fallback_system_prompt = True
                    active_system_prompt = build_fallback_system_prompt()
                    print("Primary expert system prompt failed after all retries; trying minimal JSON-only system prompt without changing hypothesis text...")
                    try:
                        content, raw_data, parsed, answers = call_and_parse_model_with_retries(
                            args.base_url,
                            api_key,
                            args.model_name,
                            args.timeout,
                            active_system_prompt,
                            user_prompt,
                            args.temperature,
                            args.max_retries,
                            args.retry_sleep,
                            args.retry_sleep_max,
                        )
                    except Exception as fallback_exc:
                        failure = fallback_exc

            if failure is not None:
                if not args.skip_failed:
                    raise failure
                write_jsonl_line(
                    failed_jsonl_output,
                    {
                        "failed_at": datetime.now().isoformat(timespec="seconds"),
                        "expert_id": expert_id,
                        "expert_label": expert_label,
                        "model_name": args.model_name,
                        "pair_number": record["Pair_Number"],
                        "hypothesis_a_id": hypothesis_a_id,
                        "hypothesis_b_id": hypothesis_b_id,
                        "hypothesis_a_content": hypothesis_a_content,
                        "hypothesis_b_content": hypothesis_b_content,
                        "system_prompt": active_system_prompt,
                        "expert_system_prompt": system_prompt,
                        "used_fallback_system_prompt": used_fallback_system_prompt,
                        "error_type": type(failure).__name__,
                        "error_message": str(failure),
                    },
                )
                failed_keys.add(completed_key)
                print(f"Skipped failed {expert_id} pair {record['Pair_Number']} after retries; recorded in {failed_jsonl_output}")
                continue

            for key, dimension in DIMENSIONS:
                comparison_rows.append(
                    [
                        expert_id,
                        hypothesis_a_id,
                        hypothesis_b_id,
                        dimension,
                        answer_to_result(answers[key], hypothesis_a_id, hypothesis_b_id),
                    ]
                )

            raw_rows.append(
                [
                    expert_id,
                    expert_label,
                    args.model_name,
                    record["Pair_Number"],
                    hypothesis_a_id,
                    hypothesis_b_id,
                    answers["novelty"],
                    answers["significance"],
                    answers["testability"],
                    json.dumps(parsed, ensure_ascii=False),
                ]
            )
            write_jsonl_line(
                raw_jsonl_output,
                {
                    "completed_at": datetime.now().isoformat(timespec="seconds"),
                    "expert_id": expert_id,
                    "expert_label": expert_label,
                    "model_name": args.model_name,
                    "pair_number": record["Pair_Number"],
                    "hypothesis_a_id": hypothesis_a_id,
                    "hypothesis_b_id": hypothesis_b_id,
                    "hypothesis_a_content": hypothesis_a_content,
                    "hypothesis_b_content": hypothesis_b_content,
                    "hypothesis_a_content_original": record["Hypothesis_A_content"],
                    "hypothesis_b_content_original": record["Hypothesis_B_content"],
                    "system_prompt": active_system_prompt,
                    "expert_system_prompt": system_prompt,
                    "used_fallback_system_prompt": used_fallback_system_prompt,
                    "user_prompt": user_prompt,
                    "parsed_answers": answers,
                    "parsed_json": parsed,
                    "raw_message_content": content,
                    "raw_api_response": raw_data,
                },
            )
            print(f"Completed {expert_id} pair {record['Pair_Number']}: {answers}")
            if args.checkpoint_every and len(raw_rows) % args.checkpoint_every == 0:
                write_workbook(comparison_output, comparison_rows, raw_rows)
                print(f"Checkpoint written: {comparison_output}")
            if args.sleep:
                time.sleep(args.sleep)

    metadata["completed_at"] = datetime.now().isoformat(timespec="seconds")
    metadata["completed_call_count"] = len(raw_rows)
    metadata["completed_comparison_rows"] = len(comparison_rows)
    metadata["failed_call_count"] = len(failed_keys)
    metadata_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_workbook(comparison_output, comparison_rows, raw_rows)
    print(f"Wrote {comparison_output} with {len(comparison_rows)} comparison rows and {len(raw_rows)} raw rows.")
    print(f"Wrote raw model calls to {raw_jsonl_output}.")


if __name__ == "__main__":
    main()
