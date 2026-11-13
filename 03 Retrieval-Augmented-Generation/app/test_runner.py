
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
APP_SCRIPT = APP_DIR / "app.py"
OUTPUT_DIR = APP_DIR / "output"

DEFAULT_CORPUS = (
    APP_DIR.parent
    / "mc3-starter-kit"
    / "mc3-corpus"
)

QUESTIONS = [
    {
        "query": "What is the maximum junction temperature of the TQ-40?",
        "answer": "94",
        "aliases": ["94 C", "94°C", "94 degrees"],
        "citations": ["specs/tq40_datasheet_r2.pdf"],
    },
    {
        "query": "In which quarter does the TQ-60 enter customer sampling?",
        "answer": "Q3 FY27",
        "aliases": ["Q3FY27"],
        "citations": ["planning/roadmap_fy27.docx"],
    },
    {
        "query": "What is the part number of the field replaceable fan assembly for the TQ-40?",
        "answer": "ORR-FAN-2214-B",
        "aliases": [],
        "citations": ["support/rma_parts.xlsx"],
    },
    {
        "query": "Which firmware version fixed ticket ORR 1847?",
        "answer": "4.3.2",
        "aliases": ["Meridian 4.3.2"],
        "citations": ["support/bug_database.csv"],
    },
    {
        "query": "What error code is logged when the thermal throttle engages?",
        "answer": "E7731",
        "aliases": [],
        "citations": ["logs/prod_inference_2026-09-02.log"],
    },
    {
        "query": "What is the default batch timeout, in seconds, in the ingest service?",
        "answer": "180",
        "aliases": ["180 seconds", "180s"],
        "citations": ["engineering/ingest_service.py"],
    },
    {
        "query": "Which backplane pin carries THERM_ALERT# on the TQ-40?",
        "answer": "B14",
        "aliases": ["pin B14"],
        "citations": ["specs/backplane_pinout.png"],
    },
    {
        "query": "What board revision is printed on the asset label?",
        "answer": "REV-C2",
        "aliases": [],
        "citations": ["support/asset_label.jpg"],
    },
    {
        "query": "The production log shows a thermal throttle incident. Which firmware release fixed the underlying defect?",
        "answer": "4.3.2",
        "aliases": ["Meridian 4.3.2"],
        "citations": [
            "logs/prod_inference_2026-09-02.log",
            "support/bug_database.csv",
        ],
    },
    {
        "query": "What is the unit price of the TQ-40 at 10,000 unit volume?",
        "answer": "",
        "aliases": [],
        "citations": [],
    },
]


def normalize_answer(value):
    """Normalize formatting differences for local test comparison."""
    return re.sub(r"[\s\-\.·_]", "", value.upper())


def answer_matches(actual, expected, aliases):
    """Check the actual answer against the expected answer and aliases."""
    accepted_answers = [expected, *aliases]

    return normalize_answer(actual) in {
        normalize_answer(answer)
        for answer in accepted_answers
    }


def run_app(arguments, timeout):
    command = [
        sys.executable,
        str(APP_SCRIPT),
        *arguments,
    ]

    return subprocess.run(
        command,
        cwd=APP_DIR,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run the AMD OCR/RAG challenge questions locally."
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help="Path to the challenge corpus.",
    )

    args = parser.parse_args()
    corpus = args.corpus.resolve()

    if not corpus.is_dir():
        print(f"ERROR: Corpus directory does not exist: {corpus}")
        return 1

    if not APP_SCRIPT.is_file():
        print(f"ERROR: app.py not found: {APP_SCRIPT}")
        return 1

    print(f"Corpus: {corpus}")
    print(f"App:    {APP_SCRIPT}")
    print()

    # Phase 1: Index the corpus once.
    print("=" * 60)
    print("INDEXING CORPUS")
    print("=" * 60)

    try:
        result = run_app(
            ["--index", str(corpus)],
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        print("FAIL: Indexing exceeded 600 seconds.")
        return 1

    if result.returncode != 0:
        print("FAIL: Indexing failed.")
        print(result.stderr or result.stdout)
        return 1

    print("PASS: Indexing completed.")

    if result.stdout.strip():
        print(result.stdout.strip())

    # Phase 2: Ask each question in a separate app.py process.
    print()
    print("=" * 60)
    print("RUNNING QUESTIONS")
    print("=" * 60)

    passed = 0
    failed = 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for index, question in enumerate(QUESTIONS, start=1):
        query_id = f"query_{index:02d}"
        output_file = OUTPUT_DIR / f"{query_id}_output.json"

        # Avoid accidentally accepting a stale result.
        if output_file.exists():
            output_file.unlink()

        print(f"\n[{query_id}] {question['query']}")

        try:
            result = run_app(
                [
                    "--corpus", str(corpus),
                    "--query-id", query_id,
                    "--query", question["query"],
                ],
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            print("FAIL: Query exceeded 30 seconds.")
            failed += 1
            continue

        if result.returncode != 0:
            print("FAIL: app.py exited with an error.")
            print(result.stderr or result.stdout)
            failed += 1
            continue

        if not output_file.is_file():
            print(f"FAIL: Output file not found: {output_file}")
            failed += 1
            continue

        try:
            with output_file.open("r", encoding="utf-8") as file:
                actual = json.load(file)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"FAIL: Invalid output JSON: {exc}")
            failed += 1
            continue

        # Validate the required output structure.
        if not isinstance(actual, dict):
            print("FAIL: Output must be a JSON object.")
            failed += 1
            continue

        if not isinstance(actual.get("answer"), str):
            print("FAIL: Missing or invalid answer.")
            failed += 1
            continue

        if not isinstance(actual.get("citations"), list):
            print("FAIL: Missing or invalid citations.")
            failed += 1
            continue

        if not all(isinstance(c, str) for c in actual["citations"]):
            print("FAIL: Every citation must be a string.")
            failed += 1
            continue

        # Compare against the expected answer and accepted aliases.
        answer_ok = answer_matches(
            actual["answer"],
            question["answer"],
            question["aliases"],
        )

        # Citations must match as an exact set.
        citations_ok = (
            set(actual["citations"])
            == set(question["citations"])
        )

        if answer_ok and citations_ok:
            print("PASS")
            passed += 1
        else:
            print("FAIL")
            failed += 1

            if not answer_ok:
                print(f"  Expected answer: {question['answer']!r}")
                print(f"  Accepted aliases: {question['aliases']}")
                print(f"  Actual answer:   {actual['answer']!r}")

            if not citations_ok:
                print(f"  Expected citations: {question['citations']}")
                print(f"  Actual citations:   {actual['citations']}")

        if "confidence" in actual:
            print(f"  Confidence: {actual['confidence']}")

    # Summary
    print()
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{len(QUESTIONS)}")
    print(f"Failed: {failed}/{len(QUESTIONS)}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
