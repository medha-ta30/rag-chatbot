import os
import sys
import time
import json
import csv
import argparse

import config
from services.logger import logger
from services.retrieval import retrieve_relevant_chunks
from services.prompt_builder import build_prompt
from services.llm_service import generate_response


def run_evaluation(model_choice: str = "gemini"):
    """
    Runs the evaluation suite over evaluation/questions.json and generates
    evaluation_report.csv and evaluation_report.json.

    Args:
        model_choice (str): Choice of model ('gemini' or 'qwen').
    """
    model_name = config.MODEL_GEMINI if model_choice.lower() == "gemini" else config.MODEL_QWEN

    dataset_path = os.path.join("evaluation", "questions.json")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Evaluation questions file not found at {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    logger.info("Evaluation started question_count=%d model=%s", len(questions), model_name)
    eval_start_time = time.time()

    results_data = []

    for item in questions:
        q_id = item.get("id")
        category = item.get("category", "in_domain")
        question_text = item.get("question", "")
        expected_answer = item.get("expected_answer", "")

        q_start = time.time()

        # Step 1: Retrieval & Confidence Check
        try:
            chunks, confidence_passed, confidence_reason = retrieve_relevant_chunks(
                question_text, 
                top_k=config.DEFAULT_TOP_K
            )
        except Exception as e:
            chunks, confidence_passed, confidence_reason = [], False, f"Retrieval error: {str(e)}"

        retrieved_chunks_count = len(chunks)

        # Step 2: Prompt Construction & LLM Response (if confidence passed)
        if confidence_passed and chunks:
            try:
                full_prompt = build_prompt(question_text, chunks)
                generated_answer = generate_response(full_prompt, model_name)
            except Exception as e:
                generated_answer = f"Error during generation: {str(e)}"
        else:
            generated_answer = "I couldn't find enough relevant information in the uploaded document(s) to answer that question."

        latency_ms = round((time.time() - q_start) * 1000, 2)
        citation_present = "Yes" if ("Source:" in generated_answer or "(Source:" in generated_answer) else "No"

        row = {
            "Question": question_text,
            "Category": category,
            "Retrieved Chunks": retrieved_chunks_count,
            "Confidence Passed": confidence_passed,
            "Citation Present": citation_present,
            "Latency (ms)": latency_ms,
            "Model Used": model_name,
            "Generated Answer": generated_answer,
            "Expected Answer": expected_answer
        }
        results_data.append(row)

    total_time = round(time.time() - eval_start_time, 2)
    logger.info("Evaluation completed question_count=%d total_time=%.2fs model=%s", len(questions), total_time, model_name)

    # Save report as JSON
    json_path = os.path.join("evaluation", "evaluation_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    # Save report as CSV
    csv_path = os.path.join("evaluation", "evaluation_report.csv")
    fieldnames = [
        "Question", "Category", "Retrieved Chunks", "Confidence Passed",
        "Citation Present", "Latency (ms)", "Model Used",
        "Generated Answer", "Expected Answer"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_data)

    print(f"PASS: Evaluation complete. Reports generated at '{csv_path}' and '{json_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG Chatbot Evaluation Framework.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="gemini", 
        choices=["gemini", "qwen"],
        help="Target model to evaluate ('gemini' or 'qwen'). Defaults to 'gemini'."
    )
    args = parser.parse_args()
    run_evaluation(args.model)
