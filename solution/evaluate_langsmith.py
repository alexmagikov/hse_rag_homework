"""Эксперимент в LangSmith: как eval/evaluator_langsmith.py, но судья соблюдает лимит LLM и повторяет запрос,
а у эксперимента есть название варианта и метаданные.

Из корня репозитория: python -m solution.evaluate_langsmith hw-rag-test --variant agentic
Датасет сначала загрузите: python -m eval.upload_langsmith data/test.jsonl hw-rag-test

Метрики те же, что в evaluator_langsmith (функция metrics), модель и промпт судьи не меняются.
"""

import argparse
from datetime import datetime, timezone
from time import perf_counter

from langsmith import Client

import eval.evaluator_langsmith as evaluator_langsmith
from eval.rag_example import CANDIDATES, DENSE_WEIGHT, MAX_SEARCHES, RAG, SimpleRAG
from solution.generate import LLM_MODEL
from solution.judge import judge


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_name")
    parser.add_argument("--variant", choices=["agentic", "simple"], default="agentic")
    args = parser.parse_args()

    evaluator_langsmith.score_answer = judge  # metrics() вызывает score_answer из своего модуля
    system = RAG() if args.variant == "agentic" else SimpleRAG()

    def answer_question(inputs):
        start = perf_counter()
        result = system.rag(inputs["question"])
        return {**result, "time": perf_counter() - start}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    result = Client().evaluate(
        answer_question,
        data=args.dataset_name,
        evaluators=[evaluator_langsmith.metrics],
        experiment_prefix=f"{args.variant}-{stamp}",
        metadata={"dataset": args.dataset_name, "variant": args.variant, "k": system.k, "llm": LLM_MODEL,
                  "candidates": CANDIDATES, "dense_weight": DENSE_WEIGHT,
                  "max_searches": MAX_SEARCHES if args.variant == "agentic" else None},
        max_concurrency=0,  # вопросы по одному: у LLM лимит 10 запросов в минуту
    )
    print("Эксперимент:", result.experiment_name)


if __name__ == "__main__":
    main()
