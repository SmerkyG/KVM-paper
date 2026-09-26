"""Run lm-eval RULER with the maintained HotpotQA parquet source.

The upstream task downloads HotpotQA from an obsolete CMU HTTP URL. The
dataset contents are available from the official Hugging Face dataset repo;
this wrapper replaces that download/parse function, supplies K2's required
empty assistant-reasoning field, and then delegates to the normal lm-eval CLI.
"""

from functools import cache
import sys

import datasets
import requests

from lm_eval.__main__ import cli_evaluate
from lm_eval.models.openai_completions import LocalChatCompletion
from lm_eval.tasks.ruler import qa_utils


OBSOLETE_HOTPOTQA_URL = (
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
)
HOTPOTQA_VALIDATION_PARQUET = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
    "distractor/validation-00000-of-00001.parquet"
)


@cache
def _hotpotqa_rows():
    return datasets.load_dataset(
        "parquet",
        data_files={"validation": HOTPOTQA_VALIDATION_PARQUET},
        split="validation",
    )


@cache
def _read_hotpotqa_from_huggingface() -> tuple[list[dict], list[str]]:
    rows = _hotpotqa_rows()
    documents = sorted(
        {
            f"{title}\n{''.join(sentences)}"
            for row in rows
            for title, sentences in zip(
                row["context"]["title"], row["context"]["sentences"]
            )
        }
    )
    document_indices = {document: index for index, document in enumerate(documents)}
    questions = []
    for row in rows:
        context = [
            document_indices[f"{title}\n{''.join(sentences)}"]
            for title, sentences in zip(
                row["context"]["title"], row["context"]["sentences"]
            )
        ]
        questions.append(
            {
                "query": row["question"],
                "outputs": [row["answer"]],
                "context": context,
            }
        )
    return questions, documents


class _HotpotQAResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> list[dict]:
        return [
            {
                "question": row["question"],
                "answer": row["answer"],
                "context": list(
                    zip(row["context"]["title"], row["context"]["sentences"])
                ),
            }
            for row in _hotpotqa_rows()
        ]


def main() -> None:
    qa_utils.read_hotpotqa = _read_hotpotqa_from_huggingface
    # The YAML !function loader imports this as a top-level module, while the
    # normal package import above uses lm_eval.tasks.ruler.qa_utils.
    sys.modules["qa_utils"] = qa_utils
    original_get = requests.get

    def get_with_hotpotqa_fallback(url, *args, **kwargs):
        if url == OBSOLETE_HOTPOTQA_URL:
            return _HotpotQAResponse()
        return original_get(url, *args, **kwargs)

    requests.get = get_with_hotpotqa_fallback

    # lm-eval 0.4.13 represents a task's ``gen_prefix`` as the content of a
    # final assistant message. K2's chat template requires every assistant
    # message to carry an explicit thinking field, including that unfinished
    # prefix. Preserve the prefix verbatim and select K2's empty-thought form.
    original_create_message = LocalChatCompletion.create_message

    def create_message_with_k2_thinking(self, messages, generate=False):
        rendered = original_create_message(self, messages, generate=generate)
        if "K2-Horizon" not in self.model or not isinstance(rendered, list):
            return rendered
        rendered = [dict(message) for message in rendered]
        thinking_fields = {
            "think",
            "reasoning",
            "reasoning_content",
            "think_fast",
            "think_faster",
        }
        for message in rendered:
            if message.get("role") == "assistant" and not thinking_fields & message.keys():
                message["reasoning_content"] = ""
        return rendered

    LocalChatCompletion.create_message = create_message_with_k2_thinking
    cli_evaluate()


if __name__ == "__main__":
    main()
