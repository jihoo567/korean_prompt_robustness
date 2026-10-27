import json
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


load_dotenv(Path(__file__).with_name(".env.local"))

client = OpenAI()


class Judgment(BaseModel):
    review: str
    score: Literal[1, 2, 3, 4, 5]


with open("tmanswers.jsonl", encoding="utf-8") as input_file, open(
    "gmanswers.jsonl", "w", encoding="utf-8"
) as output_file:
    for line in input_file:
        if not line.strip():
            continue

        test_answer = json.loads(line)

        response = client.responses.parse(
            model="gpt-6-luna",
            input=[
                {
                    "role": "system",
                    "content": (
                        "문제와 답변을 평가하세요. "
                        "완전히 정확하면 5점, 대체로 정확하면 4점, "
                        "부분적으로 정확하면 3점, 대부분 틀리면 2점, "
                        "완전히 틀리거나 무의미하면 1점을 주세요. "
                        "심사평은 한국어로 간결하게 작성하세요."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"문제: {test_answer['prompt']}\n"
                        f"답변: {test_answer['response']}"
                    ),
                },
            ],
            text_format=Judgment,
        )

        judgment = response.output_parsed
        result = {
            **test_answer,
            "review": judgment.review,
            "score": judgment.score,
        }
        output_file.write(json.dumps(result, ensure_ascii=False) + "\n")

print("완료: gmanswers.jsonl에 심사 결과를 저장했습니다.")
