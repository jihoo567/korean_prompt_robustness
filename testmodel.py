import json
import math
from datetime import datetime
from pathlib import Path

from ollama import chat


model = "qwen3:1.7b"  # mac: qwen3:1.7b, 연구컴: qwen2.5:14b

with open("problems.json", encoding="utf-8") as input_file:
    problems = json.load(input_file)

Path("result").mkdir(exist_ok=True)
model_name = model.replace(":", "-").replace("/", "_")
result_path = Path("result") / f"result_{model_name}_{datetime.now():%Y-%m-%d(%H-%M)}.jsonal"

with open("answers.jsonal", "w", encoding="utf-8") as output_file, open(
    result_path, "w", encoding="utf-8"
) as result_file:
    for problem in problems:
        choices = "\n".join(
            f"{label}. {text}"
            for label, text in zip(problem["choices"]["label"], problem["choices"]["text"])
        )
        prompt = f'{problem["question"]}\n{choices}\n정답 보기의 알파벳 하나만 답하세요.'

        scores = {}
        for label in "ABCD":
            response = chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                format={"type": "string", "enum": [label]},
                think=False,
                logprobs=True,
                options={"temperature": 0, "seed": 12, "num_predict": 16},
            )
            if not response.logprobs:
                raise RuntimeError("로그 확률이 반환되지 않았습니다. Ollama를 최신 버전으로 업데이트하세요.")
            # 토큰 분할에 관계없이 JSON 답변 전체의 로그 확률을 합산.
            scores[label] = sum(token.logprob for token in response.logprobs)

        # A, B, C, D 안에서의 상대 확률 (합계 1).
        weights = {label: math.exp(score - max(scores.values())) for label, score in scores.items()}
        total = sum(weights.values())
        probabilities = {label: weight / total for label, weight in weights.items()}
        model_answer = max(probabilities, key=probabilities.get)
        answer = {
            "id": str(problem["id"]),
            "question": problem["question"],
            "model_answer": model_answer,
            "probabilities": probabilities,
            "correct_answer": problem["answerKey"],
            "is_correct": model_answer == problem["answerKey"],
        }
        line = json.dumps(answer, ensure_ascii=False) + "\n"
        output_file.write(line)
        result_file.write(line)

print(f"완료: answers.jsonal과 {result_path}에 결과를 저장했습니다.")
