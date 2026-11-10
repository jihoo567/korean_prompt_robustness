import json
import re

from ollama import chat


with open("problems.json", encoding="utf-8") as input_file:
    problems = json.load(input_file)

with open("answers.jsonal", "w", encoding="utf-8") as output_file:
    for problem in problems:
        choices = "\n".join(
            f"{label}. {text}"
            for label, text in zip(problem["choices"]["label"], problem["choices"]["text"])
        )
        prompt = f'{problem["question"]}\n{choices}\n정답 보기의 알파벳 하나만 답하세요.'

        response = chat(
            model="qwen3:1.7b", # mac: qwen3:1.7b, 연구컴: qwen2.5:14b
            messages=[{"role": "user", "content": prompt}],
            think=False,
            options={"temperature": 0, "seed": 42, "num_predict": 16},
        )

        match = re.search(r"\b[A-D]\b", response.message.content.upper())
        model_answer = match.group() if match else response.message.content.strip()
        answer = {
            "id": str(problem["id"]),
            "question": problem["question"],
            "model_answer": model_answer,
            "correct_answer": problem["answerKey"],
            "is_correct": model_answer == problem["answerKey"],
        }
        output_file.write(json.dumps(answer, ensure_ascii=False) + "\n")

print("완료: answers.jsonal에 결과를 저장했습니다.")
