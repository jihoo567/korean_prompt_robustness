import json

from ollama import chat


with open("problems.json", encoding="utf-8") as input_file:
    problems = json.load(input_file)

with open("answers.jsonal", "w", encoding="utf-8") as output_file:
    for problem in problems:
        response = chat(
            model="qwen3:1.7b", # mac: qwen3:1.7b, 연구컴: qwen2.5:14b
            messages=[{"role": "user", "content": problem["question"]}],
            think=False,
            options={"temperature": 0, "seed": 42, "num_predict": 512},
        )

        answer = {
            "id": str(problem["id"]),
            "question": problem["question"],
            "model_answer": response.message.content[:200],
        }
        output_file.write(json.dumps(answer, ensure_ascii=False) + "\n")

print("완료: answers.jsonal에 결과를 저장했습니다.")
