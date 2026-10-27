import json

from ollama import chat


with open("problems.jsonl", encoding="utf-8") as input_file, open(
    "answers.jsonl", "w", encoding="utf-8"
) as output_file:
    for line in input_file:
        if not line.strip():
            continue

        problem = json.loads(line)

        response = chat(
            model="qwen3:1.7b", # mac: qwen3:1.7b, 연구컴: qwen2.5:14b
            messages=[{"role": "user", "content": problem["prompt"]}],
            think=False,
            options={"temperature": 0, "seed": 42, "num_predict": 512},
        )

        answer = {
            "id": str(problem["id"]),
            "prompt": problem["prompt"],
            "response": response.message.content,
        }
        output_file.write(json.dumps(answer, ensure_ascii=False) + "\n")

print("완료: answers.jsonl에 결과를 저장했습니다.")
