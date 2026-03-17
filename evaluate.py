import json
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

# Load dataset
with open("data.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

metric = GEval(
    name="Response Quality",
    criteria="""
Evaluate correctness, relevance, completeness, and clarity.
Higher score = better answer.
""",
    evaluation_params=[
    LLMTestCaseParams.INPUT,
    LLMTestCaseParams.ACTUAL_OUTPUT
]
)

results = []

for item in dataset:
    question = item["input"]

    chatgpt_resp = item["chatgpt_response"]
    myapp_resp = item["myapp_response"]

    test_chatgpt = LLMTestCase(input=question, actual_output=chatgpt_resp)
    test_myapp = LLMTestCase(input=question, actual_output=myapp_resp)

    score_chatgpt = metric.measure(test_chatgpt)
    score_myapp = metric.measure(test_myapp)

    winner = "chatgpt" if score_chatgpt > score_myapp else "myapp"

    results.append({
        "id": item["id"],
        "chatgpt_score": score_chatgpt,
        "myapp_score": score_myapp,
        "winner": winner
    })

for r in results:
    print(r)

