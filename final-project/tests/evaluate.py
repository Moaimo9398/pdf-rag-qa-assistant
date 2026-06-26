import sys
import os
import json
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import DeepLearnAgent
from tests.test_cases import TEST_CASES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_intent_accuracy(results):
    correct = 0
    total = len(results)
    for r in results:
        if r["actual_intent"] == r["expected_intent"]:
            correct += 1
    return correct / total if total > 0 else 0


def evaluate_keyword_coverage(results):
    scores = []
    for r in results:
        if not r["expected_keywords"]:
            continue
        answer_lower = r["answer"].lower()
        matched = sum(1 for kw in r["expected_keywords"] if kw.lower() in answer_lower)
        score = matched / len(r["expected_keywords"])
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0


def evaluate_off_topic_handling(results):
    for r in results:
        if r["expected_intent"] == "off_topic":
            return r["actual_intent"] == "off_topic"
    return True


def evaluate_response_time(results):
    times = [r["elapsed"] for r in results if r["elapsed"] > 0]
    return {
        "avg": round(sum(times) / len(times), 2) if times else 0,
        "min": round(min(times), 2) if times else 0,
        "max": round(max(times), 2) if times else 0,
    }


def evaluate_hallucination_rate(results):
    hallucinated = 0
    total = 0
    for r in results:
        if r.get("reflection") and r["expected_intent"] != "off_topic":
            total += 1
            if r["reflection"].get("has_hallucination", False):
                hallucinated += 1
    return hallucinated / total if total > 0 else 0


def evaluate_has_sources(results):
    with_sources = 0
    total = 0
    for r in results:
        if r["expected_intent"] != "off_topic":
            total += 1
            if r.get("sources") and len(r["sources"]) > 0:
                with_sources += 1
    return with_sources / total if total > 0 else 0


def run_evaluation():
    print("=" * 60)
    print("DeepLearnQA 评估脚本")
    print("=" * 60)

    agent = DeepLearnAgent()
    print("\n正在初始化知识库...")
    success = agent.initialize()
    if not success:
        print("知识库初始化失败！")
        return

    print("知识库初始化成功！\n")

    results = []
    for tc in TEST_CASES:
        print(f"运行测试用例 {tc['id']}: {tc['question']}")
        result = agent.run(tc["question"])
        results.append({
            "id": tc["id"],
            "category": tc["category"],
            "question": tc["question"],
            "expected_intent": tc["expected_intent"],
            "actual_intent": result.get("intent", ""),
            "expected_keywords": tc["expected_keywords"],
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "reflection": result.get("reflection", {}),
            "elapsed": result.get("elapsed", 0),
            "error": result.get("error"),
        })
        print(f"  意图: {result.get('intent')} | 耗时: {result.get('elapsed')}s")
        if result.get("error"):
            print(f"  错误: {result['error']}")

        if tc["id"] == 6:
            agent.memory.add_turn("什么是CNN？", "CNN是卷积神经网络，主要用于图像处理...")

    print("\n" + "=" * 60)
    print("评估结果汇总")
    print("=" * 60)

    intent_acc = evaluate_intent_accuracy(results)
    keyword_cov = evaluate_keyword_coverage(results)
    off_topic_ok = evaluate_off_topic_handling(results)
    time_stats = evaluate_response_time(results)
    halluc_rate = evaluate_hallucination_rate(results)
    source_rate = evaluate_has_sources(results)

    print(f"\n1. 意图识别准确率: {intent_acc:.1%}")
    print(f"2. 关键词覆盖率: {keyword_cov:.1%}")
    print(f"3. 无关问题拒绝: {'通过' if off_topic_ok else '未通过'}")
    print(f"4. 幻觉率: {halluc_rate:.1%}")
    print(f"5. 来源引用率: {source_rate:.1%}")
    print(f"6. 响应时间: 平均{time_stats['avg']}s, 最快{time_stats['min']}s, 最慢{time_stats['max']}s")

    print("\n各测试用例详情:")
    print("-" * 60)
    for r in results:
        status = "PASS" if r["actual_intent"] == r["expected_intent"] else "FAIL"
        print(f"[{status}] 用例{r['id']}: {r['category']}")
        print(f"  问题: {r['question']}")
        print(f"  预期意图: {r['expected_intent']} | 实际意图: {r['actual_intent']}")
        print(f"  来源数: {len(r['sources'])} | 耗时: {r['elapsed']}s")
        if r.get("error"):
            print(f"  错误: {r['error']}")
        print()

    report = {
        "summary": {
            "intent_accuracy": round(intent_acc, 4),
            "keyword_coverage": round(keyword_cov, 4),
            "off_topic_handling": off_topic_ok,
            "hallucination_rate": round(halluc_rate, 4),
            "source_citation_rate": round(source_rate, 4),
            "response_time": time_stats,
        },
        "details": results,
    }

    report_path = os.path.join(os.path.dirname(__file__), "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"评估报告已保存至: {report_path}")


if __name__ == "__main__":
    run_evaluation()
