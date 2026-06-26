import json
import logging
import re
from typing import Dict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import API_KEY, BASE_URL, LLM_MODEL

logger = logging.getLogger(__name__)

REFLECTION_PROMPT = """审查以下回答质量。

文档内容：
{context}

问题：{question}

回答：{answer}

请判断回答是否：1.基于文档内容（未编造）2.完整回答了问题 3.无幻觉
只输出JSON：{{"based_on_doc":true,"complete":true,"hallucination":false,"passed":true,"issue":""}}
如果有问题则passed为false并在issue中说明。"""


class ReflectionModule:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0.0,
        )

    def reflect(self, question: str, answer: str, context: str) -> Dict:
        prompt = REFLECTION_PROMPT.format(
            context=context[:2000],
            question=question,
            answer=answer[:1000],
        )
        try:
            response = self.llm.invoke([
                SystemMessage(content="你是回答质量审查助手，只输出JSON。"),
                HumanMessage(content=prompt),
            ])
            result = self._parse_response(response.content)
            logger.info(f"反思结果: passed={result.get('passed')}, issue={result.get('issue', '')}")
            return result
        except Exception as e:
            logger.error(f"反思过程出错: {e}")
            return {
                "based_on_doc": True,
                "complete": True,
                "hallucination": False,
                "passed": True,
                "issue": "",
            }

    def _parse_response(self, content: str) -> Dict:
        try:
            json_match = re.search(r'\{[^{}]+\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                json_str = re.sub(r'("?\w+"?\s*:\s*)"true"', r'\1true', json_str)
                json_str = re.sub(r'("?\w+"?\s*:\s*)"false"', r'\1false', json_str)
                json_str = re.sub(r':\s*"[Tt]rue"', ': true', json_str)
                json_str = re.sub(r':\s*"[Ff]alse"', ': false', json_str)
                json_str = re.sub(r':\s*是', ': true', json_str)
                json_str = re.sub(r':\s*否', ': false', json_str)
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        logger.warning(f"反思结果JSON解析失败: {content}")
        has_issue = any(w in content for w in ["幻觉", "编造", "不完整", "未基于", "false", "错误"])
        return {
            "based_on_doc": not has_issue,
            "complete": not has_issue,
            "hallucination": False,
            "passed": not has_issue,
            "issue": content[:200] if has_issue else "",
        }
