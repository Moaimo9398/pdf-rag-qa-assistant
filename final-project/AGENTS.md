# AGENTS.md - 编程智能体操作说明

## 项目概述
DeepLearnQA - 深度学习知识库智能问答系统，基于RAG技术构建。

## 技术栈
- Python 3.10+
- LangChain + ChromaDB + SiliconFlow API (OpenAI 兼容)
- DeepSeek-V4-Flash 模型
- BAAI/bge-large-zh-v1.5 嵌入模型
- PyMuPDF (fitz) PDF解析

## 项目结构
```
final-project/
├── src/
│   ├── config.py         # 配置管理（从.env读取）
│   ├── knowledge_base.py # 知识库构建与管理
│   ├── agent.py          # 核心智能体逻辑
│   ├── memory.py         # 对话记忆管理
│   ├── reflection.py     # 反思与质量校验
│   ├── main.py           # 程序入口（CLI + Gradio）
│   └── tools/
│       ├── pdf_parser.py      # PDF文档解析
│       ├── vector_retriever.py # 向量检索
│       └── code_executor.py   # 代码执行沙箱
├── tests/
│   ├── test_cases.py     # 测试用例
│   └── evaluate.py       # 评估脚本
├── data/                 # 知识库数据
└── docs/                 # 文档
```

## 运行命令
- 安装依赖: `pip install -r requirements.txt`
- 配置环境: 复制 `.env.example` 为 `.env` 并填入 API Key
- CLI 模式: `python src/main.py --mode cli`
- Gradio 模式: `python src/main.py --mode gradio`
- 重建知识库: `python src/main.py --rebuild`
- 运行评估: `python tests/evaluate.py`

## 关键设计模式
1. RAG 检索增强生成
2. 提示链（意图识别→检索→生成→反思）
3. 反思机制
4. 记忆管理
5. 工具路由
6. 异常处理
7. 安全护栏

## 注意事项
- 不得硬编码 API Key
- 所有异常必须有友好提示
- 拒绝回答与深度学习无关的问题
- 反思模块检测幻觉并重新生成
