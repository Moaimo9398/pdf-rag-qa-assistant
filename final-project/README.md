# DeepLearnQA - 深度学习知识库智能问答系统

## 项目介绍

DeepLearnQA 是一个基于 RAG（检索增强生成）技术的深度学习知识库智能问答系统。系统基于《深度学习核心知识点》PDF 文档构建知识库，通过智能体技术实现准确、无幻觉的问答服务，支持多轮对话、知识点关联和学习路径推荐。

## 功能特点

- **RAG 检索增强生成**：所有回答基于提供的权威教材内容，杜绝幻觉
- **智能意图识别**：自动识别问题类型（概念查询、原理推导、对比分析等）
- **多轮对话**：支持上下文关联问答，记忆管理最近 10 轮对话
- **反思校验**：自动检测回答是否存在幻觉、是否完整，不通过则重新生成
- **安全护栏**：拒绝回答与深度学习无关的问题
- **来源标注**：每条回答标注来源页码和章节，便于查阅原文
- **知识点推荐**：回答后推荐相关知识点，助力深入学习
- **代码执行**：支持安全沙箱中运行 Python 代码示例

## 系统架构

```
用户提问 → 意图识别与分类 → 检索规划（生成检索关键词）
    → 向量库检索 → 检索结果重排序 → 上下文整合
    → 生成初步回答 → 质量反思与校验 → 修正回答
    → 输出结果 + 相关知识点推荐
```

## 安装步骤

### 1. 环境要求
- Python 3.10+
- 网络连接（调用 SiliconFlow API）

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置环境变量
```bash
# 复制环境变量示例文件
copy .env.example .env

# 编辑 .env 文件，填入你的 API Key
# API_KEY=你的API密钥
# BASE_URL=https://api.siliconflow.cn/v1
# LLM_MODEL=deepseek-ai/DeepSeek-V4-Flash
# EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
```

### 4. 准备知识库数据
将深度学习教材 PDF 文件放置在 `data/dl_know.pdf`。

## 运行方式

### CLI 命令行模式（默认）
```bash
cd final-project
python src/main.py --mode cli
```

### Gradio Web 界面模式
```bash
cd final-project
python src/main.py --mode gradio
```

### 强制重建知识库
```bash
python src/main.py --rebuild
```

### 运行评估脚本
```bash
cd final-project
python tests/evaluate.py
```

## 使用示例

### 示例 1：基础概念查询
```
你: 什么是 Transformer 架构？它的核心组件是什么？
AI: Transformer 是一种基于自注意力机制的序列处理模型...
    [来源: 第X页, 第X章]
    📚 相关知识点：• 自注意力机制详解 • 位置编码原理 • 编码器-解码器结构
```

### 示例 2：对比分析
```
你: 比较 CNN、RNN 和 Transformer 三种架构的区别和适用场景
AI: | 特点 | CNN | RNN | Transformer |
    |核心机制| 卷积 | 循环 | 自注意力 |
    ...
```

### 示例 3：复杂推理
```
你: 深度学习中为什么会出现梯度消失问题？有哪些解决方法？
AI: 梯度消失问题的原因是...
    解决方法包括：1. ReLU激活函数 2. 残差连接 3. LSTM/GRU ...
```

## 项目结构

```
final-project/
├── AGENTS.md              # 编程智能体操作说明
├── README.md              # 项目说明
├── report.md              # 结题报告
├── requirements.txt       # Python 依赖
├── .env.example           # 环境变量示例
├── screenshots/           # 运行截图
├── src/                   # 源代码
│   ├── __init__.py
│   ├── config.py          # 配置管理
│   ├── knowledge_base.py  # 知识库构建
│   ├── agent.py           # 核心智能体
│   ├── memory.py          # 对话记忆
│   ├── reflection.py      # 反思校验
│   ├── main.py            # 程序入口
│   └── tools/
│       ├── __init__.py
│       ├── pdf_parser.py       # PDF 解析
│       ├── vector_retriever.py # 向量检索
│       └── code_executor.py    # 代码执行
├── tests/                 # 测试与评估
│   ├── __init__.py
│   ├── test_cases.py      # 测试用例
│   └── evaluate.py        # 评估脚本
├── data/                  # 知识库数据
│   └── dl_know.pdf
└── docs/                  # 文档
```

## 设计模式

本项目应用了以下 7 种智能体设计模式：

1. **RAG 检索增强生成**：基于 PDF 构建向量知识库，所有回答基于检索结果
2. **提示链**：分阶段执行意图识别→检索规划→回答生成→质量反思
3. **反思机制**：生成回答后自动校验是否与检索内容一致，检测幻觉
4. **记忆管理**：保存对话历史，支持多轮上下文关联问答
5. **工具路由**：根据问题类型调用不同工具（PDF 解析/向量检索/代码执行）
6. **异常处理**：处理检索为空、模型调用失败、输入不合法等情况
7. **安全护栏**：拒绝回答与深度学习无关的问题，限制代码执行权限

## 测试用例

详见 `tests/test_cases.py`，包含 7 个测试用例，覆盖：
- 基础概念查询、原理理解、对比分析、复杂推理、无关问题拒绝、概念解释、多轮对话

## 注意事项

- 首次运行会自动解析 PDF 并构建向量知识库，耗时约 1-3 分钟
- 后续运行会加载已有知识库，启动更快
- 所有 API 密钥通过 `.env` 文件管理，请勿硬编码
- 回答仅供学习参考，如有疑问请查阅权威教材
