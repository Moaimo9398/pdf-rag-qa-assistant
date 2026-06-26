import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import DeepLearnAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def cli_mode():
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()
    agent = DeepLearnAgent()

    console.print(Panel.fit(
        "[bold blue]DeepLearnQA - 深度学习知识库智能问答系统[/bold blue]\n"
        "[dim]基于RAG技术 · 多轮对话 · 反思校验 · 知识点关联[/dim]",
        border_style="blue",
    ))

    with console.status("[bold green]正在初始化知识库...[/bold green]"):
        success = agent.initialize()

    if not success:
        console.print("[bold red]知识库初始化失败，请检查PDF路径和API配置[/bold red]")
        return

    console.print("[bold green]知识库初始化成功！[/bold green]")
    console.print("[dim]输入问题开始提问 | quit/exit 退出 | clear 清空历史 | rebuild 重建知识库[/dim]\n")

    while True:
        try:
            question = console.input("[bold green]你: [/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]再见！[/dim]")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            console.print("[dim]再见！[/dim]")
            break
        if question.lower() == "clear":
            agent.clear_memory()
            console.print("[yellow]对话历史已清空[/yellow]\n")
            continue
        if question.lower() == "rebuild":
            with console.status("[bold yellow]正在重建知识库...[/bold yellow]"):
                agent.initialize(force_rebuild=True)
            console.print("[bold green]知识库重建完成！[/bold green]\n")
            continue

        with console.status("[bold cyan]思考中...[/bold cyan]"):
            result = agent.run(question)

        console.print()
        console.print(Panel(
            Markdown(result["answer"]),
            title="[bold cyan]AI回答[/bold cyan]",
            border_style="cyan",
        ))

        if result.get("sources"):
            source_table = Table(title="参考文献", show_header=True, header_style="bold magenta")
            source_table.add_column("序号", style="dim", width=4)
            source_table.add_column("页码", width=6)
            source_table.add_column("章节", width=20)
            source_table.add_column("相关度", width=8)
            source_table.add_column("内容预览", width=45)
            for i, src in enumerate(result["sources"], 1):
                source_table.add_row(
                    str(i),
                    str(src["page_num"]),
                    src["chapter"][:18],
                    str(src["relevance_score"]),
                    src["content"][:40] + "...",
                )
            console.print(source_table)

        meta = Text()
        meta.append(f"意图: {result.get('intent', 'N/A')} | ")
        meta.append(f"耗时: {result.get('elapsed', 0)}s | ")
        reflection = result.get("reflection", {})
        meta.append(f"反思: {'通过' if reflection.get('passed', True) else '未通过'}")
        console.print(f"\n[dim]{meta}[/dim]\n")


def gradio_mode():
    import gradio as gr

    agent = DeepLearnAgent()

    if not agent.initialize():
        print("知识库初始化失败")
        return

    def chat_fn(message, history):
        agent.memory.clear()
        for h in history:
            role = h.get("role", "")
            content = h.get("content", "")
            if role in ("user", "assistant"):
                agent.memory.history.append({"role": role, "content": content})

        result = agent.run(message)
        answer = result["answer"]

        if result.get("sources"):
            answer += "\n\n--- 参考来源 ---"
            for i, src in enumerate(result["sources"], 1):
                p = src["page_num"]
                ch = src["chapter"]
                sc = src["relevance_score"]
                answer += f"\n[{i}] 第{p}页 | {ch} | 相关度:{sc}"

        if result.get("related"):
            answer += "\n\n📚 相关知识点："
            for item in result["related"]:
                answer += f"\n- {item}"

        return answer

    demo = gr.ChatInterface(
        chat_fn,
        title="DeepLearnQA - 深度学习知识库智能问答",
        description="基于RAG技术的深度学习知识库问答系统，支持多轮对话、反思校验和知识点关联。",
    )
    demo.launch(share=False)


def main():
    parser = argparse.ArgumentParser(description="DeepLearnQA - 深度学习知识库智能问答系统")
    parser.add_argument("--mode", choices=["cli", "gradio"], default="cli", help="运行模式：cli 或 gradio")
    parser.add_argument("--rebuild", action="store_true", help="强制重建知识库")
    args = parser.parse_args()

    if args.mode == "gradio":
        gradio_mode()
    else:
        cli_mode()


if __name__ == "__main__":
    main()
