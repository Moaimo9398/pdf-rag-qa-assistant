import os
import subprocess
import tempfile
from typing import Dict


DANGEROUS_PATTERNS = [
    "import os", "import subprocess", "import sys", "import shutil",
    "os.", "subprocess.", "open(", "exec(", "eval(",
    "__import__", "globals()", "locals()", "vars()",
    "compile(", "memoryview", "ctypes",
]


class CodeExecutor:
    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def _is_safe(self, code: str) -> tuple:
        for pattern in DANGEROUS_PATTERNS:
            if pattern in code:
                return False, f"不允许执行包含 '{pattern}' 的代码"
        return True, ""

    def execute(self, code: str) -> Dict:
        is_safe, reason = self._is_safe(code)
        if not is_safe:
            return {"success": False, "output": "", "error": reason}

        prefix = "# 安全沙箱环境\nimport math\nimport random\nfrom collections import *\n"
        full_code = prefix + code

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(full_code)
                tmp_path = f.name

            result = subprocess.run(
                ["python", tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if result.returncode == 0:
                return {"success": True, "output": result.stdout, "error": ""}
            else:
                return {"success": False, "output": result.stdout, "error": result.stderr}
        except subprocess.TimeoutExpired:
            return {"success": False, "output": "", "error": f"代码执行超时（{self.timeout}秒）"}
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
