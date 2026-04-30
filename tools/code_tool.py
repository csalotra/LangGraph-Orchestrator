from langchain_core.tools import tool
import subprocess
import tempfile
import sys
import os
import textwrap

# ---- deny-list (safety layer) ----
FORBIDDEN_PATTERNS = [
    "import os",
    "import sys",
    "import subprocess",
    "import socket",
    "import requests",
    "from os",
    "from sys",
    "open(",
    "__import__",
    "eval(",
    "exec(",
]


def _is_safe(code: str) -> bool:
    lowered = code.lower()
    return not any(p in lowered for p in FORBIDDEN_PATTERNS)


@tool
def code_tool(code: str) -> str:
    """
    Executes Python code in an isolated subprocess and returns stdout or error output.

    The tool applies basic safety checks and enforces execution timeout.
    """
    try:
        # safety checks
        if not _is_safe(code):
            return "Unsafe code detected"

        if len(code) > 2000:
            return "Code too long"

        # normalize indentation
        code = textwrap.dedent(code).strip()

        lines = code.split("\n")
        if lines and not lines[-1].strip().startswith("print"):
            lines[-1] = f"print({lines[-1]})"
        code = "\n".join(lines)

        # execute in temp file
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "code.py")

            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)

            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=3,
                cwd=tmp,
                env=os.environ.copy(),
            )

            stdout = (result.stdout or "").strip()
            stderr = (result.stderr or "").strip()

            if result.returncode == 0:
                return stdout if stdout else "No output"
            else:
                return f"Error: {stderr}"

    except subprocess.TimeoutExpired:
        return "Execution timed out"

    except Exception as e:
        return f"Execution error: {str(e)}"