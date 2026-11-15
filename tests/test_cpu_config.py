"""CPU-only configuration checks. No model downloads."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_embedding_model_is_forced_to_cpu():
    source = _read("backend/embedding.py")
    assert '"device": "cpu"' in source or "'device': 'cpu'" in source
    assert "cuda" not in source.lower()


def test_reranker_is_forced_to_cpu():
    source = _read("backend/reranker.py")
    assert 'device="cpu"' in source or "device='cpu'" in source
    assert "cuda" not in source.lower()


def test_requirements_do_not_pin_cuda_torch_wheels():
    text = _read("requirements.txt").lower()
    assert "cudatoolkit" not in text
    assert "+cu" not in text
    assert "nvidia-" not in text


def test_readme_documents_cpu_pytorch_install():
    readme = _read("README.md").lower()
    assert "cpu" in readme
    assert "pytorch.org/whl/cpu" in readme or "index-url" in readme


def test_health_ollama_probe_stays_short():
    source = _read("backend/main.py")
    tree = ast.parse(source)
    found = False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "DEFAULT_OLLAMA_TIMEOUT":
                    assert isinstance(node.value, ast.Constant)
                    assert float(node.value.value) <= 2.0
                    found = True
    assert found, "DEFAULT_OLLAMA_TIMEOUT missing"


def test_python_requires_310_compatible_syntax():
    """Smoke: evaluation package imports under the project Python."""

    from backend.evaluation.metrics import precision_at_k

    assert precision_at_k([True], k=1) == 1.0
