"""Unified rich-based logging. Replaces the scattered ``print("---Node:...")`` calls
in the original main.py while preserving the node-trace semantics."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console(stderr=True)
_configured = False


def setup_logging(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, markup=True, rich_tracebacks=True, show_path=False)],
    )
    _configured = True


def get_logger(name: str = "agentic_rag") -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def node(msg: str) -> None:
    """Log a graph-node trace line (preserves the old ---Node:...--- intent)."""
    get_logger().info(f"[bold cyan]●[/] {msg}")


def edge(msg: str) -> None:
    """Log a graph-edge / routing decision."""
    get_logger().info(f"[magenta]↳[/] {msg}")
