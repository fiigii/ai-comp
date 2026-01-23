"""
Pass Manager Infrastructure

Provides the framework for running optimization passes on HIR.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any
import json

from .hir import HIRFunction, ForLoop, If, Statement


@dataclass
class PassConfig:
    """Configuration for a single pass."""
    name: str
    enabled: bool = True
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class PassMetrics:
    """Metrics collected by a pass during execution."""
    ir_size_before: int = 0
    ir_size_after: int = 0
    ssa_count_before: int = 0
    ssa_count_after: int = 0
    custom: dict[str, Any] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)


def count_statements(hir: HIRFunction) -> int:
    """Count total statements in HIR including nested structures."""
    def count_in_body(body: list[Statement]) -> int:
        total = 0
        for stmt in body:
            total += 1
            if isinstance(stmt, ForLoop):
                total += count_in_body(stmt.body)
            elif isinstance(stmt, If):
                total += count_in_body(stmt.then_body)
                total += count_in_body(stmt.else_body)
        return total
    return count_in_body(hir.body)


class Pass(ABC):
    """Base class for all HIR transformation passes."""

    def __init__(self):
        self._metrics: Optional[PassMetrics] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the pass name for config matching."""
        pass

    @abstractmethod
    def run(self, hir: HIRFunction, config: PassConfig) -> HIRFunction:
        """Transform HIR and return new HIRFunction."""
        pass

    def get_metrics(self) -> Optional[PassMetrics]:
        """Return metrics from the last run, if collected."""
        return self._metrics

    def _init_metrics(self):
        """Initialize metrics for a new run."""
        self._metrics = PassMetrics()

    def _add_metric_message(self, msg: str):
        """Add a diagnostic message to metrics."""
        if self._metrics:
            self._metrics.messages.append(msg)


@dataclass
class PassManager:
    """Manages and runs HIR transformation passes."""
    passes: list[Pass] = field(default_factory=list)
    config: dict[str, PassConfig] = field(default_factory=dict)
    print_after_all: bool = False
    print_metrics: bool = False

    def add_pass(self, p: Pass) -> None:
        """Register a pass."""
        self.passes.append(p)

    def load_config(self, config_path: str) -> None:
        """Load pass configs from JSON file."""
        with open(config_path) as f:
            data = json.load(f)
        for pass_name, opts in data.get("passes", {}).items():
            self.config[pass_name] = PassConfig(
                name=pass_name,
                enabled=opts.get("enabled", True),
                options=opts.get("options", {})
            )

    def _print_pass_metrics(self, p: Pass, cfg: PassConfig, before_size: int,
                            before_ssa: int, hir: HIRFunction):
        """Print metrics for a pass execution."""
        from .printing import print_hir

        after_size = count_statements(hir)
        after_ssa = hir.num_ssa_values

        print(f"\n=== Pass: {p.name} ===")
        print(f"Config: {', '.join(f'{k}={v}' for k, v in cfg.options.items()) or '(default)'}")

        # IR size change
        if before_size > 0:
            pct = ((after_size - before_size) / before_size) * 100
            print(f"IR size: {before_size} -> {after_size} statements ({pct:+.0f}%)")
        else:
            print(f"IR size: {before_size} -> {after_size} statements")

        # SSA count change
        if before_ssa > 0:
            pct = ((after_ssa - before_ssa) / before_ssa) * 100
            print(f"SSA values: {before_ssa} -> {after_ssa} ({pct:+.0f}%)")
        else:
            print(f"SSA values: {before_ssa} -> {after_ssa}")

        # Pass-specific metrics
        metrics = p.get_metrics()
        if metrics:
            if metrics.custom:
                print(f"Custom metrics: {metrics.custom}")
            if metrics.messages:
                print("Diagnostics:")
                for msg in metrics.messages:
                    print(f"  - {msg}")

    def run(self, hir: HIRFunction) -> HIRFunction:
        """Run all enabled passes in order."""
        from .printing import print_hir

        if self.print_after_all:
            print("=== HIR (before passes) ===")
            print_hir(hir)

        for p in self.passes:
            cfg = self.config.get(p.name, PassConfig(name=p.name))
            if not cfg.enabled:
                if self.print_metrics:
                    print(f"\n=== Pass: {p.name} === (SKIPPED - disabled)")
                continue

            # Collect before metrics
            before_size = count_statements(hir) if self.print_metrics else 0
            before_ssa = hir.num_ssa_values if self.print_metrics else 0

            hir = p.run(hir, cfg)

            # Print metrics
            if self.print_metrics:
                self._print_pass_metrics(p, cfg, before_size, before_ssa, hir)

            if self.print_after_all:
                print(f"=== HIR (after {p.name}) ===")
                print_hir(hir)

        return hir
