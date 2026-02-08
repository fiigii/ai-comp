"""
MIR Register Pressure Profiler

Computes per-bundle register pressure from live intervals (reusing regalloc's
liveness infrastructure) and optionally outputs an interactive HTML graph
showing the pressure curve over the program's lifetime.

This is a pure analysis pass — MIR is returned unmodified.
"""

from __future__ import annotations

import json
import os
from string import Template

from vm import SCRATCH_SIZE, VLEN

from ..pass_manager import MIRPass, PassConfig
from ..mir import MachineFunction
from .mir_register_allocation import (
    _detect_vector_bases,
    _compute_liveness,
    _build_live_intervals,
    LiveInterval,
)


def _compute_pressure_curve(
    mfunc: MachineFunction,
    intervals: list[LiveInterval],
) -> list[tuple[int, int, int]]:
    """Compute per-bundle register pressure via active-interval sweep.

    Returns list of (bundle_idx, scalar_pressure, vector_pressure) tuples.
    vector_pressure is measured in scalar-equivalent words (count * VLEN).
    """
    # Count total bundles
    block_order = mfunc.get_block_order()
    total_bundles = 0
    for name in block_order:
        block = mfunc.blocks[name]
        total_bundles += max(len(block.bundles), 1)

    if total_bundles == 0 or not intervals:
        return []

    # Sort intervals by start for sweep
    sorted_ivs = sorted(intervals, key=lambda iv: (iv.start, -iv.end))

    curve: list[tuple[int, int, int]] = []
    iv_idx = 0
    active: list[LiveInterval] = []

    for b in range(total_bundles):
        # Pressure is maximal at the def point (end of bundle)
        p = b * 2 + 1

        # Expire intervals that ended before this point
        active = [a for a in active if a.end >= p]

        # Add intervals that start at or before this point
        while iv_idx < len(sorted_ivs) and sorted_ivs[iv_idx].start <= p:
            active.append(sorted_ivs[iv_idx])
            iv_idx += 1

        scalar = 0
        vector = 0
        for a in active:
            if a.is_vector:
                vector += VLEN
            else:
                scalar += 1

        curve.append((b, scalar, vector))

    return curve


_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "tools")


def _generate_html(
    curve: list[tuple[int, int, int]],
    output_path: str,
    peak_pressure: int,
    peak_bundle: int,
    avg_pressure: float,
) -> None:
    """Generate a self-contained HTML file with a Canvas pressure chart."""
    template_path = os.path.join(_TEMPLATE_DIR, "pressure_profile.html")
    with open(template_path) as f:
        tmpl = Template(f.read())

    html = tmpl.substitute(
        peak_pressure=peak_pressure,
        peak_bundle=peak_bundle,
        avg_pressure=f"{avg_pressure:.1f}",
        scratch_size=SCRATCH_SIZE,
        total_bundles=len(curve),
        bundles_json=json.dumps([c[0] for c in curve]),
        scalars_json=json.dumps([c[1] for c in curve]),
        vectors_json=json.dumps([c[2] for c in curve]),
        vlen=VLEN,
    )

    with open(output_path, "w") as f:
        f.write(html)


class MIRRegPressureProfilerPass(MIRPass):
    """
    Analysis pass that computes per-bundle register pressure from live intervals.

    Reuses regalloc's liveness infrastructure to build live intervals, then
    sweeps them to produce a pressure curve. Optionally generates an HTML
    visualization of the pressure profile.

    This is a pure analysis pass — MIR is returned unmodified.
    """

    @property
    def name(self) -> str:
        return "mir-reg-pressure-profiler"

    def run(self, mir: MachineFunction, config: PassConfig) -> MachineFunction:
        self._init_metrics()

        html_output = bool(config.options.get("html_output", True))
        output_path = config.options.get("output_path", "pressure_profile.html")

        # Reuse regalloc infrastructure
        vector_bases, vector_addrs = _detect_vector_bases(mir)
        liveness = _compute_liveness(mir, vector_bases, vector_addrs)
        intervals = _build_live_intervals(mir, liveness, vector_bases, vector_addrs)

        n_scalar = sum(1 for iv in intervals if not iv.is_vector)
        n_vector = sum(1 for iv in intervals if iv.is_vector)

        # Compute pressure curve
        curve = _compute_pressure_curve(mir, intervals)

        # Derive stats
        if curve:
            peak_pressure = 0
            peak_bundle = 0
            total_pressure = 0
            for b, s, v in curve:
                total = s + v
                total_pressure += total
                if total > peak_pressure:
                    peak_pressure = total
                    peak_bundle = b
            avg_pressure = total_pressure / len(curve)
        else:
            peak_pressure = 0
            peak_bundle = 0
            avg_pressure = 0.0

        # Generate HTML visualization
        abs_output = os.path.abspath(output_path)
        if html_output:
            _generate_html(curve, abs_output, peak_pressure, peak_bundle, avg_pressure)
            self._add_metric_message(f"HTML pressure profile written to {abs_output}")

        # Report metrics
        if self._metrics:
            self._metrics.custom = {
                "peak_pressure": peak_pressure,
                "peak_bundle_idx": peak_bundle,
                "avg_pressure": round(avg_pressure, 1),
                "scalar_intervals": n_scalar,
                "vector_intervals": n_vector,
                "total_bundles": len(curve),
            }
            if html_output:
                self._metrics.custom["html_output_path"] = abs_output

        return mir
