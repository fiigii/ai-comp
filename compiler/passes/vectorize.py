"""
Local Vectorization Pass (HIR)

Performs a simple local vectorization on straight-line HIR blocks by
combining repeated scalar ALU hash chains across unrolled iterations
into vector operations (VLEN lanes). This is a best-effort, pattern-
based transform intended to run after CSE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from problem import VLEN

from ..hir import (
    SSAValue,
    VectorSSAValue,
    Const,
    Op,
    ForLoop,
    If,
    Statement,
    HIRFunction,
)
from ..pass_manager import Pass, PassConfig


VECTOR_OP_MAP = {
    "+": "v+",
    "-": "v-",
    "*": "v*",
    "//": "v//",
    "%": "v%",
    "^": "v^",
    "&": "v&",
    "|": "v|",
    "<<": "v<<",
    ">>": "v>>",
    "<": "v<",
    "==": "v==",
}


@dataclass
class _VecContext:
    next_vec_id: int

    def new_vec(self, name: Optional[str] = None) -> VectorSSAValue:
        v = VectorSSAValue(self.next_vec_id, name)
        self.next_vec_id += 1
        return v


class VectorizePass(Pass):
    """
    Local vectorization pass that targets repeated scalar hash chains
    within unrolled loop bodies and replaces them with vector ops.
    """

    @property
    def name(self) -> str:
        return "vectorize"

    def __init__(self):
        super().__init__()
        self._groups_vectorized = 0
        self._ops_vectorized = 0

    def run(self, hir: HIRFunction, config: PassConfig) -> HIRFunction:
        self._init_metrics()
        self._groups_vectorized = 0
        self._ops_vectorized = 0

        if not config.enabled:
            return hir

        ctx = _VecContext(hir.num_vec_ssa_values)
        new_body = self._vectorize_statements(hir.body, ctx)

        if self._metrics:
            self._metrics.custom = {
                "groups_vectorized": self._groups_vectorized,
                "scalar_ops_replaced": self._ops_vectorized,
                "new_vec_ssa_values": ctx.next_vec_id - hir.num_vec_ssa_values,
            }

        return HIRFunction(
            name=hir.name,
            body=new_body,
            num_ssa_values=hir.num_ssa_values,
            num_vec_ssa_values=ctx.next_vec_id,
        )

    def _vectorize_statements(
        self, stmts: list[Statement], ctx: _VecContext
    ) -> list[Statement]:
        processed: list[Statement] = []

        for stmt in stmts:
            if isinstance(stmt, ForLoop):
                new_body = self._vectorize_statements(stmt.body, ctx)
                processed.append(
                    ForLoop(
                        counter=stmt.counter,
                        start=stmt.start,
                        end=stmt.end,
                        iter_args=stmt.iter_args,
                        body_params=stmt.body_params,
                        body=new_body,
                        yields=stmt.yields,
                        results=stmt.results,
                        pragma_unroll=stmt.pragma_unroll,
                    )
                )
            elif isinstance(stmt, If):
                new_then = self._vectorize_statements(stmt.then_body, ctx)
                new_else = self._vectorize_statements(stmt.else_body, ctx)
                processed.append(
                    If(
                        cond=stmt.cond,
                        then_body=new_then,
                        then_yields=stmt.then_yields,
                        else_body=new_else,
                        else_yields=stmt.else_yields,
                        results=stmt.results,
                    )
                )
            else:
                processed.append(stmt)

        return self._vectorize_op_segments(processed, ctx)

    def _vectorize_op_segments(
        self, stmts: list[Statement], ctx: _VecContext
    ) -> list[Statement]:
        result: list[Statement] = []
        segment: list[Op] = []

        def flush():
            nonlocal segment
            if segment:
                result.extend(self._vectorize_ops_segment(segment, ctx))
                segment = []

        for stmt in stmts:
            if isinstance(stmt, Op):
                segment.append(stmt)
            else:
                flush()
                result.append(stmt)
        flush()
        return result

    def _vectorize_ops_segment(self, ops: list[Op], ctx: _VecContext) -> list[Statement]:
        idx_positions = [
            i
            for i, op in enumerate(ops)
            if op.result is not None and op.result.name == "idx_addr"
        ]

        if len(idx_positions) < VLEN:
            return ops

        starts: list[int] = []
        prev = -1
        for pos in idx_positions:
            window_start = prev + 1
            defs: dict[int, int] = {}
            for j in range(window_start, pos + 1):
                opj = ops[j]
                if opj.result is not None:
                    defs[opj.result.id] = j

            seed_ids: list[int] = []
            for operand in ops[pos].operands:
                if isinstance(operand, SSAValue):
                    seed_ids.append(operand.id)

            min_idx = pos
            visited: set[int] = set()
            stack = list(seed_ids)
            while stack:
                sid = stack.pop()
                if sid in visited:
                    continue
                visited.add(sid)
                def_idx = defs.get(sid)
                if def_idx is None:
                    continue
                if def_idx < min_idx:
                    min_idx = def_idx
                for opnd in ops[def_idx].operands:
                    if isinstance(opnd, SSAValue):
                        stack.append(opnd.id)

            starts.append(min_idx)
            prev = pos

        starts = sorted(set(starts))
        if len(starts) < VLEN:
            return ops

        pre_ops = ops[: starts[0]]
        chunks: list[list[Op]] = []
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(ops)
            chunks.append(ops[start:end])

        out: list[Statement] = list(pre_ops)

        i = 0
        while i + VLEN <= len(chunks):
            group = chunks[i : i + VLEN]
            vec_group = self._vectorize_full_group(group, ctx)
            if vec_group is None:
                vec_group = self._vectorize_hash_chain_group(group, ctx)
            if vec_group is None:
                for c in group:
                    out.extend(c)
            else:
                out.extend(vec_group)
            i += VLEN

        for c in chunks[i:]:
            out.extend(c)

        return out

    def _vectorize_full_group(
        self, chunks: list[list[Op]], ctx: _VecContext
    ) -> Optional[list[Statement]]:
        """Vectorize an entire group of VLEN lanes using vload/vstore."""
        if len(chunks) != VLEN:
            return None

        base_idx_addr = self._find_result_op(chunks[0], "idx_addr")
        base_val_addr = self._find_result_op(chunks[0], "val_addr")
        base_node_addr = self._find_result_op(chunks[0], "node_addr")
        base_xored = self._find_result_op(chunks[0], "xored")
        base_h5 = self._find_result_op(chunks[0], "h5_val")
        base_mod = self._find_result_op(chunks[0], "mod_val")
        base_final = self._find_result_op(chunks[0], "final_idx")

        if (
            base_idx_addr is None
            or base_val_addr is None
            or base_node_addr is None
            or base_xored is None
            or base_h5 is None
            or base_mod is None
            or base_final is None
        ):
            return None

        x_idx = self._find_result_index(chunks[0], "xored")
        h5_idx = self._find_result_index(chunks[0], "h5_val")
        mod_idx = self._find_result_index_from(chunks[0], "mod_val", start=h5_idx + 1)
        fin_idx = self._find_result_index_from(chunks[0], "final_idx", start=h5_idx + 1)
        if x_idx is None or h5_idx is None or mod_idx is None or fin_idx is None:
            return None
        if not (x_idx < h5_idx < mod_idx < fin_idx):
            return None

        hash_chain0 = chunks[0][x_idx + 1 : h5_idx + 1]
        post_chain0 = chunks[0][mod_idx : fin_idx + 1]

        for op in hash_chain0 + post_chain0:
            if op.engine != "alu" or op.opcode not in VECTOR_OP_MAP:
                return None
            if len(op.operands) != 2:
                return None

        for chunk in chunks:
            xi = self._find_result_index(chunk, "xored")
            hi = self._find_result_index(chunk, "h5_val")
            mi = self._find_result_index_from(chunk, "mod_val", start=hi + 1)
            fi = self._find_result_index_from(chunk, "final_idx", start=hi + 1)
            if xi is None or hi is None or mi is None or fi is None:
                return None
            if (hi - xi) != (h5_idx - x_idx) or (fi - mi) != (fin_idx - mod_idx):
                return None

        idx_vals = []
        for chunk in chunks:
            idx_op = self._find_result_op(chunk, "idx")
            na_op = self._find_result_op(chunk, "node_addr")
            nv_op = self._find_result_op(chunk, "node_val")
            if idx_op is None or na_op is None or nv_op is None:
                return None
            idx_vals.append(idx_op.result)

        if any(v is None for v in idx_vals):
            return None

        if not base_node_addr.operands or not isinstance(base_node_addr.operands[0], SSAValue):
            return None
        forest_base = base_node_addr.operands[0]

        out: list[Statement] = []

        needed_ids = {base_idx_addr.operands[0].id if isinstance(base_idx_addr.operands[0], SSAValue) else None,
                      base_idx_addr.operands[1].id if isinstance(base_idx_addr.operands[1], SSAValue) else None,
                      base_val_addr.operands[0].id if isinstance(base_val_addr.operands[0], SSAValue) else None,
                      base_val_addr.operands[1].id if isinstance(base_val_addr.operands[1], SSAValue) else None}
        needed_ids = {i for i in needed_ids if i is not None}

        for op in chunks[0]:
            if op is base_idx_addr or op is base_val_addr:
                out.append(op)
                continue
            if op.result is not None and op.result.id in needed_ids:
                out.append(op)

        vec_idx = ctx.new_vec("vec_idx")
        vec_val = ctx.new_vec("vec_val")
        out.append(Op("vload", vec_idx, [base_idx_addr.result], "load"))
        out.append(Op("vload", vec_val, [base_val_addr.result], "load"))

        vec_forest = ctx.new_vec("vec_forest")
        out.append(Op("vbroadcast", vec_forest, [forest_base], "valu"))
        vec_addr = ctx.new_vec("vec_node_addr")
        out.append(Op("v+", vec_addr, [vec_forest, vec_idx], "valu"))
        vec_node = ctx.new_vec("vec_node")
        out.append(Op("vgather", vec_node, [vec_addr], "load"))
        vec_xored = ctx.new_vec("vec_xored")
        out.append(Op("v^", vec_xored, [vec_val, vec_node], "valu"))

        vec_ops, vec_h5 = self._emit_vector_chain_from_vec(
            vec_xored, base_xored.result, hash_chain0, ctx
        )
        if vec_ops is None or vec_h5 is None:
            return None
        out.extend(vec_ops)

        post_ops, vec_final = self._emit_vector_chain_with_seeds(
            {base_xored.result.id: vec_xored, base_h5.result.id: vec_h5, idx_vals[0].id: vec_idx},
            post_chain0,
            ctx,
        )
        if post_ops is None or vec_final is None:
            return None
        out.extend(post_ops)

        out.append(Op("vstore", None, [base_idx_addr.result, vec_final], "store"))
        out.append(Op("vstore", None, [base_val_addr.result, vec_h5], "store"))

        self._groups_vectorized += 1
        self._ops_vectorized += (len(hash_chain0) + len(post_chain0) + 6) * VLEN

        return out

    def _emit_vector_chain_from_vec(
        self,
        vec_seed: VectorSSAValue,
        seed_scalar: SSAValue,
        chain0: list[Op],
        ctx: _VecContext,
    ) -> tuple[Optional[list[Op]], Optional[VectorSSAValue]]:
        vec_map: dict[int, VectorSSAValue] = {seed_scalar.id: vec_seed}
        const_vec_cache: dict[object, VectorSSAValue] = {}
        ops: list[Op] = []

        for op in chain0:
            vec_operands: list[VectorSSAValue] = []
            for operand in op.operands:
                if isinstance(operand, SSAValue):
                    if operand.id in vec_map:
                        vec_operands.append(vec_map[operand.id])
                    else:
                        vec_const = const_vec_cache.get(operand.id)
                        if vec_const is None:
                            vec_const = ctx.new_vec(
                                f"vb_{operand.name}" if operand.name else None
                            )
                            ops.append(Op("vbroadcast", vec_const, [operand], "valu"))
                            const_vec_cache[operand.id] = vec_const
                        vec_operands.append(vec_const)
                elif isinstance(operand, Const):
                    key = ("const", operand.value)
                    vec_const = const_vec_cache.get(key)
                    if vec_const is None:
                        vec_const = ctx.new_vec(f"vb_{operand.value}")
                        ops.append(Op("vbroadcast", vec_const, [operand], "valu"))
                        const_vec_cache[key] = vec_const
                    vec_operands.append(vec_const)
                else:
                    return None, None

            vec_opcode = VECTOR_OP_MAP.get(op.opcode)
            if vec_opcode is None:
                return None, None

            vec_res = ctx.new_vec(op.result.name if op.result else None)
            ops.append(Op(vec_opcode, vec_res, vec_operands, "valu"))
            if op.result is not None:
                vec_map[op.result.id] = vec_res

        last_res = chain0[-1].result
        if last_res is None or last_res.id not in vec_map:
            return None, None
        return ops, vec_map[last_res.id]

    def _emit_vector_chain_with_seeds(
        self,
        seed_map: dict[int, VectorSSAValue],
        chain0: list[Op],
        ctx: _VecContext,
    ) -> tuple[Optional[list[Op]], Optional[VectorSSAValue]]:
        vec_map = dict(seed_map)
        const_vec_cache: dict[object, VectorSSAValue] = {}
        ops: list[Op] = []

        for op in chain0:
            vec_operands: list[VectorSSAValue] = []
            for operand in op.operands:
                if isinstance(operand, SSAValue):
                    if operand.id in vec_map:
                        vec_operands.append(vec_map[operand.id])
                    else:
                        vec_const = const_vec_cache.get(operand.id)
                        if vec_const is None:
                            vec_const = ctx.new_vec(
                                f"vb_{operand.name}" if operand.name else None
                            )
                            ops.append(Op("vbroadcast", vec_const, [operand], "valu"))
                            const_vec_cache[operand.id] = vec_const
                        vec_operands.append(vec_const)
                elif isinstance(operand, Const):
                    key = ("const", operand.value)
                    vec_const = const_vec_cache.get(key)
                    if vec_const is None:
                        vec_const = ctx.new_vec(f"vb_{operand.value}")
                        ops.append(Op("vbroadcast", vec_const, [operand], "valu"))
                        const_vec_cache[key] = vec_const
                    vec_operands.append(vec_const)
                else:
                    return None, None

            vec_opcode = VECTOR_OP_MAP.get(op.opcode)
            if vec_opcode is None:
                return None, None

            vec_res = ctx.new_vec(op.result.name if op.result else None)
            ops.append(Op(vec_opcode, vec_res, vec_operands, "valu"))
            if op.result is not None:
                vec_map[op.result.id] = vec_res

        last_res = chain0[-1].result
        if last_res is None or last_res.id not in vec_map:
            return None, None
        return ops, vec_map[last_res.id]

    def _vectorize_hash_chain_group(
        self, chunks: list[list[Op]], ctx: _VecContext
    ) -> Optional[list[Statement]]:
        xored_idx: list[int] = []
        h5_idx: list[int] = []
        for chunk in chunks:
            xi = self._find_result_index(chunk, "xored")
            hi = self._find_result_index(chunk, "h5_val")
            if xi is None or hi is None or hi <= xi:
                return None
            xored_idx.append(xi)
            h5_idx.append(hi)

        chain_len = h5_idx[0] - xored_idx[0]
        for xi, hi in zip(xored_idx, h5_idx):
            if hi - xi != chain_len:
                return None

        chain0 = chunks[0][xored_idx[0] + 1 : h5_idx[0] + 1]
        if not chain0:
            return None

        for op in chain0:
            if op.engine != "alu" or op.opcode not in VECTOR_OP_MAP:
                return None
            if len(op.operands) != 2:
                return None

        for chunk, xi, hi in zip(chunks, xored_idx, h5_idx):
            chain = chunk[xi + 1 : hi + 1]
            if len(chain) != len(chain0):
                return None
            for op, tmpl in zip(chain, chain0):
                if op.opcode != tmpl.opcode or op.engine != tmpl.engine:
                    return None
                if len(op.operands) != len(tmpl.operands):
                    return None

        for chunk, xi, hi in zip(chunks, xored_idx, h5_idx):
            suffix = chunk[hi + 1 :]
            used = self._collect_used_ids(suffix)
            for op in chunk[xi + 1 : hi]:
                if op.result and op.result.id in used:
                    return None

        prefixes = [chunk[: xi + 1] for chunk, xi in zip(chunks, xored_idx)]
        xored_vals = [chunk[xi].result for chunk, xi in zip(chunks, xored_idx)]
        h5_vals = [chunk[hi].result for chunk, hi in zip(chunks, h5_idx)]

        if any(v is None for v in xored_vals) or any(v is None for v in h5_vals):
            return None

        vec_ops, vec_h5 = self._emit_vector_hash_chain(xored_vals, chain0, ctx)
        if vec_ops is None or vec_h5 is None:
            return None

        extract_h5 = [Op("vextract", h5_vals[i], [vec_h5, Const(i)], "alu") for i in range(VLEN)]

        post_vec_ops, extract_final, suffixes = self._vectorize_post_chain(
            chunks, h5_idx, vec_h5, ctx
        )
        if post_vec_ops is None:
            post_vec_ops = []
            extract_final = []
            suffixes = [chunk[hi + 1 :] for chunk, hi in zip(chunks, h5_idx)]

        out: list[Statement] = []
        for pref in prefixes:
            out.extend(pref)
        out.extend(vec_ops)
        out.extend(extract_h5)
        out.extend(post_vec_ops)
        out.extend(extract_final)
        for suf in suffixes:
            out.extend(suf)

        self._groups_vectorized += 1
        self._ops_vectorized += len(chain0) * VLEN

        return out

    def _emit_vector_hash_chain(
        self, xored_vals: list[SSAValue], chain0: list[Op], ctx: _VecContext
    ) -> tuple[Optional[list[Op]], Optional[VectorSSAValue]]:
        ops: list[Op] = []
        vec_map: dict[int, VectorSSAValue] = {}
        const_vec_cache: dict[object, VectorSSAValue] = {}

        vec = ctx.new_vec("vec_xored")
        ops.append(Op("vpack", vec, xored_vals, "alu"))

        vec_map[xored_vals[0].id] = vec

        for op in chain0:
            vec_operands: list[VectorSSAValue] = []
            for operand in op.operands:
                if isinstance(operand, SSAValue):
                    if operand.id in vec_map:
                        vec_operands.append(vec_map[operand.id])
                    else:
                        vec_const = const_vec_cache.get(operand.id)
                        if vec_const is None:
                            vec_const = ctx.new_vec(
                                f"vb_{operand.name}" if operand.name else None
                            )
                            ops.append(Op("vbroadcast", vec_const, [operand], "valu"))
                            const_vec_cache[operand.id] = vec_const
                        vec_operands.append(vec_const)
                elif isinstance(operand, Const):
                    key = ("const", operand.value)
                    vec_const = const_vec_cache.get(key)
                    if vec_const is None:
                        vec_const = ctx.new_vec(f"vb_{operand.value}")
                        ops.append(Op("vbroadcast", vec_const, [operand], "valu"))
                        const_vec_cache[key] = vec_const
                    vec_operands.append(vec_const)
                else:
                    return None, None

            vec_opcode = VECTOR_OP_MAP.get(op.opcode)
            if vec_opcode is None:
                return None, None

            vec_res = ctx.new_vec(op.result.name if op.result else None)
            ops.append(Op(vec_opcode, vec_res, vec_operands, "valu"))
            if op.result is not None:
                vec_map[op.result.id] = vec_res

        last_res = chain0[-1].result
        if last_res is None or last_res.id not in vec_map:
            return None, None

        return ops, vec_map[last_res.id]

    def _vectorize_post_chain(
        self,
        chunks: list[list[Op]],
        h5_idx: list[int],
        vec_h5: VectorSSAValue,
        ctx: _VecContext,
    ) -> tuple[Optional[list[Op]], Optional[list[Op]], Optional[list[list[Op]]]]:
        idx_vals: list[SSAValue] = []
        mod_idx: list[int] = []
        fin_idx: list[int] = []
        for chunk, hi in zip(chunks, h5_idx):
            idx_op = self._find_result_op(chunk, "idx")
            mod_i = self._find_result_index_from(chunk, "mod_val", start=hi + 1)
            fin_i = self._find_result_index_from(chunk, "final_idx", start=hi + 1)
            if idx_op is None or mod_i is None or fin_i is None or fin_i <= mod_i:
                return None, None, None
            idx_vals.append(idx_op.result)
            mod_idx.append(mod_i)
            fin_idx.append(fin_i)

        if any(v is None for v in idx_vals):
            return None, None, None

        post_len = fin_idx[0] - mod_idx[0]
        for mi, fi in zip(mod_idx, fin_idx):
            if fi - mi != post_len:
                return None, None, None

        post0 = chunks[0][mod_idx[0] : fin_idx[0] + 1]
        if not post0:
            return None, None, None

        for op in post0:
            if op.engine != "alu" or op.opcode not in VECTOR_OP_MAP:
                return None, None, None
            if len(op.operands) != 2:
                return None, None, None

        for chunk, mi, fi in zip(chunks, mod_idx, fin_idx):
            post = chunk[mi : fi + 1]
            if len(post) != len(post0):
                return None, None, None
            for op, tmpl in zip(post, post0):
                if op.opcode != tmpl.opcode or op.engine != tmpl.engine:
                    return None, None, None

        for chunk, mi, fi in zip(chunks, mod_idx, fin_idx):
            suffix = chunk[fi + 1 :]
            used = self._collect_used_ids(suffix)
            for op in chunk[mi:fi]:
                if op.result and op.result.id in used:
                    return None, None, None

        vec_ops: list[Op] = []
        vec_idx = ctx.new_vec("vec_idx")
        vec_ops.append(Op("vpack", vec_idx, idx_vals, "alu"))

        vec_map: dict[int, VectorSSAValue] = {}
        vec_map[idx_vals[0].id] = vec_idx
        vec_map[chunks[0][h5_idx[0]].result.id] = vec_h5

        const_vec_cache: dict[object, VectorSSAValue] = {}

        for op in post0:
            vec_operands: list[VectorSSAValue] = []
            for operand in op.operands:
                if isinstance(operand, SSAValue):
                    if operand.id in vec_map:
                        vec_operands.append(vec_map[operand.id])
                    else:
                        vec_const = const_vec_cache.get(operand.id)
                        if vec_const is None:
                            vec_const = ctx.new_vec(
                                f"vb_{operand.name}" if operand.name else None
                            )
                            vec_ops.append(Op("vbroadcast", vec_const, [operand], "valu"))
                            const_vec_cache[operand.id] = vec_const
                        vec_operands.append(vec_const)
                elif isinstance(operand, Const):
                    key = ("const", operand.value)
                    vec_const = const_vec_cache.get(key)
                    if vec_const is None:
                        vec_const = ctx.new_vec(f"vb_{operand.value}")
                        vec_ops.append(Op("vbroadcast", vec_const, [operand], "valu"))
                        const_vec_cache[key] = vec_const
                    vec_operands.append(vec_const)
                else:
                    return None, None, None

            vec_opcode = VECTOR_OP_MAP.get(op.opcode)
            if vec_opcode is None:
                return None, None, None

            vec_res = ctx.new_vec(op.result.name if op.result else None)
            vec_ops.append(Op(vec_opcode, vec_res, vec_operands, "valu"))
            if op.result is not None:
                vec_map[op.result.id] = vec_res

        last_res = post0[-1].result
        if last_res is None or last_res.id not in vec_map:
            return None, None, None

        vec_final = vec_map[last_res.id]
        extract_final = [
            Op("vextract", chunks[i][fin_idx[i]].result, [vec_final, Const(i)], "alu")
            for i in range(VLEN)
        ]

        suffixes = [chunk[fi + 1 :] for chunk, fi in zip(chunks, fin_idx)]

        self._ops_vectorized += len(post0) * VLEN

        return vec_ops, extract_final, suffixes

    @staticmethod
    def _find_result_index(ops: list[Op], name: str) -> Optional[int]:
        for i, op in enumerate(ops):
            if op.result is not None and op.result.name == name:
                return i
        return None

    @staticmethod
    def _find_result_index_from(ops: list[Op], name: str, start: int) -> Optional[int]:
        for i in range(start, len(ops)):
            op = ops[i]
            if op.result is not None and op.result.name == name:
                return i
        return None

    @staticmethod
    def _find_result_op(ops: list[Op], name: str) -> Optional[Op]:
        for op in ops:
            if op.result is not None and op.result.name == name:
                return op
        return None

    @staticmethod
    def _collect_used_ids(ops: list[Op]) -> set[int]:
        used: set[int] = set()
        for op in ops:
            for operand in op.operands:
                if isinstance(operand, SSAValue):
                    used.add(operand.id)
        return used
