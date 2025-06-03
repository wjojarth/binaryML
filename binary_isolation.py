from sympy import symbols
from sympy.logic.boolalg import Or, And, Not, simplify_logic

# Define input variables x0 to x7
x = symbols('x0 x1 x2 x3 x4 x5 x6 x7')
sympy_vars = list(x)

# Compute truth table for 8-bit inputs
truth_table = {i: int(i**0.5) for i in range(256)}
output_exprs = {b: [] for b in range(4)}  # for bits y0 to y3

# Build minterms for each output bit
for val in range(256):
    in_bits = [(val >> (7 - i)) & 1 for i in range(8)]
    sqrt_val = truth_table[val]
    sqrt_bits = [(sqrt_val >> (3 - b)) & 1 for b in range(4)]

    literals = [
        sympy_vars[i] if bit else Not(sympy_vars[i])
        for i, bit in enumerate(in_bits)
    ]
    minterm = And(*literals)

    for b in range(4):
        if sqrt_bits[b]:
            output_exprs[b].append(minterm)

# Simplify output expressions
simplified_outputs = {}
for b in range(4):
    combined_expr = Or(*output_exprs[b])
    simplified_outputs[b] = simplify_logic(combined_expr, form='dnf')

# Count gates
def count_gates(expr):
    counts = {"AND": 0, "OR": 0, "NOT": 0}
    if expr is True or expr is False:
        return counts
    if isinstance(expr, Or):
        counts["OR"] += len(expr.args) - 1
        for arg in expr.args:
            sub_counts = count_gates(arg)
            for k in counts:
                counts[k] += sub_counts[k]
    elif isinstance(expr, And):
        counts["AND"] += len(expr.args) - 1
        for arg in expr.args:
            sub_counts = count_gates(arg)
            for k in counts:
                counts[k] += sub_counts[k]
    elif isinstance(expr, Not):
        counts["NOT"] += 1
        sub_counts = count_gates(expr.args[0])
        for k in counts:
            counts[k] += sub_counts[k]
    return counts

# Gather results
gate_summary = {}
total_gate_count = {"AND": 0, "OR": 0, "NOT": 0}
for bit, expr in simplified_outputs.items():
    gate_summary[bit] = count_gates(expr)
    for k in total_gate_count:
        total_gate_count[k] += gate_summary[bit][k]

# Format for display
total_gate_count["TOTAL"] = sum(total_gate_count.values())

# Evaluate example values
def evaluate_expr(expr, input_val):
    in_bits = [(input_val >> (7 - i)) & 1 for i in range(8)]
    env = {sympy_vars[i]: bool(bit) for i, bit in enumerate(in_bits)}
    result = expr.subs(env)
    return int(bool(result))

examples = [10, 20, 30, 40, 50, 60, 70]
evaluations = {}
for val in examples:
    bits = [evaluate_expr(simplified_outputs[b], val) for b in range(4)]
    predicted = int("".join(map(str, bits)), 2)
    evaluations[val] = {"bits": bits, "predicted": predicted, "actual": int(val**0.5)}

# Output all data
import pandas as pd
{
    "Gate Counts": total_gate_count,
    "Per Bit Gate Usage": gate_summary,
    "Simplified Logic": simplified_outputs,
    "Examples": pd.DataFrame(evaluations).T
}
