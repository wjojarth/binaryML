import itertools
from collections import defaultdict

# Define logic gate classes
class NOTGate:
    def __init__(self, input_signal):
        self.input = input_signal
        self.output = f"NOT_{input_signal}"

class ANDGate:
    def __init__(self, input_signals):
        self.inputs = input_signals
        self.output = f"AND_{'_'.join(input_signals)}"

class ORGate:
    def __init__(self, input_signals):
        self.inputs = input_signals
        self.output = f"OR_{'_'.join(input_signals)}"

# Build sum-of-products network with gate reuse
def build_sop_circuit():
    truth_table = {i: int(i**0.5) for i in range(256)}
    inputs = [f"x{i}" for i in range(8)]
    gates = []
    not_gates = {}
    and_gates = {}
    output_bits = defaultdict(list)

    for value, sqrt_val in truth_table.items():
        bits = [(value >> (7 - i)) & 1 for i in range(8)]
        sqrt_bits = [(sqrt_val >> (3 - b)) & 1 for b in range(4)]

        minterm_inputs = []
        for i, bit_val in enumerate(bits):
            if bit_val == 1:
                minterm_inputs.append(inputs[i])
            else:
                if inputs[i] not in not_gates:
                    not_gate = NOTGate(inputs[i])
                    not_gates[inputs[i]] = not_gate
                    gates.append(not_gate)
                minterm_inputs.append(not_gates[inputs[i]].output)

        and_key = tuple(sorted(minterm_inputs))
        if and_key not in and_gates:
            and_gate = ANDGate(minterm_inputs)
            and_gates[and_key] = and_gate
            gates.append(and_gate)
        and_output = and_gates[and_key].output

        for b in range(4):
            if sqrt_bits[b] == 1:
                output_bits[b].append(and_output)

    final_or_gates = {}
    for b in range(4):
        if output_bits[b]:
            or_gate = ORGate(output_bits[b])
            gates.append(or_gate)
            final_or_gates[b] = or_gate.output
        else:
            final_or_gates[b] = "0"

    return inputs, gates, final_or_gates

# Simulate the logic circuit
def simulate(inputs_signals, gates, final_outputs, input_vector):
    signal_values = {}
    for i, bit in enumerate(inputs_signals):
        signal_values[bit] = input_vector[i]

    for gate in gates:
        if isinstance(gate, NOTGate):
            signal_values[gate.output] = 1 - signal_values[gate.input]
        elif isinstance(gate, ANDGate):
            vals = [signal_values[i] for i in gate.inputs]
            signal_values[gate.output] = int(all(vals))
        elif isinstance(gate, ORGate):
            vals = [signal_values[i] for i in gate.inputs]
            signal_values[gate.output] = int(any(vals))

    return [signal_values[final_outputs[b]] for b in range(4)]

# Build the circuit
inputs, gates, outputs = build_sop_circuit()

# Print gate summary
print(f"Total NOT gates: {len([g for g in gates if isinstance(g, NOTGate)])}")
print(f"Total AND gates: {len([g for g in gates if isinstance(g, ANDGate)])}")
print(f"Total OR gates: {len([g for g in gates if isinstance(g, ORGate)])}")

# Print 10 example evaluations
print("\nExample evaluations:")
for val in range(10):
    in_bits = [(val >> (7 - i)) & 1 for i in range(8)]
    out_bits = simulate(inputs, gates, outputs, in_bits)
    predicted = int("".join(str(x) for x in out_bits), 2)
    actual = int(val**0.5)
    print(f"√{val:3} = predicted: {predicted:2}  |  actual: {actual:2}  | bits: {out_bits}")

# Check correctness
errors = 0
for val in range(256):
    in_bits = [(val >> (7 - i)) & 1 for i in range(8)]
    out_bits = simulate(inputs, gates, outputs, in_bits)
    expected = [(int(val**0.5) >> (3 - b)) & 1 for b in range(4)]
    if out_bits != expected:
        errors += 1

print(f"\nVerification errors: {errors} out of 256")
