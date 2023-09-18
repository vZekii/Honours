from helpers import draw_dag, draw_circuit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler.layout import Layout

from collections import defaultdict

from pprint import pprint
from rich import print  # fancier colouring

from enum import Enum


# Dummy putting gate here as it will be used to handle single gates later on
Gap = Enum("Gap", "GATE TARGET CONTROL FREE")


def _successors(node, dag):
    for _, successor, _ in dag.edges(node):
        if isinstance(successor, DAGOpNode):
            yield successor

            # Need to print the repr version since the superclass outputs a unique id on the str representation :skull:
            # print(successor.__repr__())


def _build_required_predecessors(dag):
    out = defaultdict(int)
    # We don't need to count in- or out-wires: outs can never be predecessors, and all input
    # wires are automatically satisfied at the start.
    for node in dag.op_nodes():
        for successor in _successors(node, dag):
            out[successor] += 1
    return out


def apply_gate_commutative(gate):
    """
    The bread and butter of the new algorithm. Will attempt to apply gates with commutativity, otherwise build a storage list of up to one gate.
    """

    # We can use the logical qubits for now, but otherwise
    # we would need to swap because of the layout
    v0, v1 = (
        current_layout._v2p[gate.qargs[0]],
        current_layout._v2p[gate.qargs[1]],
    )

    print(f"Attempting application of {gate.name} gate on qubits {v0} and {v1}")
    print(gap_storage)

    for i in range(min(v0, v1) + 1, max(v0, v1)):
        print(f"{i} is free")
        gap_storage[i] = Gap.FREE

    # * The "and" is required here to ensure that the 2 gates are not applied on the same qubits, as there would be no option there.
    if gap_storage[v0] == Gap.CONTROL and gap_storage[v1] != Gap.TARGET:
        print(f"Found potential control match on qubit {v0}")

    if gap_storage[v1] == Gap.TARGET and gap_storage[v0] != Gap.CONTROL:
        print(f"Found potential target match on qubit {v0}")

    gap_storage[v0], gap_storage[v1] = Gap.CONTROL, Gap.TARGET


# Setup circuit
circuit_in = QuantumCircuit(QuantumRegister(6, "q"))
qasm_import = circuit_in.from_qasm_file("./test.qasm")
circuit_in.compose(qasm_import, inplace=True)
dag_in = circuit_to_dag(circuit_in)
# draw_circuit(circuit_in, "testing")
draw_dag(dag_in, filename="testdag.png")  # No graphviz on my laptop
print(dag_in.properties())

# Test code
pred = _build_required_predecessors(dag_in)
pprint(pred)

canonical_register = dag_in.qregs["q"]
current_layout = Layout.generate_trivial_layout(canonical_register)
print(current_layout)

physical_qubits = [0, 1, 2, 3, 4, 5]
gap_storage = {
    qubit: Gap.FREE for qubit in physical_qubits
}  # Dictionary to store gap info
gate_storage = {idx: [] for idx in physical_qubits}  # dictionary to store 2 qubit gates


# Execution
front_layer = dag_in.front_layer()

# * this is loosely copied from SABRE
while front_layer:
    execute_list = []

    for gate in front_layer:
        if len(gate.qargs) == 2:
            execute_list.append(gate)

    front_layer = []
    if execute_list:
        for gate in execute_list:
            apply_gate_commutative(gate)

            for successor in _successors(gate, dag_in):
                pred[successor] -= 1
                if pred[successor] == 0:
                    front_layer.append(successor)

        ops_since_progress = []
        extended_set = None
        continue

    print("Got to the end")
