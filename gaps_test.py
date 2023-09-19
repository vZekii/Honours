from helpers import draw_dag, draw_circuit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler.layout import Layout

from collections import defaultdict

from rich import print  # fancier colouring
from rich.panel import Panel

from enum import Enum

EXTENDED_SET_SIZE = (
    20  # Size of lookahead window. TODO: set dynamically to len(current_layout)
)


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


def _obtain_extended_set(dag, front_layer, pred):
    """Populate extended_set by looking ahead a fixed number of gates.
    For each existing element add a successor until reaching limit.
    """
    extended_set = []
    decremented = []
    tmp_front_layer = front_layer
    done = False
    while tmp_front_layer and not done:
        new_tmp_front_layer = []
        for node in tmp_front_layer:
            for successor in _successors(node, dag):
                decremented.append(successor)
                pred[successor] -= 1
                if pred[successor] == 0:
                    new_tmp_front_layer.append(successor)
                    if len(successor.qargs) == 2:
                        extended_set.append(successor)
            if len(extended_set) >= EXTENDED_SET_SIZE:
                done = True
                break
        tmp_front_layer = new_tmp_front_layer
    for node in decremented:
        pred[node] += 1
    return extended_set


def apply_gate_commutative(gate, dag, front_layer, pred):
    """
    The bread and butter of the new algorithm. Will attempt to apply gates with commutativity, otherwise build a storage list of up to one gate.
    """

    # We can use the logical qubits for now, but otherwise
    # we would need to swap because of the layout
    v0, v1 = (
        current_layout._v2p[gate.qargs[0]],
        current_layout._v2p[gate.qargs[1]],
    )

    print(
        Panel(
            f"Attempting application of {gate.name} gate on qubits {v0} and {v1}",
            highlight=True,
        )
    )
    print(gap_storage)

    for i in range(min(v0, v1) + 1, max(v0, v1)):
        print(f"{i} is free")
        gap_storage[i] = Gap.FREE

    # * The "and" is required here to ensure that the 2 gates are not applied on the same qubits, as there would be no option there.
    if gap_storage[v0] == Gap.CONTROL and gap_storage[v1] != Gap.TARGET:
        print(f"Found potential control match on qubit {v0}")

    elif gap_storage[v1] == Gap.TARGET and gap_storage[v0] != Gap.CONTROL:
        print(f"Found potential target match on qubit {v0}")

    else:
        gate_storage[v0], gate_storage[v1] = gate, gate

    gap_storage[v0], gap_storage[v1] = Gap.CONTROL, Gap.TARGET

    print(_obtain_extended_set(dag, front_layer, pred))

    print(gate_storage)


# Setup circuit
circuit_in = QuantumCircuit(QuantumRegister(6, "q"))
qasm_import = circuit_in.from_qasm_file("./test.qasm")
circuit_in.compose(qasm_import, inplace=True)
dag_in = circuit_to_dag(circuit_in)
# draw_circuit(circuit_in, "testing")
draw_dag(dag_in, filename="testdag.png")  # No graphviz on my laptop
print(dag_in.properties())
pred = _build_required_predecessors(dag_in)

canonical_register = dag_in.qregs["q"]
current_layout = Layout.generate_trivial_layout(canonical_register)
print(current_layout)

# Preserve input DAG's name, regs, wire_map, etc. but replace the graph.
mapped_dag = dag_in.copy_empty_like()


physical_qubits = [0, 1, 2, 3, 4, 5]
gap_storage = {
    qubit: Gap.FREE for qubit in physical_qubits
}  # Dictionary to store gap info
gate_storage = {idx: [] for idx in physical_qubits}  # dictionary to store 2 qubit gates


# Execution
front_layer = dag_in.front_layer()
num_search_steps = 0

# * this is loosely copied from SABRE
while front_layer:
    execute_list = []

    for gate in front_layer:
        if len(gate.qargs) == 2:
            # coupling check has been left out here
            execute_list.append(gate)

        else:
            execute_list.append(gate)  # fallback for single qubit gates

    front_layer = []
    if execute_list:
        for gate in execute_list:
            apply_gate_commutative(gate, dag_in, front_layer, pred)

            for successor in _successors(gate, dag_in):
                pred[successor] -= 1
                if pred[successor] == 0:
                    front_layer.append(successor)

        ops_since_progress = []
        extended_set = None
        continue

    num_search_steps += 1

    print("Got to the end")
