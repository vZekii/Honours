from helpers import draw_dag, draw_circuit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler.layout import Layout

from collections import defaultdict

from pprint import pprint

from enum import Enum


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


# Dummy putting gate here as it will be used to handle single gates later on
Gap = Enum("Gap", "GATE TARGET CONTROL FREE")


# Setup circuit
circuit_in = QuantumCircuit(QuantumRegister(6, "q"))
qasm_import = circuit_in.from_qasm_file("./test.qasm")
circuit_in.compose(qasm_import, inplace=True)
dag_in = circuit_to_dag(circuit_in)
# draw_circuit(circuit_in, "testing")
# draw_dag(dag_in, filename="testdag.png")  No graphviz on my laptop
print(dag_in.properties())

# Test code
pred = _build_required_predecessors(dag_in)
pprint(pred)

canonical_register = dag_in.qregs["q"]
current_layout = Layout.generate_trivial_layout(canonical_register)
print(current_layout)

physical_qubits = [0, 1, 2, 3, 4, 5]
storage = {qubit: None for qubit in physical_qubits}


# Execution
front_layer = dag_in.front_layer()

# * this is loosely copied from SABRE
while front_layer:
    execute_list = []

    for gate in front_layer:
        if len(gate.qargs) == 2:
            # We can use the logical qubits for now, but otherwise
            # we would need to swap because of the layout
            v0, v1 = (
                current_layout._v2p[gate.qargs[0]],
                current_layout._v2p[gate.qargs[1]],
            )
            print(f"Executed {gate.name} gate on qubits {v0} and {v1}")

            for i in range(min(v0, v1) + 1, max(v0, v1)):
                print(f"{i} is free")
                storage[i] = Gap.FREE

        execute_list.append(gate)

    front_layer = []

    for gate in execute_list:
        for successor in _successors(gate, dag_in):
            pred[successor] -= 1
            if pred[successor] == 0:
                front_layer.append(successor)
