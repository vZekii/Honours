# Main file
# author: Zach Clare

# Main research imports
from ZCswap import SabreSwap
from helpers import dag_drawer, draw_circuit
from qiskit.transpiler.passes import SabreLayout
import ag

# Required qiskit libraries and tools
from qiskit.transpiler.passes import Decompose, ApplyLayout
from qiskit.transpiler import CouplingMap, PassManager
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.library.standard_gates.swap import SwapGate
import numpy as np
import os


def main() -> None:
    """
    Main function of the algorithm
    """

    # Setup the routing pass based on the architecture coupling graph
    AG_qiskit = CouplingMap(couplinglist=ag.qgrid(2, 3).edges())
    sabre_routing_pass = SabreSwap(coupling_map=AG_qiskit, heuristic="lookahead")
    sabre_layout_pass = SabreLayout(coupling_map=AG_qiskit, skip_routing=True)
    pm = PassManager([sabre_layout_pass, ApplyLayout()])

    # Select a benchmark/test circuit
    benchmark_path = "./benchmarks/6qbt/"
    bench_file = os.listdir(benchmark_path)[6]

    # Initialise the circuit and registers
    register = QuantumRegister(6, "q")
    circuit_in = QuantumCircuit(register)
    qasm_import = circuit_in.from_qasm_file(benchmark_path + bench_file)
    # qasm_import.draw(output="mpl", filename="qasm.png")
    circuit_in.compose(qasm_import, inplace=True)
    draw_circuit(circuit_in, "initial")

    initial_depth = circuit_in.depth()
    initial_dag = circuit_to_dag(circuit_in)
    initial_2q_gates = len(initial_dag.two_qubit_ops())
    initial_1q_gates = initial_dag.size() - initial_2q_gates

    # print(f"Initial depth: {initial_depth}\nInitial 2 qubit gates: {initial_2q_gates}")

    print(
        f"Running routing pass with initial depth: {initial_depth} and {initial_2q_gates} 2 qubit gates"
    )

    mapped_circuit = pm.run(circuit_in)
    inital_mapping = mapped_circuit.layout.final_layout

    print(f"Inititial mapping: {inital_mapping}")
    draw_circuit(mapped_circuit, "mapped")

    ouput_dag = sabre_routing_pass.run(dag=initial_dag)
    draw_circuit(dag_to_circuit(ouput_dag), "output")


if __name__ == "__main__":
    main()
