# Main file
# author: Zach Clare

import os

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.standard_gates.swap import SwapGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import CouplingMap, PassManager

# Required qiskit libraries and tools
from qiskit.transpiler.passes import (
    ApplyLayout,
    CommutationAnalysis,
    CommutativeCancellation,
    Decompose,
    SabreLayout,
)

import ag
from helpers import draw_circuit, draw_dag

# Main research imports
# from ZCswap import SabreSwap
from ZCswap_gaps import SabreSwap


def main() -> None:
    """
    Main function of the algorithm
    """

    # Setup the routing pass based on the architecture coupling graph
    AG_qiskit = CouplingMap(couplinglist=ag.qgrid(3, 2).edges())
    sabre_routing_pass = SabreSwap(coupling_map=AG_qiskit, heuristic="lookahead")
    sabre_layout_pass = SabreLayout(coupling_map=AG_qiskit, skip_routing=True)
    pm = PassManager([sabre_layout_pass, ApplyLayout()])

    # Select a benchmark/test circuit
    benchmark_path = "./benchmarks/custom/"
    # bench_file = os.listdir(benchmark_path)[6]
    # bench_file = "custom2.qasm"
    bench_file = "custom3.qasm"

    # Initialise the circuit and registers
    register = QuantumRegister(AG_qiskit.size(), "q")
    circuit_in = QuantumCircuit(register)
    qasm_import = circuit_in.from_qasm_file(benchmark_path + bench_file)
    # qasm_import.draw(output="mpl", filename="qasm.png")
    circuit_in.compose(qasm_import, inplace=True)
    draw_circuit(circuit_in, "initial")

    # get some initial info
    initial_depth = circuit_in.depth()
    initial_dag = circuit_to_dag(circuit_in)
    draw_dag(initial_dag, filename=f"debug/dag_in.png")
    print("Runs:", initial_dag.collect_runs(["cx"]))
    initial_2q_gates = len(initial_dag.two_qubit_ops())
    initial_1q_gates = initial_dag.size() - initial_2q_gates

    # print(f"Initial depth: {initial_depth}\nInitial 2 qubit gates: {initial_2q_gates}")

    print(f"Running routing pass with initial depth: {initial_depth} and {initial_2q_gates} 2 qubit gates")

    # Run the Sabre layout and get the initial mapping
    mapped_circuit = pm.run(circuit_in)
    mapped_dag = circuit_to_dag(mapped_circuit)
    print(mapped_dag.properties())

    # Commutation cancelation attempt
    # cancelled = CommutativeCancellation()(mapped_circuit)
    # draw_circuit(cancelled, "commutation_cancelling")

    # property_set = {}
    # CommutationAnalysis()(mapped_circuit, property_set)
    # Collect resuls of the commutation pass
    # with open("commutation_pass_output.txt", "w") as f:
    #     print(property_set["commutation_set"], file=f)

    inital_mapping = mapped_circuit.layout.initial_layout
    print(f"Inititial mapping: {inital_mapping}")
    draw_circuit(mapped_circuit, "mapped")

    # Run the router and display the outcomes
    ouput_dag = sabre_routing_pass.run(dag=mapped_dag)
    draw_circuit(dag_to_circuit(ouput_dag), "output")
    draw_dag(ouput_dag, filename=f"debug/dag_out.png")

    print(ouput_dag.properties())
    output_depth = ouput_dag.properties()["depth"]
    print(f"Depth change: {output_depth - initial_depth} ({(output_depth / initial_depth)*100:.2f}%)")


if __name__ == "__main__":
    main()
