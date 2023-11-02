# File to benchmark the circuit against varying circuits
# author: Zach Clare

import os

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import ApplyLayout, SabreLayout, SabreSwap
from rich import print

import ag
from ZCswap_gaps import SabreSwap as ZCswap


def benchmark() -> None:
    # benchmarks the given circuit
    pass


def main() -> None:
    # main function

    # Setup architectures
    IBM_Q20 = CouplingMap(couplinglist=ag.q20().edges())
    Qgrid_20 = CouplingMap(couplinglist=ag.qgrid(4, 5).edges())

    # Set benchmark folder
    benchmark_path = "./benchmarks/qiskit_circuit_benchmark/"

    for file in os.listdir(benchmark_path):
        if "excitation" not in file:
            continue
        print(f"Benchmarking {file}")
        # Load the circuit
        circuit_in = QuantumCircuit(
            QuantumRegister(20, "q")
        )  # Since we're testing specifically with 20qubit architecture
        qasm_import = circuit_in.from_qasm_file(benchmark_path + file)
        circuit_in.compose(qasm_import, inplace=True)

        print(f"Initial info: {circuit_to_dag(circuit_in).properties()}")

        for arch in [IBM_Q20, Qgrid_20]:
            # Layout the circuit
            sabre_layout_pass = SabreLayout(coupling_map=arch, skip_routing=True)
            pm = PassManager([sabre_layout_pass, ApplyLayout()])
            mapped_dag = circuit_to_dag(pm.run(circuit_in))

            print(f"Mapped info: {mapped_dag.properties()}")

            # Setup the routers
            zachc_routing_pass = ZCswap(coupling_map=arch, heuristic="lookahead")
            sabre_routing_pass = SabreSwap(coupling_map=arch, heuristic="lookahead")

            # Run the routers
            zachc_output = zachc_routing_pass.run(dag=mapped_dag)
            sabre_output = sabre_routing_pass.run(dag=mapped_dag)

            # Display outcome
            print(f"Zach out: {zachc_output.properties()}")
            print(f"Sabre out: {sabre_output.properties()}")


if __name__ == "__main__":
    main()
