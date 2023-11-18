# File to benchmark the circuit against varying circuits
# author: Zach Clare

import csv
import os
import time
from typing import Optional

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap, PassManager, TransformationPass
from qiskit.transpiler.passes import ApplyLayout, SabreLayout
from rich import print

import ag
from SabreSwap import SabreSwap
from ZCswap_gaps import SabreSwap as ZCswap


def benchmark(
    routing_pass: TransformationPass,
    mappped_dag: DAGCircuit,
    best: Optional[dict],
    prev_zc_changes: Optional[int],
    prev_zc_activations: Optional[int],
) -> (dict, int):
    start_time = time.time()
    output = routing_pass.run(dag=mappped_dag)
    total_time = time.time() - start_time

    zc_changes, zc_activations = None, None

    if type(output) == tuple:
        print("got changes")
        output, zc_changes, zc_activations = output

    properties = output.properties()

    if not best:
        # for the first run
        best = properties

    if properties["depth"] < best["depth"]:
        best = properties

    elif properties["depth"] <= best["depth"]:
        try:
            if properties["operations"]["swap"] < best["operations"]["swap"]:
                best = properties
        except KeyError:
            pass
    else:
        zc_changes, zc_activations = prev_zc_changes, prev_zc_activations

    if zc_changes is not None and zc_activations is not None:
        return best, total_time, zc_changes, zc_activations
    else:
        return best, total_time, prev_zc_changes, prev_zc_activations


def main() -> None:
    # main function

    # setup some properties
    iterations = 5
    heuristic = "basic"  # "basic", "depth", etc

    # Setup architectures
    IBM_Q20 = CouplingMap(couplinglist=ag.q20().edges())
    Qgrid_20 = CouplingMap(couplinglist=ag.qgrid(4, 5).edges())

    # Set benchmark folder
    benchmark_path = "./benchmarks/qiskit_circuit_benchmark/"
    # benchmark_path = "./benchmarks/6qbt/"

    # Setup the benchmarking file
    fieldnames = [
        "Filename",
        "Architecture",
        "Initial Depth",
        "Sabre Depth",
        "Zach Depth",
        "Sabre Swaps",
        "Zach Swaps",
        "Zach Changes",
        "Zach Activations",
        "Sabre Depth reduction",
        "Zach Depth reduction",
        "Sabre Time taken",
        "Zach Time taken",
    ]
    with open("benchmarks.csv", mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    for file in os.listdir(benchmark_path):
        if any(x in file for x in ["12", "13", "14", "15", "16", "17", "18", "19", "20"]):
            continue
        print(f"Benchmarking {file}")
        # Load the circuit
        # Since we're testing specifically with 20qubit architecture
        circuit_in = QuantumCircuit(QuantumRegister(20, "q"))
        qasm_import = circuit_in.from_qasm_file(benchmark_path + file)
        circuit_in.compose(qasm_import, inplace=True)

        initial_properties = circuit_to_dag(circuit_in).properties()

        print(f"Initial info: {initial_properties}")

        for i, arch in enumerate([IBM_Q20, Qgrid_20]):
            best_zach, best_sabre = None, None
            zach_time, sabre_time = 0, 0
            zc_changes, zc_activations = 0, 0

            # iterate the benchmarks
            for iteration in range(iterations):
                # Layout the circuit
                sabre_layout_pass = SabreLayout(coupling_map=arch, skip_routing=True)
                pm = PassManager([sabre_layout_pass, ApplyLayout()])
                mapped_dag = circuit_to_dag(pm.run(circuit_in))

                # Setup the routers
                zachc_routing_pass = ZCswap(coupling_map=arch, heuristic=heuristic)
                sabre_routing_pass = SabreSwap(coupling_map=arch, heuristic=heuristic)

                # Run the benchmarks and record the best
                best_zach, ztime, zc_changes, zc_activations = benchmark(
                    zachc_routing_pass, mapped_dag, best_zach, zc_changes, zc_activations
                )
                best_sabre, stime, _, _ = benchmark(sabre_routing_pass, mapped_dag, best_sabre, None, None)

                # adjust the times
                zach_time += ztime
                sabre_time += stime

            print(f"Best Zach: {best_zach}\nBest Sabre: {best_sabre}")
            zach_time /= iterations
            sabre_time /= iterations

            print(f"Zach time: {zach_time:.2f}\nSabre time: {sabre_time:.2f}")

            zach_swaps, sabre_swaps = 0, 0

            if "swap" in best_sabre["operations"]:
                sabre_swaps = best_sabre["operations"]["swap"]

            if "swap" in best_zach["operations"]:
                zach_swaps = best_zach["operations"]["swap"]

            with open("benchmarks.csv", mode="a", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow(
                    {
                        "Filename": file,
                        "Architecture": ["IBM_Q20" if i == 0 else "Qgrid_20"],
                        "Initial Depth": initial_properties["depth"],
                        "Sabre Depth": best_sabre["depth"],
                        "Zach Depth": best_zach["depth"],
                        "Sabre Swaps": sabre_swaps,
                        "Zach Swaps": zach_swaps,
                        "Zach Changes": zc_changes,
                        "Zach Activations": zc_activations,
                        "Sabre Depth reduction": round(best_sabre["depth"] / initial_properties["depth"], 3),
                        "Zach Depth reduction": round(best_zach["depth"] / initial_properties["depth"], 3),
                        "Sabre Time taken": round(sabre_time, 3),
                        "Zach Time taken": round(zach_time, 3),
                    }
                )


if __name__ == "__main__":
    main()
