# Main file
# author: Zach Clare

# Main research imports
from ZCswap import SabreSwap
from helpers import dag_drawer
from qiskit.transpiler.passes import SabreLayout

# Required qiskit libraries and tools
from qiskit.transpiler.passes import Decompose, ApplyLayout
from qiskit.transpiler import CouplingMap, PassManager
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.library.standard_gates.swap import SwapGate
import numpy as np


def main():
    """
    Main function of the algorithm


    """




if __name__ == "__main__":
    main()