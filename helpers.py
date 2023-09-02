import os
from qiskit.visualization import dag_drawer
from qiskit import QuantumCircuit


def draw_dag(*args, **kwargs):
    """Helper function for drawing dags without error checking each time"""
    try:
        os.environ["PATH"] += (
            os.pathsep + "C:/Users/Zekii/Documents/Scholarship/sabre_exp/Graphviz/bin/"
        )
        dag_drawer(*args, **kwargs)
    except:
        print("Missing graphviz to draw DAG - Skipping...")


def draw_circuit(circuit: QuantumCircuit, filename: str) -> None:
    circuit.draw(output="mpl", filename="debug/" + filename + ".png")
