import os
from qiskit.visualization import dag_drawer
from qiskit import QuantumCircuit

graphviz_path = "C:/Users/Zekii/Documents/Scholarship/sabre_exp/Graphviz/bin/"


def draw_dag(dag, filename):
    """Helper function for drawing dags without error checking each time"""
    try:
        os.environ["PATH"] += os.pathsep + graphviz_path
        dag_drawer(dag, filename=filename)
    except Exception as e:
        print(e)
        print("Missing graphviz to draw DAG - Skipping...")


def draw_circuit(circuit: QuantumCircuit, filename: str) -> None:
    circuit.draw(output="mpl", filename="debug/" + filename + ".png")
