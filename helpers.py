import os
from qiskit.visualization import dag_drawer

def draw_dag(*args, **kwargs):
    """ Helper function for drawing dags without error checking each time"""
    try:
        os.environ["PATH"] += os.pathsep + "C:/Users/Zekii/Documents/Scholarship/sabre_exp/Graphviz/bin/"
        dag_drawer(*args, **kwargs)
    except:
        print("Missing graphviz to draw DAG - Skipping...")