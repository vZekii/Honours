# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Routing via SWAP insertion using the SABRE method from Li et al."""

import logging
from collections import defaultdict
from copy import copy, deepcopy

import numpy as np
import retworkx

from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGOpNode

# zc ---------
from qiskit.dagcircuit.dagcircuit import DAGCircuit  # better for debugging
from qiskit import QuantumRegister
from rich import print
from rich.panel import Panel
from helpers import draw_circuit
from qiskit.converters import dag_to_circuit
from typing import Union, Optional

# zc ---------

logger = logging.getLogger(__name__)

EXTENDED_SET_SIZE = 20  # Size of lookahead window. TODO: set dynamically to len(current_layout)
EXTENDED_SET_WEIGHT = 0.5  # Weight of lookahead window compared to front_layer.

DECAY_RATE = 0.001  # Decay coefficient for penalizing serial swaps.
DECAY_RESET_INTERVAL = 5  # How often to reset all decay rates to 1.

from enum import Enum

# Dummy putting gate here as it will be used to handle single gates later on
Gap = Enum("Gap", "GATE TARGET CONTROL FREE SWAP RZ RX U")


def get_qubits_from_layout(node: DAGOpNode, current_layout: Layout) -> Union[int, tuple[int, int]]:
    """Get the associated qubits that the gate is being applied on"""
    if len(node.qargs) == 2:
        return (current_layout._v2p[node.qargs[0]], current_layout._v2p[node.qargs[1]])

    # ! Fix this to return a type that is the same as above, probably switch to list
    return current_layout._v2p[node.qargs[0]]


def apply_on_dag(dag: DAGCircuit, gate: DAGOpNode) -> None:
    """Apply the gate on the dag"""
    dag.apply_operation_back(gate.op, gate.qargs, gate.cargs)


class GateStorage:
    storage: dict[int, dict[str, Union[DAGOpNode, bool]]]

    def __init__(self, qubits: int) -> None:
        self.storage = {idx: {"gate": None, "applied": False} for idx in qubits}

    def add_gate(self, gate: DAGOpNode, layout: Layout, applied=False) -> None:
        """Add a new gate to the storage, as applied or not"""
        if len(gate.qargs) == 1:
            self.storage[get_qubits_from_layout(gate, layout)] = {"gate": gate, "applied": applied}
            return

        for qubit in get_qubits_from_layout(gate, layout):
            self.storage[qubit] = {"gate": gate, "applied": applied}

    def get_gate(self, qubit: int):
        """Fetch the gate thats on a certain qubit"""
        gate_info = self.storage.get(qubit, None)
        if gate_info:
            return gate_info["gate"]
        return None

    def mark_applied(self, gate: DAGOpNode, layout: Layout):
        """Mark the input gate as applied on each qubit"""

        # ! genuinely so gross
        if len(gate.qargs) == 1:
            qubit = get_qubits_from_layout(gate, layout)
            if qubit in self.storage:
                self.storage[qubit]["applied"] = True

            return

        for qubit in get_qubits_from_layout(gate, layout):
            if qubit in self.storage:
                self.storage[qubit]["applied"] = True

    def is_applied(self, qubit: int) -> bool:
        """Check if a gate is applied on a qubit"""
        return self.storage[qubit]["applied"] == True

    def swap_qubits(self, qubit_1: int, qubit_2: int) -> None:
        """Swap the storage in 2 qubits, for use after a swap gate"""
        self.storage[qubit_1], self.storage[qubit_2] = (
            self.storage[qubit_2],
            self.storage[qubit_1],
        )

    def __str__(self):
        return self.storage.__str__()


class SabreSwap(TransformationPass):
    node_buffer: Optional[DAGOpNode]
    gap_storage: dict[int, Gap]
    gate_storage: GateStorage

    def __init__(self, coupling_map, heuristic="basic", seed=None, fake_run=False):
        super().__init__()

        # Assume bidirectional couplings, fixing gate direction is easy later.
        if coupling_map is None or coupling_map.is_symmetric:
            self.coupling_map = coupling_map
        else:
            self.coupling_map = deepcopy(coupling_map)
            self.coupling_map.make_symmetric()

        self.heuristic = heuristic
        self.seed = seed
        self.fake_run = fake_run
        self.required_predecessors = None
        self.qubits_decay = None
        self._bit_indices = None
        self.dist_matrix = None
        # zc -------
        self.node_buffer = None
        self.gap_storage = {}
        self.gate_storage = {}
        self.iteration = 1
        # zc -------

    def insert_free_gaps(self, qarg1: int, qarg2: int) -> None:
        """Fill any gaps between 2 qubit nodes with free space"""
        for i in range(min(qarg1, qarg2) + 1, max(qarg1, qarg2)):
            print(f"{i} is free")
            self.gap_storage[i] = Gap.FREE

    def zc_handle_cnot(self, mapped_dag: DAGCircuit, new_node: DAGOpNode, current_layout: Layout):
        """Apply a CNOT gate to the circuit, attempting to apply any commutativity rules"""

        gate_control, gate_target = get_qubits_from_layout(new_node, current_layout)

        # we can make any qubits inbetween the gate a gap.
        self.insert_free_gaps(gate_control, gate_target)

        print(
            Panel(
                f"{self.iteration}: Attempting application of {new_node.name} gate on control {gate_control} and target {gate_target}",
                highlight=True,
            )
        )

        # * Begin main logic here

        # Check if the new gate's control lies on the same line as a previous gates control, and their targets don't match - and the gate hasn't been applied yet
        # Implements Inter-CNOT Rule A
        if (
            self.gap_storage[gate_control] == Gap.CONTROL
            and self.gap_storage[gate_target] != Gap.TARGET
            and not self.gate_storage.is_applied(gate_control)
        ):
            print(f"Found potential control match on qubit {gate_control}")

            prior = self.gate_storage.get_gate(gate_control)
            prior_control, prior_target = get_qubits_from_layout(prior, current_layout)
            print(f"prior gate on qubits {prior_control}, {prior_target}")

            if self.apply_before(mapped_dag, prior, new_node):
                print("we should apply the new node before")
                # mapped_dag.apply_operation_back(new_node.op, new_node.qargs, new_node.cargs)
                apply_on_dag(mapped_dag, new_node)
                self.gap_storage[prior_control] = Gap.CONTROL
                self.gap_storage[prior_target] = Gap.TARGET

                # since we applied the prior gate second, the storage doesn't have to update
                return

            print("we should do it the same way")
            # Apply the prior gate and save the new one
            # mapped_dag.apply_operation_back(prior.op, prior.qargs, prior.cargs)
            apply_on_dag(mapped_dag, prior)

            self.gap_storage[gate_control] = Gap.CONTROL
            self.gap_storage[gate_target] = Gap.TARGET

            # update prior in storage
            self.gate_storage.mark_applied(prior, current_layout)

            # and put the new gate in
            self.gate_storage.add_gate(new_node, current_layout)

            # the new node is now also the last applied
            self.node_buffer = new_node

            return

        # Check if the new gate's target lies on the same line as a previous gates target, and their controls don't match - and the gate hasn't been applied yet
        # Implements Inter-CNOT Rule B
        if (
            self.gap_storage[gate_target] == Gap.TARGET
            and self.gap_storage[gate_control] != Gap.CONTROL
            and not self.gate_storage.is_applied(gate_target)
        ):
            print(f"Found potential target match on qubit {gate_target}")

            prior = self.gate_storage.get_gate(gate_target)
            prior_control, prior_target = get_qubits_from_layout(prior, current_layout)
            print(f"prior gate on qubits {prior_control}, {prior_target}")

            if self.apply_before(mapped_dag, prior, new_node):
                print("we should apply the new node before")
                # mapped_dag.apply_operation_back(new_node.op, new_node.qargs, new_node.cargs)
                apply_on_dag(mapped_dag, new_node)
                # self.gap_storage[gate_control] = Gap.FREE
                self.gap_storage[prior_control] = Gap.CONTROL
                self.gap_storage[prior_target] = Gap.TARGET

                # since we applied the prior gate second, the storage doesn't have to update
                # * If we decide to remove the gap storage, we'll need to add the new gate applied to the storage
                return

            print("we should do it the same way")
            # Apply the prior gate and save the new one
            # mapped_dag.apply_operation_back(prior.op, prior.qargs, prior.cargs)
            apply_on_dag(mapped_dag, prior)

            # self.gap_storage[prior_target] = Gap.FREE
            self.gap_storage[gate_control] = Gap.CONTROL
            self.gap_storage[gate_target] = Gap.TARGET

            # update prior in storage
            self.gate_storage.mark_applied(prior, current_layout)

            # and put the new gate in
            self.gate_storage.add_gate(new_node, current_layout)

            # the new node is now also the last applied
            self.node_buffer = new_node

            return

            # we need to test both applications, both forward and back

        # if the gate doesnt match but applies on a line with gates stored, we need to apply the prior gate if possible, otherwise just add the gate into the buffer
        prior = None
        if self.gate_storage.get_gate(gate_control) is not None and prior is None:
            # Prior gate on the control
            if not self.gate_storage.is_applied(gate_control):
                print("prior on control wasn't applied")
                prior = self.gate_storage.get_gate(gate_control)

        if self.gate_storage.get_gate(gate_target) is not None and prior is None:
            # Prior gate on the target
            if not self.gate_storage.is_applied(gate_target):
                print("prior on target wasn't applied")
                prior = self.gate_storage.get_gate(gate_target)

        # Handles the case where there isn't an applicable gate, and applies the current gate in the buffer (or last gate)
        if isinstance(self.node_buffer, DAGOpNode) and prior is None:
            print("Didn't find a prior, lets pull from storage")
            prior = self.node_buffer

        if prior:
            print("found a prior and applying it")
            # mapped_dag.apply_operation_back(prior.op, prior.qargs, prior.cargs)
            apply_on_dag(mapped_dag, prior)
            # update prior in storage
            self.gate_storage.mark_applied(prior, current_layout)

        # This code below is only if it's the first gate to be seen - or if the buffer is cleared for whatever reason
        self.gate_storage.add_gate(new_node, current_layout)

        # the new node is now also the last applied
        self.node_buffer = new_node

        self.gap_storage[gate_control], self.gap_storage[gate_target] = (
            Gap.CONTROL,
            Gap.TARGET,
        )

    def zc_handle_swap(self, mapped_dag: DAGCircuit, swap_node: DAGOpNode, current_layout: Layout):
        # ! quick hack to quick apply the last gate in storage
        if self.node_buffer:
            apply_on_dag(mapped_dag, self.node_buffer)

            # TODO this will need to be ammended to allow for single qubit gates as well
            if len(self.node_buffer.qargs) == 2:
                # It's a cnot gate for now but will need to adjust
                buffer_control, buffer_target = get_qubits_from_layout(self.node_buffer, current_layout)
                self.gap_storage[buffer_control], self.gap_storage[buffer_target] = (
                    Gap.CONTROL,
                    Gap.TARGET,
                )
            else:
                buffer_qubit = get_qubits_from_layout(self.node_buffer, current_layout)
                # ! super duper ultra gross
                if self.gate_storage.get_gate(buffer_qubit).op.name == "rz":
                    self.gap_storage[buffer_qubit] = Gap.RZ
                else:
                    self.gap_storage[buffer_qubit] = Gap.RX

            # Update in storage
            self.gate_storage.mark_applied(self.node_buffer, current_layout)

            self.node_buffer = None

        # ! end of quick hack

        targ1, targ2 = get_qubits_from_layout(swap_node, current_layout)
        # code to transform each gate that lies on the swap line
        # for wire in (targ1, targ2):
        #     if self.gate_storage[wire]:
        #         storage_node = self.gate_storage[wire]
        #         self.gate_storage[wire] = _transform_gate_for_layout(
        #             storage_node, current_layout, canonical_register
        #         )

        print(Panel(f"{self.iteration}: Applying SWAP gate on qubits {targ1} and {targ2}"))

        # modify the gaps to include swap
        self.gap_storage[targ1], self.gap_storage[targ2] = Gap.SWAP, Gap.SWAP
        self.gate_storage.add_gate(swap_node, current_layout, applied=True)

        # TODO apply swap commutation rules
        apply_on_dag(mapped_dag, swap_node)

    def zc_handle_rz(self, mapped_dag: DAGCircuit, rz_node: DAGOpNode, current_layout: Layout):
        # Need to handle the situation where the rz gate comes after the CNOT gate

        qubit = get_qubits_from_layout(rz_node, current_layout)

        # Handle the on control situation
        if self.gap_storage[qubit] == Gap.CONTROL and not self.gate_storage.is_applied(qubit):
            prior = self.gate_storage.get_gate(qubit)
            if self.apply_before(mapped_dag, prior, rz_node):
                # apply the rz node first
                apply_on_dag(mapped_dag, rz_node)

                # We know it's a CNOT gate due to the first part of the if statement
                prior_control, prior_target = get_qubits_from_layout(prior, current_layout)
                self.gap_storage[prior_control] = Gap.CONTROL
                self.gap_storage[prior_target] = Gap.TARGET
                return

            # Apply the rz node after
            apply_on_dag(mapped_dag, prior)
            self.gap_storage[qubit] = Gap.RZ

            # update prior in storage
            self.gate_storage.mark_applied(prior, current_layout)

            # and put the new gate in
            self.gate_storage.add_gate(rz_node, current_layout)

            # the new node is now also the last applied
            self.node_buffer = rz_node
            return

        # Apply the node otherwise
        # TODO make this a global function
        prior = None
        if self.gate_storage.get_gate(qubit) is not None and prior is None:
            # Prior gate on the target
            if not self.gate_storage.is_applied(qubit):
                print("prior on target wasn't applied")
                prior = self.gate_storage.get_gate(qubit)

        # Handles the case where there isn't an applicable gate, and applies the current gate in the buffer (or last gate)
        if isinstance(self.node_buffer, DAGOpNode) and prior is None:
            print("Didn't find a prior, lets pull from storage")
            prior = self.node_buffer

        if prior:
            print("found a prior and applying it")
            # mapped_dag.apply_operation_back(prior.op, prior.qargs, prior.cargs)
            apply_on_dag(mapped_dag, prior)
            # update prior in storage
            self.gate_storage.mark_applied(prior, current_layout)

        # This code below is only if it's the first gate to be seen - or if the buffer is cleared for whatever reason
        self.gate_storage.add_gate(rz_node, current_layout)

        # the new node is now also the last applied
        self.node_buffer = rz_node
        self.gap_storage[qubit] = Gap.RZ

    def zc_handle_rx(self, mapped_dag: DAGCircuit, rx_node: DAGOpNode, current_layout: Layout):
        # Need to handle the situation where the rz gate comes after the CNOT gate

        qubit = get_qubits_from_layout(rx_node, current_layout)

        if self.gap_storage[qubit] == Gap.TARGET and not self.gate_storage.is_applied(qubit):
            prior = self.gate_storage.get_gate(qubit)
            if self.apply_before(mapped_dag, prior, rx_node):
                # apply the rx node first
                apply_on_dag(mapped_dag, rx_node)

                # We know it's a CNOT gate due to the first part of the if statement
                prior_control, prior_target = get_qubits_from_layout(prior, current_layout)
                self.gap_storage[prior_control] = Gap.CONTROL
                self.gap_storage[prior_target] = Gap.TARGET
                return

            # Apply the rx node after
            apply_on_dag(mapped_dag, prior)
            self.gap_storage[qubit] = Gap.RX

            # update prior in storage
            self.gate_storage.mark_applied(prior, current_layout)

            # and put the new gate in
            self.gate_storage.add_gate(rx_node, current_layout)

            # the new node is now also the last applied
            self.node_buffer = rx_node
            return

        # Apply the node otherwise
        # TODO make this a global function
        prior = None
        if self.gate_storage.get_gate(qubit) is not None and prior is None:
            # Prior gate on the target
            if not self.gate_storage.is_applied(qubit):
                print("prior on target wasn't applied")
                prior = self.gate_storage.get_gate(qubit)

        # Handles the case where there isn't an applicable gate, and applies the current gate in the buffer (or last gate)
        if isinstance(self.node_buffer, DAGOpNode) and prior is None:
            print("Didn't find a prior, lets pull from storage")
            prior = self.node_buffer

        if prior:
            print("found a prior and applying it")
            # mapped_dag.apply_operation_back(prior.op, prior.qargs, prior.cargs)
            apply_on_dag(mapped_dag, prior)
            # update prior in storage
            self.gate_storage.mark_applied(prior, current_layout)

        # This code below is only if it's the first gate to be seen - or if the buffer is cleared for whatever reason
        self.gate_storage.add_gate(rx_node, current_layout)

        # the new node is now also the last applied
        self.node_buffer = rx_node
        self.gap_storage[qubit] = Gap.RX

    def _apply_gate_commutative(
        self, mapped_dag: DAGCircuit, node: DAGOpNode, current_layout: Layout, canonical_register: QuantumRegister
    ):
        """
        The bread and butter of the new algorithm. Will attempt to apply gates with commutativity, otherwise build a storage list of up to one gate.
        """

        # First change the gate depending on the layout, This transformation seems to be all thats required
        node = _transform_gate_for_layout(node, current_layout, canonical_register)

        # * We first need to figure out what gate it is, and what options we have in terms of commutativity
        if len(node.qargs) == 2:
            # We first handle 2 qubit gates as they offer more in depth commutativity
            if node.op.name == "cx":  # cx = cnot gate
                # Handle inter-cnot rules (swapping on same control or same target)
                self.zc_handle_cnot(mapped_dag, node, current_layout)

            if node.op.name == "swap":
                # Swap can trade places with any single qubit gate, or a flipped version of a cnot gate
                self.zc_handle_swap(mapped_dag, node, current_layout)
                pass
        elif len(node.qargs) == 1:
            if node.op.name == "rz":
                # TODO implement rz
                self.zc_handle_rz(mapped_dag, node, current_layout)
            if node.op.name == "rx":
                # TODO implement rx
                self.zc_handle_rx(mapped_dag, node, current_layout)
            else:
                # TODO implement U (this is just swap gates i reckon)
                # Handle non specific "U" gates here.
                pass
        else:
            # raise an exception as a backup
            raise Exception(f"Unexpected gate found when applying rules. qargs: {node.qargs}, op: {node.op.name}")

        for i, gap in enumerate(self.gap_storage):
            print(f"gap: {self.gap_storage[i]}, gate: {self.gate_storage.storage[i]}")
        draw_circuit(dag_to_circuit(mapped_dag), f"output{self.iteration}")
        self.iteration += 1
        ## input()

        if self.fake_run:
            return node
        # return mapped_dag.apply_operation_back(
        #     new_node.op, new_node.qargs, new_node.cargs
        # )

    def apply_before(self, mapped_dag: DAGCircuit, prior_node: DAGOpNode, new_node: DAGOpNode) -> bool:
        """Trial a node before and after and determine which has a better effect on the depth"""
        trial1 = deepcopy(mapped_dag)
        trial1.apply_operation_back(prior_node.op, prior_node.qargs, prior_node.cargs)
        trial1.apply_operation_back(new_node.op, new_node.qargs, new_node.cargs)
        score1 = trial1.properties()["depth"]

        trial2 = deepcopy(mapped_dag)
        trial2.apply_operation_back(new_node.op, new_node.qargs, new_node.cargs)
        trial2.apply_operation_back(prior_node.op, prior_node.qargs, prior_node.cargs)
        score2 = trial2.properties()["depth"]

        print(f"Applied new gate last and got depth {score1}")
        print(f"Applied new gate first and got depth {score2}")

        if score1 > score2:
            # if the depth applying the new node first is lower, we apply first
            # This is only explicitly done if there is a depth decrease, otherwise no change is made
            draw_circuit(dag_to_circuit(trial1), f"trial1")
            draw_circuit(dag_to_circuit(trial2), f"trial2")
            # quit()
            return True
        else:
            return False

    def run(self, dag):
        """Run the SabreSwap pass on `dag`.

        Args:
            dag (DAGCircuit): the directed acyclic graph to be mapped.
        Returns:
            DAGCircuit: A dag mapped to be compatible with the coupling_map.
        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG
        """
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("Sabre swap runs on physical circuits only.")

        if len(dag.qubits) > self.coupling_map.size():
            raise TranspilerError("More virtual qubits exist than physical.")

        max_iterations_without_progress = 10 * len(dag.qubits)  # Arbitrary.
        ops_since_progress = []
        extended_set = None

        # Normally this isn't necessary, but here we want to log some objects that have some
        # non-trivial cost to create.
        do_expensive_logging = logger.isEnabledFor(logging.DEBUG)

        self.dist_matrix = self.coupling_map.distance_matrix

        rng = np.random.default_rng(self.seed)

        # Preserve input DAG's name, regs, wire_map, etc. but replace the graph.
        mapped_dag = None
        if not self.fake_run:
            # * zc doesn't update the map on a layout pass due to fake run
            mapped_dag = dag.copy_empty_like()

        canonical_register = dag.qregs["q"]
        current_layout = Layout.generate_trivial_layout(canonical_register)

        self._bit_indices = {bit: idx for idx, bit in enumerate(canonical_register)}

        # A decay factor for each qubit used to heuristically penalize recently
        # used qubits (to encourage parallelism).
        self.qubits_decay = dict.fromkeys(dag.qubits, 1)

        # Start algorithm from the front layer and iterate until all gates done.
        self.required_predecessors = self._build_required_predecessors(dag)
        num_search_steps = 0
        front_layer = dag.front_layer()

        # zc ---------
        # Doesnt need to be a class variable, as it is only used here in the initial setup
        phy_qubits = [current_layout._v2p[qubit] for qubit in dag.qubits]  # list of qubit nums e.g. [0, 1, 2, 3, 4, 5]

        self.gap_storage = {qubit: Gap.FREE for qubit in phy_qubits}  # Dictionary to store gap info

        self.gate_storage = GateStorage(phy_qubits)  # dictionary to store 2 qubit gates

        # zc ---------

        while front_layer:
            execute_gate_list = []

            # Remove as many immediately applicable gates as possible
            new_front_layer = []
            for node in front_layer:
                if len(node.qargs) == 2:
                    gate_control, gate_target = node.qargs
                    # Accessing layout._v2p directly to avoid overhead from __getitem__ and a
                    # single access isn't feasible because the layout is updated on each iteration
                    if self.coupling_map.graph.has_edge(
                        current_layout._v2p[gate_control],
                        current_layout._v2p[gate_target],
                    ):
                        execute_gate_list.append(node)
                        # * if the qubits are connected, we can execute it straight away
                    else:
                        new_front_layer.append(node)
                        # * otherwise, we add it as a not executeable gate, and the new front layer becomes the list of these unexecutables

                else:  # * Execute any single qubit gates straight away
                    execute_gate_list.append(node)
                    # TODO: This may allow us to look at the execute_gate_list for potential commutativity rules

            front_layer = new_front_layer
            # * Finish executing as many as possible, and override the front layer with the gates we can't execute

            # ! the code below is new - and prevents the algorithm from getting stuck - only works on lists where nothing is executable
            # if (
            #     not execute_gate_list
            #     and len(ops_since_progress) > max_iterations_without_progress
            # ):
            #     # Backtrack to the last time we made progress, then greedily insert swaps to route
            #     # the gate with the smallest distance between its arguments.  This is a release
            #     # valve for the algorithm to avoid infinite loops only, and should generally not
            #     # come into play for most circuits.
            #     self._undo_operations(ops_since_progress, mapped_dag, current_layout)
            #     self._add_greedy_swaps(
            #         front_layer, mapped_dag, current_layout, canonical_register
            #     )
            #     continue

            # * We can now apply the gates in the execute list
            if execute_gate_list:
                for node in execute_gate_list:
                    self._apply_gate_commutative(mapped_dag, node, current_layout, canonical_register)
                    # self._apply_gate(
                    #     mapped_dag, node, current_layout, canonical_register
                    # )

                    # Need to look into how this works, as it may be getting called for gates that haven't been applied yet
                    for successor in self._successors(node, dag):
                        self.required_predecessors[successor] -= 1
                        if self._is_resolved(successor):
                            front_layer.append(successor)

                    if node.qargs:
                        self._reset_qubits_decay()

                # Diagnostics
                if do_expensive_logging:
                    logger.debug(
                        "free! %s",
                        [(n.name if isinstance(n, DAGOpNode) else None, n.qargs) for n in execute_gate_list],
                    )
                    logger.debug(
                        "front_layer: %s",
                        [(n.name if isinstance(n, DAGOpNode) else None, n.qargs) for n in front_layer],
                    )

                ops_since_progress = []
                extended_set = None
                continue

            # After all free gates are exhausted, heuristically find
            # the best swap and insert it. When two or more swaps tie
            # for best score, pick one randomly.
            if extended_set is None:
                extended_set = self._obtain_extended_set(dag, front_layer)

            # print(extended_set)
            # draw_dag(mapped_dag, filename="testing_mapped_dag.png")
            # quit()

            swap_scores = {}
            for swap_qubits in self._obtain_swaps(front_layer, current_layout):
                trial_layout = current_layout.copy()
                trial_layout.swap(*swap_qubits)
                score = self._score_heuristic(self.heuristic, front_layer, extended_set, trial_layout, swap_qubits)
                swap_scores[swap_qubits] = score
            min_score = min(swap_scores.values())
            best_swaps = [k for k, v in swap_scores.items() if v == min_score]
            best_swaps.sort(key=lambda x: (self._bit_indices[x[0]], self._bit_indices[x[1]]))
            best_swap = rng.choice(best_swaps)
            # swap_node = self._apply_gate(
            #     mapped_dag,
            #     DAGOpNode(op=SwapGate(), qargs=best_swap),
            #     current_layout,
            #     canonical_register,
            # )
            # zc --------------
            swap_node = self._apply_gate_commutative(
                mapped_dag,
                DAGOpNode(op=SwapGate(), qargs=best_swap),
                current_layout,
                canonical_register,
            )

            # zc --------------

            # zc -----------
            p0, p1 = (
                current_layout._v2p[best_swap[0]],
                current_layout._v2p[best_swap[1]],
            )
            self.gap_storage[p0], self.gap_storage[p1] = (
                self.gap_storage[p1],
                self.gap_storage[p0],
            )
            self.gate_storage.swap_qubits(p0, p1)
            # self.gate_storage[p0], self.gate_storage[p1] = (
            #     self.gate_storage[p1],
            #     self.gate_storage[p0],
            # )
            # zc -----------

            current_layout.swap(*best_swap)
            ops_since_progress.append(swap_node)

            num_search_steps += 1
            if num_search_steps % DECAY_RESET_INTERVAL == 0:
                self._reset_qubits_decay()
            else:
                self.qubits_decay[best_swap[0]] += DECAY_RATE
                self.qubits_decay[best_swap[1]] += DECAY_RATE

            # Diagnostics
            if do_expensive_logging:
                logger.debug("SWAP Selection...")
                logger.debug("extended_set: %s", [(n.name, n.qargs) for n in extended_set])
                logger.debug("swap scores: %s", swap_scores)
                logger.debug("best swap: %s", best_swap)
                logger.debug("qubits decay: %s", self.qubits_decay)

        # Apply the last gate
        self._apply_gate(
            mapped_dag,
            self.node_buffer,
            current_layout,
            canonical_register,
        )

        self.property_set["final_layout"] = current_layout
        if not self.fake_run:
            return mapped_dag
        return dag

    def _apply_gate(self, mapped_dag, node, current_layout, canonical_register):
        # * zc Applies the gate onto the current dag, after transforming the gate
        # new_node = _transform_gate_for_layout(node, current_layout, canonical_register)
        new_node = node
        if self.fake_run:
            return new_node
        return mapped_dag.apply_operation_back(new_node.op, new_node.qargs, new_node.cargs)

    def _reset_qubits_decay(self):
        """Reset all qubit decay factors to 1 upon request (to forget about
        past penalizations).
        """
        self.qubits_decay = {k: 1 for k in self.qubits_decay.keys()}

    def _build_required_predecessors(self, dag):
        out = defaultdict(int)
        # We don't need to count in- or out-wires: outs can never be predecessors, and all input
        # wires are automatically satisfied at the start.
        for node in dag.op_nodes():
            for successor in self._successors(node, dag):
                out[successor] += 1
        return out

    def _successors(self, node, dag):
        """Return an iterable of the successors along each wire from the given node.

        This yields the same successor multiple times if there are parallel wires (e.g. two adjacent
        operations that have one clbit and qubit in common), which is important in the swapping
        algorithm for detecting if each wire has been accounted for."""
        for _, successor, _ in dag.edges(node):
            if isinstance(successor, DAGOpNode):
                yield successor

    def _is_resolved(self, node):
        """Return True if all of a node's predecessors in dag are applied."""
        return self.required_predecessors[node] == 0

    def _obtain_extended_set(self, dag, front_layer):
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
                for successor in self._successors(node, dag):
                    decremented.append(successor)
                    self.required_predecessors[successor] -= 1
                    if self._is_resolved(successor):
                        new_tmp_front_layer.append(successor)
                        if len(successor.qargs) == 2:
                            extended_set.append(successor)
                if len(extended_set) >= EXTENDED_SET_SIZE:
                    done = True
                    break
            tmp_front_layer = new_tmp_front_layer
        for node in decremented:
            self.required_predecessors[node] += 1
        return extended_set

    def _obtain_swaps(self, front_layer, current_layout):
        """Return a set of candidate swaps that affect qubits in front_layer.

        For each virtual qubit in front_layer, find its current location
        on hardware and the physical qubits in that neighborhood. Every SWAP
        on virtual qubits that corresponds to one of those physical couplings
        is a candidate SWAP.

        Candidate swaps are sorted so SWAP(i,j) and SWAP(j,i) are not duplicated.
        """
        candidate_swaps = set()
        for node in front_layer:
            for virtual in node.qargs:
                physical = current_layout[virtual]
                for neighbor in self.coupling_map.neighbors(physical):
                    virtual_neighbor = current_layout[neighbor]
                    swap = sorted([virtual, virtual_neighbor], key=lambda q: self._bit_indices[q])
                    candidate_swaps.add(tuple(swap))
        return candidate_swaps

    def _add_greedy_swaps(self, front_layer, dag, layout, qubits):
        """Mutate ``dag`` and ``layout`` by applying greedy swaps to ensure that at least one gate
        can be routed."""
        layout_map = layout._v2p
        target_node = min(
            front_layer,
            key=lambda node: self.dist_matrix[layout_map[node.qargs[0]], layout_map[node.qargs[1]]],
        )
        for pair in _shortest_swap_path(tuple(target_node.qargs), self.coupling_map, layout):
            self._apply_gate(dag, DAGOpNode(op=SwapGate(), qargs=pair), layout, qubits)
            layout.swap(*pair)

    def _compute_cost(self, layer, layout):
        cost = 0
        layout_map = layout._v2p
        for node in layer:
            cost += self.dist_matrix[layout_map[node.qargs[0]], layout_map[node.qargs[1]]]
        return cost

    def _score_heuristic(self, heuristic, front_layer, extended_set, layout, swap_qubits=None):
        """Return a heuristic score for a trial layout.

        Assuming a trial layout has resulted from a SWAP, we now assign a cost
        to it. The goodness of a layout is evaluated based on how viable it makes
        the remaining virtual gates that must be applied.
        """
        first_cost = self._compute_cost(front_layer, layout)
        if heuristic == "basic":
            return first_cost

        first_cost /= len(front_layer)
        second_cost = 0
        if extended_set:
            second_cost = self._compute_cost(extended_set, layout) / len(extended_set)
        total_cost = first_cost + EXTENDED_SET_WEIGHT * second_cost
        if heuristic == "lookahead":
            return total_cost

        if heuristic == "decay":
            return max(self.qubits_decay[swap_qubits[0]], self.qubits_decay[swap_qubits[1]]) * total_cost

        raise TranspilerError("Heuristic %s not recognized." % heuristic)

    def _undo_operations(self, operations, dag, layout):
        """Mutate ``dag`` and ``layout`` by undoing the swap gates listed in ``operations``."""
        if dag is None:
            for operation in reversed(operations):
                layout.swap(*operation.qargs)
        else:
            for operation in reversed(operations):
                dag.remove_op_node(operation)
                p0 = self._bit_indices[operation.qargs[0]]
                p1 = self._bit_indices[operation.qargs[1]]
                layout.swap(p0, p1)


def _transform_gate_for_layout(op_node, layout, device_qreg):
    """Return node implementing a virtual op on given layout."""
    # print("Transforming gate for layout")
    # print(f"Inputs: {op_node}, {layout}, {device_qreg}")
    # print(f"qargs: {op_node.qargs}")
    mapped_op_node = copy(op_node)
    mapped_op_node.qargs = tuple(device_qreg[layout._v2p[x]] for x in op_node.qargs)

    print(f"Transformed {op_node.op.name} from {op_node.qargs} to {mapped_op_node.qargs}")

    return mapped_op_node


def _shortest_swap_path(target_qubits, coupling_map, layout):
    """Return an iterator that yields the swaps between virtual qubits needed to bring the two
    virtual qubits in ``target_qubits`` together in the coupling map."""
    v_start, v_goal = target_qubits
    start, goal = layout._v2p[v_start], layout._v2p[v_goal]
    # TODO: remove the list call once using retworkx 0.12, as the return value can be sliced.
    path = list(retworkx.dijkstra_shortest_paths(coupling_map.graph, start, target=goal)[goal])
    # Swap both qubits towards the "centre" (as opposed to applying the same swaps to one) to
    # parallelise and reduce depth.
    split = len(path) // 2
    forwards, backwards = path[1:split], reversed(path[split:-1])
    for swap in forwards:
        yield v_start, layout._p2v[swap]
    for swap in backwards:
        yield v_goal, layout._p2v[swap]
