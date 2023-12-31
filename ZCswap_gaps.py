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
from collections import OrderedDict, defaultdict
from copy import copy, deepcopy
from typing import Optional, Union

import numpy as np
import retworkx
from qiskit import QuantumRegister
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.converters import dag_to_circuit
from qiskit.dagcircuit import DAGOpNode
from qiskit.dagcircuit.dagcircuit import DAGCircuit  # better for debugging
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from rich import print
from rich.panel import Panel

from helpers import draw_circuit

logger = logging.getLogger(__name__)

EXTENDED_SET_SIZE = 20  # Size of lookahead window. TODO: set dynamically to len(current_layout)
EXTENDED_SET_WEIGHT = 0.5  # Weight of lookahead window compared to front_layer.

DECAY_RATE = 0.001  # Decay coefficient for penalizing serial swaps.
DECAY_RESET_INTERVAL = 5  # How often to reset all decay rates to 1.

from enum import Enum

# Dummy putting gate here as it will be used to handle single gates later on
Gap = Enum("Gap", "GATE TARGET CONTROL FREE SWAP RZ RX U")


def get_qubits_from_layout(node: DAGOpNode, current_layout: Layout) -> Union[list[int], list[int, int]]:
    """Get the associated qubits that the gate is being applied on"""
    if len(node.qargs) == 2:
        return [current_layout._v2p[node.qargs[0]], current_layout._v2p[node.qargs[1]]]
    return [current_layout._v2p[node.qargs[0]]]


def apply_on_dag(dag: DAGCircuit, gate: DAGOpNode) -> None:
    """Apply the gate on the dag"""
    dag.apply_operation_back(gate.op, gate.qargs, gate.cargs)


class GateStorage:
    storage: OrderedDict[int, dict[str, Union[DAGOpNode, bool]]]

    def __init__(self, qubits: list[int]) -> None:
        # self.storage = {idx: {"gate": None, "applied": False} for idx in qubits}
        self.storage = OrderedDict.fromkeys(qubits, {"gate": None, "applied": False})

    def add_gate(self, gate: DAGOpNode, layout: Layout, applied=False) -> None:
        """Add a new gate to the storage, as applied or not"""
        for qubit in get_qubits_from_layout(gate, layout):
            self.storage[qubit] = {"gate": gate, "applied": applied}
            # move the most recently applied gates to the end to allow for easy final application
            self.storage.move_to_end(qubit)

    def get_gate(self, qubit: int) -> Optional[DAGOpNode]:
        """Fetch the gate thats on a certain qubit"""
        gate_info = self.storage.get(qubit, None)
        if gate_info:
            return gate_info["gate"]
        return None

    def mark_applied(self, gate: DAGOpNode, layout: Layout) -> None:
        """Mark the input gate as applied on each qubit"""
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

    def remove_gate(self, gate: DAGOpNode, layout: Layout) -> None:
        # print("removing gate")
        for qubit in get_qubits_from_layout(gate, layout):
            # print(f"removing {qubit}")
            self.storage[qubit] = {"gate": None, "applied": False}

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
        self.depth_increases = 0
        self.activations = 0
        # zc -------

    def insert_free_gaps(self, qarg1: int, qarg2: int) -> None:
        """Fill any gaps between 2 qubit nodes with free space"""
        for i in range(min(qarg1, qarg2) + 1, max(qarg1, qarg2)):
            # print(f"{i} is free")
            self.gap_storage[i] = Gap.FREE

    def apply_prior_or_buffer(self, qubits: list[int], mapped_dag: DAGCircuit, layout: Layout) -> list[int]:
        # if the gate doesnt match but applies on a line with gates stored, we need to apply the prior gate if possible, otherwise just add the gate into the buffer
        prior = None
        for qubit in qubits:
            if not self.gate_storage.is_applied(qubit) and prior is None:
                prior = self.gate_storage.get_gate(qubit)

        # Handles the case where there isn't an applicable gate, and applies the current gate in the buffer (or last gate)
        if isinstance(self.node_buffer, DAGOpNode) and prior is None:
            prior = self.node_buffer

        if prior:
            # print("found a prior and applying it")
            apply_on_dag(mapped_dag, prior)
            # update prior in storage
            self.gate_storage.mark_applied(prior, layout)

    def get_prior(self, new_node: DAGOpNode, current_layout: Layout) -> list[DAGOpNode]:
        # Get the prior gates of the new node
        priors = []
        for qubit in get_qubits_from_layout(new_node, current_layout):
            prior = self.gate_storage.get_gate(qubit)
            if prior is not None:
                priors.append(prior)

        p1, p2 = 0, 0
        if len(priors) == 2 and None not in priors:
            for i, (key, value) in enumerate(self.gate_storage.storage.items()):
                # print(key, value["gate"])
                if value["gate"] == priors[0]:
                    p1 = i
                    # print(f"We got a match at {i}")
                elif value["gate"] == priors[1]:
                    p2 = i
                    # print(f"Other found at {i}")

        if p2 > p1:
            # if the gate comes before, swap them
            priors = priors[::-1]

        # return it sorted by how many qubits the gate span, with smaller being returned first
        # sorted(priors, key=lambda x: x.op.num_qubits)
        return priors

    def zc_handle_single_universal(self, mapped_dag: DAGCircuit, new_node: DAGOpNode, current_layout: Layout):
        # print("Handling universal")
        # TODO find out why some gates aren't making it... - seems to be a majority of H gates....

        qubit = get_qubits_from_layout(new_node, current_layout)[0]

        prior = self.gate_storage.get_gate(qubit)
        if prior is not None:
            # Apply special considerations if it's a swap gate
            if prior.op.name == "swap":
                # print("working with uni gate ", new_node.op.name)
                if self.trial_single_on_swap(mapped_dag, prior, new_node):
                    # print("better to apply it before rather than after")
                    apply_on_dag(mapped_dag, new_node)
                    return
                else:
                    # print("nothing changed")
                    apply_on_dag(mapped_dag, prior)
                    self.gate_storage.remove_gate(prior, current_layout)
                    self.gate_storage.add_gate(new_node, current_layout)
                    return
            else:
                # if it has a prior gate, we need to remove and replace it
                apply_on_dag(mapped_dag, prior)
                self.gate_storage.remove_gate(prior, current_layout)
                self.gate_storage.add_gate(new_node, current_layout)
                return

        # if theres no prior we just store it
        self.gate_storage.add_gate(new_node, current_layout)

    def trial_single_on_swap(
        self, mapped_dag: DAGCircuit, swap_node: DAGOpNode, new_node: DAGOpNode, after=True
    ) -> bool:
        # Trial a single qubit gate before and after a swap
        # after: if the gate is after the swap

        # Collect and modify the qargs to be on the other side of the swap
        qargs = list(swap_node.qargs)
        # We know that the gate lies on the swap so the opposite qubit is also in the list
        qargs.remove(list(new_node.qargs)[0])

        new_node_before = deepcopy(new_node)
        new_node_before.qargs = tuple(qargs)

        if not after:
            # swap the order if the initial new node is before rather than after
            new_node, new_node_before = new_node_before, new_node

        # print(f"new op: {new_node.op.name}\n new before: {new_node_before.op.name}\n swap: {swap_node.op.name}")

        # Trial the swap before the node
        trial1 = deepcopy(mapped_dag)
        trial1.apply_operation_back(swap_node.op, swap_node.qargs, swap_node.cargs)
        trial1.apply_operation_back(new_node.op, new_node.qargs, new_node.cargs)
        score1 = trial1.properties()["depth"]

        # Trial the node before the swap
        trial2 = deepcopy(mapped_dag)
        trial2.apply_operation_back(new_node_before.op, new_node_before.qargs, new_node_before.cargs)
        trial2.apply_operation_back(swap_node.op, swap_node.qargs, swap_node.cargs)
        score2 = trial2.properties()["depth"]

        # print(f"Applied new gate first and got depth {score1}")
        # print(f"Applied new gate last and got depth {score2}")
        self.activations += 1

        if score1 < score2:
            # Better to apply before
            # draw_circuit(dag_to_circuit(trial1), f"trial1")
            # draw_circuit(dag_to_circuit(trial2), f"trial2")
            self.depth_increases += 1
            return True

        else:
            # better to apply after
            return False

    def trial_cnot_swap(
        self, mapped_dag: DAGCircuit, current_layout: Layout, swap_node: DAGOpNode, cnot_node: DAGOpNode, after=True
    ) -> bool:
        # A Function to determine whether or not to apply a cnot gate with commutativity rules with a swap
        #
        # it will first determine whether the cnot is on both lines of the swap, or vice versa
        #   if it is, we can reverse the qargs of the cnot and try both sides
        # if it's only on one line of the swap
        #   we need to swap it to the other side, and swap the lines of the node that lies on the swap - but also check connectivity before attempting

        # print("we trialling the swap")
        swap_qargs, cnot_qargs = list(swap_node.qargs), list(cnot_node.qargs)

        common_qargs = [qarg for qarg in cnot_qargs if qarg in swap_qargs]

        cnot_before = deepcopy(cnot_node)  # create a copy node for the other side of the swap

        # print(f"cnot qargs {cnot_qargs}; swap qargs {swap_qargs}")
        # print(f"Common {common_qargs}")

        # Setup the before node with the correct qargs
        if len(common_qargs) == 2:
            # Both lie on the same line
            cnot_before.qargs = tuple(cnot_qargs[::-1])  # reverse the qargs

        elif len(common_qargs) == 1:
            # only one lies on the same line

            # we need to know if the control or target matches, for the sake of order
            index = cnot_qargs.index(common_qargs[0])

            # remove the common qargs
            cnot_qargs.remove(common_qargs[0])
            swap_qargs.remove(common_qargs[0])

            if index == 0:
                # match on control
                cnot_before.qargs = tuple(swap_qargs + cnot_qargs)

            else:
                # match on target
                cnot_before.qargs = tuple(cnot_qargs + swap_qargs)

        else:
            # none lie here - no commutativity
            raise Exception("Prior gate issue when trialling swap - Prior gate is not related")

        # revert the swap to see what happens
        current_layout.swap(*get_qubits_from_layout(swap_node, current_layout))
        if not self.coupling_map.graph.has_edge(*get_qubits_from_layout(cnot_before, current_layout)):
            # print("commutation doesnt work here")
            return False

        if not after:
            cnot_node, cnot_before = cnot_before, cnot_node

        # Trial the swap before the node
        trial1 = deepcopy(mapped_dag)
        trial1.apply_operation_back(swap_node.op, swap_node.qargs, swap_node.cargs)
        trial1.apply_operation_back(cnot_node.op, cnot_node.qargs, cnot_node.cargs)
        score1 = trial1.properties()["depth"]

        # Trial the node before the swap
        trial2 = deepcopy(mapped_dag)
        trial2.apply_operation_back(cnot_before.op, cnot_before.qargs, cnot_before.cargs)
        trial2.apply_operation_back(swap_node.op, swap_node.qargs, swap_node.cargs)
        score2 = trial2.properties()["depth"]

        self.activations += 1
        if score1 < score2:
            # Better to apply before
            # draw_circuit(dag_to_circuit(trial1), f"trial1")
            # draw_circuit(dag_to_circuit(trial2), f"trial2")
            self.depth_increases += 1
            return True

        else:
            # better to apply after
            return False

    def zc_handle_rz_new(self, mapped_dag: DAGCircuit, new_node: DAGOpNode, current_layout: Layout):
        qubit = get_qubits_from_layout(new_node, current_layout)[0]

        prior = self.gate_storage.get_gate(qubit)
        if prior is not None:
            if prior.op.name == "cx":
                # print("we got a prior for rz")
                # only rule applies on cnot gates
                prior_qubits = get_qubits_from_layout(prior, current_layout)
                if qubit == prior_qubits[0]:
                    # We can attempt to apply it if its on the control
                    if self.apply_before(mapped_dag, prior, new_node):
                        # print("rz apply first")
                        apply_on_dag(mapped_dag, new_node)
                        # Prior is still unapplied, so we leave it
                        return
                    # print("rz apply after")
                    apply_on_dag(mapped_dag, prior)
                    self.gate_storage.remove_gate(prior, current_layout)
                    self.gate_storage.add_gate(new_node, current_layout)
                    return

                else:
                    # print("rz apply after")
                    apply_on_dag(mapped_dag, prior)
                    self.gate_storage.remove_gate(prior, current_layout)
                    self.gate_storage.add_gate(new_node, current_layout)
                    return

            elif prior.op.name == "swap":
                if self.trial_single_on_swap(mapped_dag, prior, new_node):
                    # print("better to apply it before rather than after")
                    apply_on_dag(mapped_dag, new_node)
                    return
                else:
                    # print("nothing changed")
                    apply_on_dag(mapped_dag, prior)
                    self.gate_storage.remove_gate(prior, current_layout)
                    self.gate_storage.add_gate(new_node, current_layout)
                    return

            else:
                # handle any other gate here by applying prior first
                apply_on_dag(mapped_dag, prior)
                self.gate_storage.remove_gate(prior, current_layout)
                self.gate_storage.add_gate(new_node, current_layout)
                return

        # print("storing rz")
        # if theres no prior we just store it
        self.gate_storage.add_gate(new_node, current_layout)

    def zc_handle_rx_new(self, mapped_dag: DAGCircuit, new_node: DAGOpNode, current_layout: Layout):
        qubit = get_qubits_from_layout(new_node, current_layout)[0]

        prior = self.gate_storage.get_gate(qubit)
        if prior is not None:
            if prior.op.name == "cx":
                # print("we got a prior for rx")
                # only rule applies on cnot gates
                prior_qubits = get_qubits_from_layout(prior, current_layout)
                if qubit == prior_qubits[1]:  # ! could potetntially include this with rz with just the change here
                    # We can attempt to apply it if its on the control
                    if self.apply_before(mapped_dag, prior, new_node):
                        # print("rx apply first")
                        apply_on_dag(mapped_dag, new_node)
                        # Prior is still unapplied, so we leave it
                        return
                    # print("rx apply after")
                    apply_on_dag(mapped_dag, prior)
                    self.gate_storage.remove_gate(prior, current_layout)
                    self.gate_storage.add_gate(new_node, current_layout)
                    return

                else:
                    # print("rx apply after")
                    apply_on_dag(mapped_dag, prior)
                    self.gate_storage.remove_gate(prior, current_layout)
                    self.gate_storage.add_gate(new_node, current_layout)
                    return

            elif prior.op.name == "swap":
                if self.trial_single_on_swap(mapped_dag, prior, new_node):
                    # print("better to apply it before rather than after")
                    apply_on_dag(mapped_dag, new_node)
                    return
                else:
                    # print("nothing changed")
                    apply_on_dag(mapped_dag, prior)
                    self.gate_storage.remove_gate(prior, current_layout)
                    self.gate_storage.add_gate(new_node, current_layout)
                    return

            else:
                # handle any other gate here by applying prior first
                apply_on_dag(mapped_dag, prior)
                self.gate_storage.remove_gate(prior, current_layout)
                self.gate_storage.add_gate(new_node, current_layout)
                return

        # print("storing rx")
        # if theres no prior we just store it
        self.gate_storage.add_gate(new_node, current_layout)

    def zc_handle_cnot_new(self, mapped_dag: DAGCircuit, new_node: DAGOpNode, current_layout: Layout):
        priors = self.get_prior(new_node, current_layout)
        for prior in priors:  # ! remove this later dont need it
            # print(f"handling prior gate on {get_qubits_from_layout(prior, current_layout)}")
            if len(priors) == 2:  # we need to apply the other gate first - this is universal at this point
                # print("Applying 2nd gate first")
                apply_on_dag(mapped_dag, prior)
                self.gate_storage.remove_gate(prior, current_layout)
                priors = priors[1:]
                continue

            # Anything below here is commuatative
            prior_qubits = get_qubits_from_layout(prior, current_layout)
            prior_op = prior.op.name
            # print(f"prior q: {prior_qubits}, prior op: {prior_op}")
            new_qubits = get_qubits_from_layout(new_node, current_layout)
            if len(prior_qubits) == 2:  #! honestly could remove this as well
                # manage 2 qubit gates (swap and cnot)
                if prior_op == "cx":
                    # cnot are ordered by [control, target] so [0] for control and [1] for target
                    if new_qubits[0] == prior_qubits[0] and new_qubits[1] != prior_qubits[1]:
                        # control match
                        if self.apply_before(mapped_dag, prior, new_node):
                            # print("cnot control match apply first")
                            apply_on_dag(mapped_dag, new_node)
                            # Prior is still unapplied, so we leave it
                            return
                        # print("cnot control match apply after")
                        apply_on_dag(mapped_dag, prior)
                        self.gate_storage.remove_gate(prior, current_layout)
                        self.gate_storage.add_gate(new_node, current_layout)
                        return

                    elif new_qubits[1] == prior_qubits[1] and new_qubits[0] != new_qubits[0]:
                        # target match
                        if self.apply_before(mapped_dag, prior, new_node):
                            # print("cnot target match apply first")
                            apply_on_dag(mapped_dag, new_node)
                            # Prior is still unapplied, so we leave it
                            return
                        # print("cnot control match apply after")
                        apply_on_dag(mapped_dag, prior)
                        self.gate_storage.remove_gate(prior, current_layout)
                        self.gate_storage.add_gate(new_node, current_layout)
                        return

                    else:
                        # no match just store to prior
                        apply_on_dag(mapped_dag, prior)
                        self.gate_storage.remove_gate(prior, current_layout)
                        self.gate_storage.add_gate(new_node, current_layout)
                        return

                elif prior_op == "swap":
                    # handle swap
                    # ! I think here we should just force apply the swap - because the change in layout may prove difficult to manage but I'll double check later
                    if self.trial_cnot_swap(mapped_dag, current_layout, prior, new_node):
                        # print("apply the new cnot gate before the swap")
                        apply_on_dag(mapped_dag, new_node)
                        # Prior is still unapplied, so we leave it
                        return
                    # print("apply the prior swap first")
                    apply_on_dag(mapped_dag, prior)
                    self.gate_storage.remove_gate(prior, current_layout)
                    self.gate_storage.add_gate(new_node, current_layout)
                    return

                else:
                    # generic 2 qubit gates here like cz, cant do commutation so apply the new before
                    apply_on_dag(mapped_dag, prior)
                    self.gate_storage.remove_gate(prior, current_layout)
                    self.gate_storage.add_gate(new_node, current_layout)
                    return

            elif len(prior_qubits) == 1:
                # manage single qubit gates (rz, rx, phase gates)
                # they only have a single qubit, hence the 0 index on prior
                if prior_op == "rz":
                    if new_qubits[0] == prior_qubits[0]:
                        # rz gates can only be applied on control
                        if self.apply_before(mapped_dag, prior, new_node):
                            # print("rz apply first")
                            apply_on_dag(mapped_dag, new_node)
                            # Prior is still unapplied, so we leave it
                            return
                        # print("rz apply after")
                        apply_on_dag(mapped_dag, prior)
                        self.gate_storage.remove_gate(prior, current_layout)
                        self.gate_storage.add_gate(new_node, current_layout)
                        return
                    else:
                        # ! gross but works for now
                        # print("rz apply after")
                        apply_on_dag(mapped_dag, prior)
                        self.gate_storage.remove_gate(prior, current_layout)
                        self.gate_storage.add_gate(new_node, current_layout)
                        return

                elif prior_op == "rx":
                    if new_qubits[1] == prior_qubits[0]:
                        # rx gates can only be applied on target
                        if self.apply_before(mapped_dag, prior, new_node):
                            # print("rx apply first")
                            apply_on_dag(mapped_dag, new_node)
                            # Prior is still unapplied, so we leave it
                            return
                        # print("rx apply after")
                        apply_on_dag(mapped_dag, prior)
                        self.gate_storage.remove_gate(prior, current_layout)
                        self.gate_storage.add_gate(new_node, current_layout)
                        return
                    else:
                        # ! again gross but needed
                        # print("rx apply after")
                        apply_on_dag(mapped_dag, prior)
                        self.gate_storage.remove_gate(prior, current_layout)
                        self.gate_storage.add_gate(new_node, current_layout)
                        return

                else:
                    # generic single qubit gates here like x, cant do commutation so apply the new before
                    apply_on_dag(mapped_dag, prior)
                    self.gate_storage.remove_gate(prior, current_layout)
                    self.gate_storage.add_gate(new_node, current_layout)
                    return

            raise Exception("if it gets here we got an issue")
            # # ! Old code fix later
            # if self.apply_before(mapped_dag, prior, new_node):
            #     print("Applying it first")
            #     apply_on_dag(mapped_dag, new_node)
            #     # Prior is still unapplied, so we leave it
            #     return
            # print("Better to apply it after")
            # # Apply the prior first as we can't apply a rule
            # apply_on_dag(mapped_dag, prior)
            # self.gate_storage.remove_gate(prior, current_layout)
            # self.gate_storage.add_gate(new_node, current_layout)
            # return

        # If no prior gates then we can just save it
        # print("no matches so we just save it for now")
        self.gate_storage.add_gate(new_node, current_layout)

        # new_control, new_target = get_qubits_from_layout(new_node, current_layout)
        # prior_control = self.gate_storage.get_gate(new_control)
        # prior_target = self.gate_storage.get_gate(new_target)

        # if prior_control is not None:  # Could be none or a gate
        #     if prior_control.op.name == "cx":

        #     if prior_control.op.name == "swap":
        #         pass
        #     # Put other if statements here for other gates

    def zc_handle_cnot(self, mapped_dag: DAGCircuit, new_node: DAGOpNode, current_layout: Layout):
        """Apply a CNOT gate to the circuit, attempting to apply any commutativity rules"""

        gate_control, gate_target = get_qubits_from_layout(new_node, current_layout)

        # we can make any qubits inbetween the gate a gap.
        self.insert_free_gaps(gate_control, gate_target)

        # print(
        #     Panel(
        #         f"{self.iteration}: Attempting application of {new_node.name} gate on control {gate_control} and target {gate_target}",
        #         highlight=True,
        #     )
        # )

        # * Begin main logic here

        # Check if the new gate's control lies on the same line as a previous gates control, and their targets don't match - and the gate hasn't been applied yet
        # Implements Inter-CNOT Rule A
        if (
            self.gap_storage[gate_control] == Gap.CONTROL
            and self.gap_storage[gate_target] != Gap.TARGET
            and not self.gate_storage.is_applied(gate_control)
        ):
            # print(f"Found potential control match on qubit {gate_control}")

            prior = self.gate_storage.get_gate(gate_control)
            prior_control, prior_target = get_qubits_from_layout(prior, current_layout)
            # print(f"prior gate on qubits {prior_control}, {prior_target}")

            if self.apply_before(mapped_dag, prior, new_node):
                # print("we should apply the new node before")
                # mapped_dag.apply_operation_back(new_node.op, new_node.qargs, new_node.cargs)
                apply_on_dag(mapped_dag, new_node)
                self.gap_storage[prior_control] = Gap.CONTROL
                self.gap_storage[prior_target] = Gap.TARGET

                # since we applied the prior gate second, the storage doesn't have to update
                return

            # print("we should do it the same way")
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
            # print(f"Found potential target match on qubit {gate_target}")

            prior = self.gate_storage.get_gate(gate_target)
            prior_control, prior_target = get_qubits_from_layout(prior, current_layout)
            # print(f"prior gate on qubits {prior_control}, {prior_target}")

            if self.apply_before(mapped_dag, prior, new_node):
                # print("we should apply the new node before")
                # mapped_dag.apply_operation_back(new_node.op, new_node.qargs, new_node.cargs)
                apply_on_dag(mapped_dag, new_node)
                # self.gap_storage[gate_control] = Gap.FREE
                self.gap_storage[prior_control] = Gap.CONTROL
                self.gap_storage[prior_target] = Gap.TARGET

                # since we applied the prior gate second, the storage doesn't have to update
                # * If we decide to remove the gap storage, we'll need to add the new gate applied to the storage
                return

            # print("we should do it the same way")
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

        self.apply_prior_or_buffer([gate_control, gate_target], mapped_dag, current_layout)

        # This code below is only if it's the first gate to be seen - or if the buffer is cleared for whatever reason
        self.gate_storage.add_gate(new_node, current_layout)

        # the new node is now also the last applied
        self.node_buffer = new_node

        self.gap_storage[gate_control], self.gap_storage[gate_target] = (
            Gap.CONTROL,
            Gap.TARGET,
        )

    def zc_handle_swap_new(self, mapped_dag: DAGCircuit, swap_node: DAGOpNode, current_layout: Layout):
        # The way this should work is
        # - if theres 2 prior gates, apply the one that comes before, and then try commutativity with the other
        #   - if its a single qubit, essentially just do apply before or after
        #   - if its a 2 qubit, it needs to be flipped if applied after

        priors = self.get_prior(swap_node, current_layout)
        # print("handling a swap")

        if len(priors) == 2:  # we need to apply the other gate first
            prior = priors[0]
            apply_on_dag(mapped_dag, prior)
            self.gate_storage.remove_gate(prior, current_layout)
            priors = priors[1:]

        if len(priors) == 1:
            prior = priors[0]
            if len(prior.qargs) == 1:
                if self.trial_single_on_swap(mapped_dag, swap_node, prior, after=False):
                    apply_on_dag(mapped_dag, swap_node)
                    return
                else:
                    # Apply the prior first as we can't apply a rule
                    apply_on_dag(mapped_dag, prior)
                    self.gate_storage.remove_gate(prior, current_layout)
                    self.gate_storage.add_gate(swap_node, current_layout)
                    return

            else:
                if self.apply_before(mapped_dag, prior, swap_node):
                    apply_on_dag(mapped_dag, swap_node)
                    # Prior is still unapplied, so we leave it
                    return
                else:
                    # Apply the prior first as we can't apply a rule
                    apply_on_dag(mapped_dag, prior)
                    self.gate_storage.remove_gate(prior, current_layout)
                    self.gate_storage.add_gate(swap_node, current_layout)
                    return

        # print("no matches so we just save it for now")
        self.gate_storage.add_gate(swap_node, current_layout)

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
                buffer_qubit = get_qubits_from_layout(self.node_buffer, current_layout)[0]
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

        # print(Panel(f"{self.iteration}: Applying SWAP gate on qubits {targ1} and {targ2}"))

        # modify the gaps to include swap
        self.gap_storage[targ1], self.gap_storage[targ2] = Gap.SWAP, Gap.SWAP
        self.gate_storage.add_gate(swap_node, current_layout, applied=True)

        # TODO apply swap commutation rules
        apply_on_dag(mapped_dag, swap_node)

    def zc_handle_rz(self, mapped_dag: DAGCircuit, rz_node: DAGOpNode, current_layout: Layout):
        # Need to handle the situation where the rz gate comes after the CNOT gate

        qubit = get_qubits_from_layout(rz_node, current_layout)[0]

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
        self.apply_prior_or_buffer([qubit], mapped_dag, current_layout)

        # This code below is only if it's the first gate to be seen - or if the buffer is cleared for whatever reason
        self.gate_storage.add_gate(rz_node, current_layout)

        # the new node is now also the last applied
        self.node_buffer = rz_node
        self.gap_storage[qubit] = Gap.RZ

    def zc_handle_rx(self, mapped_dag: DAGCircuit, rx_node: DAGOpNode, current_layout: Layout):
        # Need to handle the situation where the rz gate comes after the CNOT gate

        qubit = get_qubits_from_layout(rx_node, current_layout)[0]

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
        self.apply_prior_or_buffer([qubit], mapped_dag, current_layout)

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

        # print(Panel(f"{self.iteration}: {node.op.name} on qubits {get_qubits_from_layout(node, current_layout)}"))

        # * We first need to figure out what gate it is, and what options we have in terms of commutativity
        # self.zc_handle_cnot_new(mapped_dag, node, current_layout)

        if len(node.qargs) == 2:
            # We first handle 2 qubit gates as they offer more in depth commutativity
            if node.op.name == "cx":  # cx = cnot gate
                # Handle inter-cnot rules (swapping on same control or same target)
                # self.zc_handle_cnot(mapped_dag, node, current_layout)
                self.zc_handle_cnot_new(mapped_dag, node, current_layout)

            elif node.op.name == "swap":
                # Swap can trade places with any single qubit gate, or a flipped version of a cnot gate
                # self.zc_handle_swap(mapped_dag, node, current_layout)
                self.zc_handle_swap_new(mapped_dag, node, current_layout)
        elif len(node.qargs) == 1:
            if node.op.name == "rz":
                self.zc_handle_rz_new(mapped_dag, node, current_layout)
                # print("out of handling rz")
            elif node.op.name == "rx":
                self.zc_handle_rx_new(mapped_dag, node, current_layout)
            else:
                #     # TODO implement U (this is just swap gates i reckon)
                # Handle non specific "U" gates here.
                self.zc_handle_single_universal(mapped_dag, node, current_layout)
        else:
            # raise an exception as a backup
            # raise Exception(f"Unexpected gate found when applying rules. qargs: {node.qargs}, op: {node.op.name}")
            print(f"WARNING: Unexpected gate found when applying rules. qargs: {node.qargs}, op: {node.op.name}")

        # print("After:")
        # for i, gap in enumerate(self.gap_storage):
        #     print(f"gap: {self.gap_storage[i]}, gate: {self.gate_storage.storage[i]['gate']}")
        # print(self.gate_storage.storage)
        # draw_circuit(dag_to_circuit(mapped_dag), f"output{self.iteration}")
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

        # print(f"Applied new gate last and got depth {score1}")
        # print(f"Applied new gate first and got depth {score2}")
        self.activations += 1

        if score1 > score2:
            # if the depth applying the new node first is lower, we apply first
            # This is only explicitly done if there is a depth decrease, otherwise no change is made

            # draw_circuit(dag_to_circuit(trial1), f"trial1")
            # draw_circuit(dag_to_circuit(trial2), f"trial2")
            self.depth_increases += 1
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

        test = []
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
                    test.append(node)
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
        # self._apply_gate(
        #     mapped_dag,
        #     self.node_buffer,
        #     current_layout,
        #     canonical_register,
        # )
        # print(f"Final storage: {self.gate_storage}")

        applied = [None]
        for qubit, info in reversed(self.gate_storage.storage.items()):
            gate = info["gate"]
            # print(gate)
            if gate not in applied:
                applied.append(gate)
                # print("added new")
                self._apply_gate(mapped_dag, gate, current_layout, canonical_register)
                continue
            # print("already in there")

        self.property_set["final_layout"] = current_layout
        print(f"depth increases: {self.depth_increases}")
        if not self.fake_run:
            return mapped_dag, self.depth_increases, self.activations
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

    # print(f"Transformed {op_node.op.name} from {op_node.qargs} to {mapped_op_node.qargs}")

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
