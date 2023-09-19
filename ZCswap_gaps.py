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
from helpers import draw_dag

# zc ---------

logger = logging.getLogger(__name__)

EXTENDED_SET_SIZE = (
    20  # Size of lookahead window. TODO: set dynamically to len(current_layout)
)
EXTENDED_SET_WEIGHT = 0.5  # Weight of lookahead window compared to front_layer.

DECAY_RATE = 0.001  # Decay coefficient for penalizing serial swaps.
DECAY_RESET_INTERVAL = 5  # How often to reset all decay rates to 1.

from enum import Enum

# Dummy putting gate here as it will be used to handle single gates later on
Gap = Enum("Gap", "GATE TARGET CONTROL FREE")


class SabreSwap(TransformationPass):
    def __init__(
        self,
        coupling_map,
        heuristic="basic",
        seed=None,
        fake_run=False,
    ):
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
        self.phy_qubits = None
        self.gap_storage = None
        self.gate_storage = None
        # zc -------

    def _apply_gate_commutative(
        self,
        mapped_dag: DAGCircuit,
        node: DAGOpNode,
        current_layout: Layout,
        canonical_register: QuantumRegister,
    ):
        """
        The bread and butter of the new algorithm. Will attempt to apply gates with commutativity, otherwise build a storage list of up to one gate.
        """
        # * zc Applies the gate onto the current dag, after transforming the gate
        new_node = _transform_gate_for_layout(node, current_layout, canonical_register)

        v0, v1 = (
            current_layout._v2p[new_node.qargs[0]],
            current_layout._v2p[new_node.qargs[1]],
        )

        print(
            Panel(
                f"Attempting application of {node.name} gate on qubits {v0} and {v1}",
                highlight=True,
            )
        )

        for i in range(min(v0, v1) + 1, max(v0, v1)):
            print(f"{i} is free")
            self.gap_storage[i] = Gap.FREE

        # * The "and" is required here to ensure that the 2 gates are not applied on the same qubits, as there would be no option there.
        if self.gap_storage[v0] == Gap.CONTROL and self.gap_storage[v1] != Gap.TARGET:
            print(f"Found potential control match on qubit {v0}")

            # TODO Need to trial the current layout and mapping, and determine if applying the rule will decrease depth
            # TODO If it doesnt, we will keep the current setup, but if it does we will swap execution
            # TODO we can do this using the current mapped dag, copying it, and applying the gate before/after. This will include the prior stored gate under the gap. If we are applying it early, we apply the newer gate first (to fit it in) and then the previous gate can stay in storage (gaps will need to be updated here as well), otherwise, the prior gate will be applied and the new gate will be put into the storage, along with the updated gaps from the prior gate.

        elif self.gap_storage[v1] == Gap.TARGET and self.gap_storage[v0] != Gap.CONTROL:
            print(f"Found potential target match on qubit {v0}")

        else:
            self.gate_storage[v0], self.gate_storage[v1] = node, node

        self.gap_storage[v0], self.gap_storage[v1] = Gap.CONTROL, Gap.TARGET

        print(self.gate_storage)

        if self.fake_run:
            return new_node
        return mapped_dag.apply_operation_back(
            new_node.op, new_node.qargs, new_node.cargs
        )

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
        self.phy_qubits = [
            current_layout._v2p[qubit] for qubit in dag.qubits
        ]  # list of qubit nums e.g. [0, 1, 2, 3, 4, 5]

        self.gap_storage = {
            qubit: Gap.FREE for qubit in self.phy_qubits
        }  # Dictionary to store gap info

        self.gate_storage = {
            idx: [] for idx in self.phy_qubits
        }  # dictionary to store 2 qubit gates

        # zc ---------

        while front_layer:
            execute_gate_list = []

            # Remove as many immediately applicable gates as possible
            new_front_layer = []
            for node in front_layer:
                if len(node.qargs) == 2:
                    v0, v1 = node.qargs
                    # Accessing layout._v2p directly to avoid overhead from __getitem__ and a
                    # single access isn't feasible because the layout is updated on each iteration
                    if self.coupling_map.graph.has_edge(
                        current_layout._v2p[v0], current_layout._v2p[v1]
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
            if (
                not execute_gate_list
                and len(ops_since_progress) > max_iterations_without_progress
            ):
                # Backtrack to the last time we made progress, then greedily insert swaps to route
                # the gate with the smallest distance between its arguments.  This is a release
                # valve for the algorithm to avoid infinite loops only, and should generally not
                # come into play for most circuits.
                self._undo_operations(ops_since_progress, mapped_dag, current_layout)
                self._add_greedy_swaps(
                    front_layer, mapped_dag, current_layout, canonical_register
                )
                continue

            # * We can now apply the gates in the execute list
            if execute_gate_list:
                for node in execute_gate_list:
                    self._apply_gate(
                        mapped_dag, node, current_layout, canonical_register
                    )
                    # self._apply_gate(
                    #     mapped_dag, node, current_layout, canonical_register
                    # )
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
                        [
                            (n.name if isinstance(n, DAGOpNode) else None, n.qargs)
                            for n in execute_gate_list
                        ],
                    )
                    logger.debug(
                        "front_layer: %s",
                        [
                            (n.name if isinstance(n, DAGOpNode) else None, n.qargs)
                            for n in front_layer
                        ],
                    )

                ops_since_progress = []
                extended_set = None
                continue

            # After all free gates are exhausted, heuristically find
            # the best swap and insert it. When two or more swaps tie
            # for best score, pick one randomly.
            if extended_set is None:
                extended_set = self._obtain_extended_set(dag, front_layer)

            print(extended_set)
            draw_dag(mapped_dag, filename="testing_mapped_dag.png")
            quit()

            swap_scores = {}
            for swap_qubits in self._obtain_swaps(front_layer, current_layout):
                trial_layout = current_layout.copy()
                trial_layout.swap(*swap_qubits)
                score = self._score_heuristic(
                    self.heuristic, front_layer, extended_set, trial_layout, swap_qubits
                )
                swap_scores[swap_qubits] = score
            min_score = min(swap_scores.values())
            best_swaps = [k for k, v in swap_scores.items() if v == min_score]
            best_swaps.sort(
                key=lambda x: (self._bit_indices[x[0]], self._bit_indices[x[1]])
            )
            best_swap = rng.choice(best_swaps)
            swap_node = self._apply_gate(
                mapped_dag,
                DAGOpNode(op=SwapGate(), qargs=best_swap),
                current_layout,
                canonical_register,
            )
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
                logger.debug(
                    "extended_set: %s", [(n.name, n.qargs) for n in extended_set]
                )
                logger.debug("swap scores: %s", swap_scores)
                logger.debug("best swap: %s", best_swap)
                logger.debug("qubits decay: %s", self.qubits_decay)

        self.property_set["final_layout"] = current_layout
        if not self.fake_run:
            return mapped_dag
        return dag

    def _apply_gate(self, mapped_dag, node, current_layout, canonical_register):
        # * zc Applies the gate onto the current dag, after transforming the gate
        new_node = _transform_gate_for_layout(node, current_layout, canonical_register)
        if self.fake_run:
            return new_node
        return mapped_dag.apply_operation_back(
            new_node.op, new_node.qargs, new_node.cargs
        )

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
                    swap = sorted(
                        [virtual, virtual_neighbor], key=lambda q: self._bit_indices[q]
                    )
                    candidate_swaps.add(tuple(swap))
        return candidate_swaps

    def _add_greedy_swaps(self, front_layer, dag, layout, qubits):
        """Mutate ``dag`` and ``layout`` by applying greedy swaps to ensure that at least one gate
        can be routed."""
        layout_map = layout._v2p
        target_node = min(
            front_layer,
            key=lambda node: self.dist_matrix[
                layout_map[node.qargs[0]], layout_map[node.qargs[1]]
            ],
        )
        for pair in _shortest_swap_path(
            tuple(target_node.qargs), self.coupling_map, layout
        ):
            self._apply_gate(dag, DAGOpNode(op=SwapGate(), qargs=pair), layout, qubits)
            layout.swap(*pair)

    def _compute_cost(self, layer, layout):
        cost = 0
        layout_map = layout._v2p
        for node in layer:
            cost += self.dist_matrix[
                layout_map[node.qargs[0]], layout_map[node.qargs[1]]
            ]
        return cost

    def _score_heuristic(
        self, heuristic, front_layer, extended_set, layout, swap_qubits=None
    ):
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
            return (
                max(
                    self.qubits_decay[swap_qubits[0]], self.qubits_decay[swap_qubits[1]]
                )
                * total_cost
            )

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
    return mapped_op_node


def _shortest_swap_path(target_qubits, coupling_map, layout):
    """Return an iterator that yields the swaps between virtual qubits needed to bring the two
    virtual qubits in ``target_qubits`` together in the coupling map."""
    v_start, v_goal = target_qubits
    start, goal = layout._v2p[v_start], layout._v2p[v_goal]
    # TODO: remove the list call once using retworkx 0.12, as the return value can be sliced.
    path = list(
        retworkx.dijkstra_shortest_paths(coupling_map.graph, start, target=goal)[goal]
    )
    # Swap both qubits towards the "centre" (as opposed to applying the same swaps to one) to
    # parallelise and reduce depth.
    split = len(path) // 2
    forwards, backwards = path[1:split], reversed(path[split:-1])
    for swap in forwards:
        yield v_start, layout._p2v[swap]
    for swap in backwards:
        yield v_goal, layout._p2v[swap]
