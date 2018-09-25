import parsimonious
from .. import circuit as ct
import numpy as np
import functools

qasm_grammar = parsimonious.Grammar(r"""
        program = nl* (qubit_spec)* nl (circuit_spec)+ (dangling_wait)*
        qubit_spec = "qubits " id nl

        dangling_wait = ("qwait " numarg nl)?

        initall = initall_named / initall_fake
        initall_named = "." id
        initall_fake = ""
        gatelist = gate / ("{" gate (more_gates)* "}")
        circuit_spec = initall nl (gatelist nl)* meas_block nl*
        more_gates = ("|" gate)

        gate = !meas_block ws* (wait_gate / two_qubit_gate / single_qubit_gate) ws*


        wait_gate = "qwait" ws numarg
        single_qubit_gate = gate_name ws arg
        two_qubit_gate = gate_name ws arg ws* "," ws* arg
        arg = ~"[A-Za-z0-9]+"
        gate_name = id

        meas_block = (meas / multimeas)+
        multimeas = "{" meas ("|" meas)* "}"
        meas = ws* "measure " arg nl
        ws = " "+
        nl = (comment / " " / "\n" / "\r")*
        comment = "#" ~".*"
        text = (id / "|" / " " / "\t")*
        id = ~"[A-Za-z0-9_]+"
        circuit = nl* ~".[A-Za-z0-9_-]+" nl
        numarg = ~"[0-9]"+
        """)


sgl_qubit_gate_map = {
    "i": None,
    "mry90": functools.partial(ct.RotateY, angle=-np.pi / 2),
    "rym90": functools.partial(ct.RotateY, angle=-np.pi / 2),
    "my90": functools.partial(ct.RotateY, angle=-np.pi / 2),
    "mY90": functools.partial(ct.RotateY, angle=-np.pi / 2),
    "ry90": functools.partial(ct.RotateY, angle=np.pi / 2),
    "y90": functools.partial(ct.RotateY, angle=np.pi / 2),
    "ry180": functools.partial(ct.RotateY, angle=np.pi),
    "y180": functools.partial(ct.RotateY, angle=np.pi),
    "y": functools.partial(ct.RotateY, angle=np.pi),
    "mrx90": functools.partial(ct.RotateX, angle=-np.pi / 2),
    "mx90": functools.partial(ct.RotateX, angle=-np.pi / 2),
    "rx90": functools.partial(ct.RotateX, angle=np.pi / 2),
    "x90": functools.partial(ct.RotateX, angle=np.pi / 2),
    "rx180": functools.partial(ct.RotateX, angle=np.pi),
    "x180": functools.partial(ct.RotateX, angle=np.pi),
    "x": functools.partial(ct.RotateX, angle=np.pi),
    "prepz": None
}

dbl_qubit_gate_map = {
    "cz": ct.CPhase,
    "fl_cw_01": ct.CPhase
}


def dropnil(lst):
    return [a for a in lst if a is not None]


class QASMParser(parsimonious.NodeVisitor):
    """
    Class to generate circuits from a qasm file

    qubit_parameters is a dictionary defining the qubit properties:
        qubit_parameters = {qubit_name: qubit_pars, ...}

    where

        qubit_pars.keys() == ['t1', 't2', 'frac_1_0', 'frac_1_1']

    dt gives gate timings:
        dt = (single_qubit_gate_time, two_qubit_gate_time)
    """

    def __init__(self, qubit_parameters, timegrid=20,
                 gate_1_step=1, gate_2_step=5, toposort=True):

        self.grammar = qasm_grammar
        self.lines = []
        self.qubit_names = []

        self.timestep = 0
        self.circuits = []
        self.gates = []
        self.qubit_parameters = qubit_parameters
        self.timestep_increment_sgl = gate_1_step * timegrid
        self.timestep_increment_dbl = gate_2_step * timegrid
        self.timestep_increment = timegrid
        self.toposort = toposort

    def visit_qubit_spec(self, node, children):
        try:
            num_qubits = int(children[1])
            for i in range(num_qubits):
                self.qubit_names.append("q" + str(i))
        except ValueError as e:
            raise Exception(
                "Argument must be a number at 'qubits '" +
                children[1])

        # populate the qubit parameters from a given default, if necessary
        for qbn in self.qubit_names:
            if qbn not in self.qubit_parameters:
                self.qubit_parameters[qbn] = self.qubit_parameters['default']

    def visit_initall_named(self, node, children):
        return children[1]

    def visit_initall_fake(self, node, children):
        return "unnamed circuit"

    def visit_initall(self, node, children):
        circuit_name = children[0]
        self.timestep = 0
        self.current_circuit = ct.Circuit(circuit_name)
        for qb in self.qubit_names:
            t1 = self.qubit_parameters[qb]['T1']
            t2 = self.qubit_parameters[qb]['T2']
            self.current_circuit.add_qubit(qb, t1=t1, t2=t2)

        self.to_be_measured = set()

    def visit_gatelist(self, node, children):
        self.timestep += self.timestep_increment
        self.timestep_increment = min(
            self.timestep_increment_dbl,
            self.timestep_increment_sgl)

    def visit_wait_gate(self, node, children):
        time = int(children[-1])
        self.timestep += time

    def visit_single_qubit_gate(self, node, children):
        gate_name, arg = dropnil(children)

        dt = self.timestep_increment_sgl

        gate_factory = sgl_qubit_gate_map[gate_name.lower()]
        if gate_factory is not None:
            gate = gate_factory(bit=arg, time=self.timestep + dt / 2)
            self.current_circuit.add_gate(gate)

        self.timestep_increment = max(dt, self.timestep_increment)

    def visit_two_qubit_gate(self, node, children):
        gate_name, arg1, arg2 = dropnil(children)

        dt = self.timestep_increment_dbl
        gate_factory = dbl_qubit_gate_map[gate_name.lower()]
        gate = gate_factory(bit0=arg1, bit1=arg2, time=self.timestep + dt / 2)
        self.current_circuit.add_gate(gate)
        self.timestep_increment = max(dt, self.timestep_increment)

    def visit_arg(self, node, children):
        if node.text not in self.qubit_names:
            raise RuntimeError("Qubit '" + node.text + "' undefined")
        return node.text

    def visit_id(self, node, children):
        return node.text

    def visit_meas(self, node, children):
        self.to_be_measured.add(children[2])

    def visit_meas_block(self, node, children):

        to_be_measured = self.to_be_measured

        actually_used_qubits = set(to_be_measured)
        for g in self.current_circuit.gates:
            for qb in g.involved_qubits:
                actually_used_qubits.add(qb)

        self.current_circuit.qubits = [
            qb for qb in self.current_circuit.qubits
            if qb.name in actually_used_qubits]

        for b in self.qubit_names:
            p_exc = self.qubit_parameters[b]['frac1_0']
            p_dec = 1 - self.qubit_parameters[b]['frac1_1']
            ro_gate = ct.ButterflyGate(b, p_exc=p_exc, p_dec=p_dec,
                                       time=self.timestep)
            self.current_circuit.add_gate(ro_gate)
        self.current_circuit.add_waiting_gates(tmin=0, tmax=self.timestep)
        self.current_circuit.order(toposort=self.toposort)

        self.circuits.append(self.current_circuit)

    def visit_numarg(self, node, children):
        return int(node.text)

    def generic_visit(self, node, children):
        pass
