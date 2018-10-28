# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2018 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

"""General classes for gates.
A quantumsim gate is a method for generating and containing 
quantumsim ptms that can read a quantumsim setup file,
accept input from the user, update itself if required, 
and possibly perform some precompilation when requested
prior to being loaded into a compiler."""

from qs2.operators import (
    TwoPTMProduct,
    AdjustablePTM,
    DummyPTM)


class Gate:
    """The fundamental gate class.

    Params
    ------
    setup : quantumsim.setup.Setup object
        setup file for the experiment.
    qubits : list
        list of qubits that this gate acts on.
    compiled_flag : boolean
        whether or not the gate has been precompiled.
    ptms : list of quantumsim.transformations.PTM objects
        the Pauli transfer matrices that describe the gate,
        in time-order of operation (i.e. not product order).
    sargs : dict
        required arguments from setup.
    """

    def __init__(self, qubits, setup):
        self.setup = setup
        self.qubits = qubits
        self.sargs = {}
        self.ptms = []
        self.compiled_flag = False
        self.init_ptms()

    def init_ptms(self):
        """Initialises the PTMs as required.
        """
        raise NotImplementedError

    def get_ptms(self):
        """ Returns the set of ptms safely for use elsewhere.
        """
        if self.compiled_flag is False:
            self.compile()
        return list(self.ptms)

    def get_qubits(self):
        """ Returns the list of qubits safely for use elsewhere.
        """
        return list(self.qubits)

    def compile(self):
        """ Compiles the list of ptms.
        """
        raise NotImplementedError

    def requires_from_setup(self):
        """ Gets dictionary of parameters required from setup.
        Currently setting this to the actual dictionary under the
        assumption that something might want to fill this in.
        """
        return sargs


class AdjustableGate(Gate):
    """A gate class with some free parameters, called angles, 
    to be adjusted by the user either before or after compilation.

    WIP - not really sure how best to parametrize this yet.
    Namely, I'm not sure how much needs to be added here to
    allow generic adjustment. If this stays as it is it could be
    easily added to the Gate class, but I feel like we want
    to split this up somehow.

    Params
    ------
    uargs : dict
        required arguments from the user.
    """
    def __init__(self,  qubits, setup, **uargs):
        self.uargs = uargs
        super().__init__(qubits, setup)

    def requires_from_user(self):
        """ Gets dictionary of parameters required from the user.
        Currently setting this to the actual dictionary under the
        assumption that something might want to fill this in.
        """
        return uargs


class RotationGate(AdjustableGate):
    """ A rotation gate is an adjustable gate with
    a single parameter, namely that of an angle.

    Params
    ------
    ptm_function : function
        The function to generate the gate ptm.
    angle: the angle of rotation
    """

    def __init__(self, ptm_function, angle=None, **kwargs):
        self.ptm_function = ptm_function
        super().__init__(angle=angle, **kwargs)

    def init_ptms(self):
        if self.uargs['angle'] is not None:
            self.ptms = [self.ptm_function(self.uargs['angle'])]
            self.compiled_flag = True

    def compile(self):
        if self.uargs['angle'] is None:
            self.ptms = [AdjustablePTM(self.ptm_function)]
        else:
            self.ptms = [self.ptm_function(self.uargs['angle'])]
        self.compiled_flag = True

    def adjust(self, angle):
        self.uargs['angle'] = angle


class ContainerGate(Gate):
    """A generic gate class to contain other, smaller gates,
    and eventually subsume them (to form one glorious whole).

    -- 'But Rimmer, you already are a glorious hole' - D. Lister.

    WIP - not sure if we need to add more to this.

    Params
    ------
    gates : list of quantumsim.transformations.Gate objects
        the gates to be eventually subsumed
    """
    def __init__(self, gates=None, **kwargs):
        if gates is None:
            self.gates = []
        else:
            self.gates = gates
        super().__init__(**kwargs)


class ProductContainer(ContainerGate):
    """An extension of the ContainerGate class that does the
    simplest type of combination, namely combining multiple
    ptms in a TwoPTMProduct class.

    WIP - currently doesn't autocompile - perhaps we want this?
    """

    def init_ptms(self):
        pass

    def compile(self):
        """Combines PTMs in much the same way as the PTM class.
        """
        self.ptms = []
        current_PTM = TwoPTMProduct([])
        bit_map = {}

        for gate in self.gates:
            ops = gate.get_ptms()
            for op in ops:
                # If the ptm cannot be yet generated, then the gate
                # should produce a DummyPTM of some type that must
                # get updated later, in which case we break and add
                # this to the list.
                if isinstance(ops, DummyPTM):
                    if len(current_PTM.elements) > 0:
                        self.ptms.append(current_PTM)
                    self.ptms.append(ops)
                    current_PTM = TwoPTMProduct([])
                    bit_map = {}
                    continue

                bits = gate.get_qubits()
                if len(bits) == 1:
                    b, = bits
                    if b not in bit_map:
                        bit_map[b] = len(bit_map)
                    current_PTM.elements.append(([bit_map[b]], op))
                if len(bits) == 2:
                    b0, b1 = bits
                    if b0 not in bit_map:
                        bit_map[b0] = len(bit_map)
                    if b1 not in bit_map:
                        bit_map[b1] = len(bit_map)
                    current_PTM.elements.append(
                        ([bit_map[b0], bit_map[b1]], op))

        if len(current_PTM.elements) > 0:
            self.ptms.append(current_PTM)


class TimedGate(Gate):
    """A gate with a notion of time.

    WIP - not sure whether we need to add more to this.

    Params
    ------
    start_time : float
        start time of the gate
    end_time : float
        end time of the gate
    """

    def __init__(self, start_time, end_time, **kwargs):
        start_time = start_time
        end_time = end_time
        super().__init__(**kwargs)


class DecayContainer(TimedGate, ContainerGate):
    """A container that holds one or more gates inside and combines them
    whilst wrapping them in decay.

    -- 'Muhahaha' - some evil villain, somewhere.

    WIP - need to finish the add decay gates functionality

    Params
    ------
    can_add_decay : Whether this gate can add decay gates yet.
    """

    def __init__(self, can_add_decay=False, **kwargs):
        self.can_add_decay = can_add_decay
        super().__init__(**kwargs)

    def init_ptms(self):
        if self.can_add_decay:
            self.insert_decay_gates()

    def insert_decay_gates(self):

        # WIP - basically we want to copy add_waiting_gates here.
        raise NotImplementedError
        ts = self.start_time
        for gate in self.gates:
            if isinstance(gate, TimedGate):
                tf = gate.start_time
                next_ts = gate.end_time
            else:
                try:
                    duration = self.setup.get_gate_param(gate,'duration')
                except:
                    raise ValueError(
                        'I cant figure out how long this gate lasts')
                tf = ts + duration/2
                next_ts = tf + duration


