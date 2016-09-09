
import numpy as np

from collections import defaultdict

from . import circuit

import copy


def get_dephasing(tstart, tend, Dt, chi, kappa, alpha0):
    '''
    Calculates the photon-induced dephasing over a period of
    time, and returns a valuefor the decay constant lambda

    Input:
    @tstart : time between the start *of the coherent phase* and
            the start of the decay period.
    @tend : time between the start *of the coherent phase* and
            the end of the decay period
    @Dt   : time between the end *of the measurement pulse* and
            the start *of the coherent phase*. This should be
            about t_rest + gate_time/2.
    @chi  : energy gap between photon states
    @kappa : photon decay rate
    @alpha0 : number of photons at the end of the measurement pulse.

    Output:
    @lamda : The calculated decay rate
    '''

    # Photon number at the start of the coherent phase
    # Multiplied by other constants
    two_chi_alpha_Dt = 2 * chi * alpha0 * np.exp(-kappa * Dt)

    # Top half of the integral
    int_term_top = -np.exp(-kappa * tend) / (4 * chi**2 + kappa**2) * \
        (kappa * np.sin(2 * chi * tend) + 2 * chi * np.cos(2 * chi * tend))
    int_term_bot = -np.exp(-kappa * tstart) / (4 * chi**2 + kappa**2) * \
        (kappa * np.sin(2 * chi * tstart) + 2 * chi * np.cos(2 * chi * tstart))
    # print(int_term_top, int_term_bot, two_chi_alpha_Dt)

    lamda = np.exp(-two_chi_alpha_Dt * (int_term_top - int_term_bot))
    return lamda


def add_waiting_gates_photons(c, tmin, tmax, chi, kappa, alpha0):
    """Add AmpPhDamping gates to all qubits that involve
    a measurement in the Circuit c.

    These model the additional dephasing due to lingering photons
    from a import measurement.

    Assume that the circuit is periodic;
    the time tmin for one cycle is identified with tmax of the previous cycle.
    """

    times = [g.time for g in c.gates]
    assert min(times) >= tmin
    assert max(times) <= tmax

    gates_per_qubit = defaultdict(list)

    for g in c.gates:
        for qb in g.involved_qubits:
            b = [b for b in c.qubits if b.name == qb][0]
            gates_per_qubit[b].append(g)

    for qb in gates_per_qubit:
        gs = gates_per_qubit[qb]
        gs.sort(key=lambda g: g.time)

        # at what times to measurements occur
        meas_times = [g.time for g in gs if g.is_measurement]
        # if any, include the previous round
        if meas_times:
            meas_times = [meas_times[-1] - tmax + tmin] + meas_times

        # at what times to coherent pulses occur
        pi2_times = [
            g.time for g in gs if isinstance(
                g,
                circuit.Hadamard) or (
                isinstance(
                    g,
                    circuit.RotateY) and 1 <= abs(
                    g.angle) <= 2)]
        if pi2_times:
            pi2_times = [pi2_times[-1] - tmax + tmin] + pi2_times

        # put a copy of the last gate of the previous round to the front
        virtual_gate = copy.copy(gs[-1])
        virtual_gate.time += tmin - tmax

        gate_pairs = zip([virtual_gate] + gs[:-1], gs)

        for g1, g2 in gate_pairs:

            decay_gate = circuit.AmpPhDamp(
                bit=qb.name,
                time=(g2.time + g1.time) / 2,
                duration=g2.time - g1.time,
                t1=qb.t1,
                t2=qb.t2)

            if meas_times and pi2_times:
                last_meas = max(t for t in meas_times if t <= g1.time)
                pi2s = [t for t in pi2_times if last_meas <= t <= g1.time]

                if len(pi2s) == 1:
                    # we are in the coherent phase
                    dt = pi2s[0] - last_meas

                    photon_lamda = get_dephasing(
                        g1.time, g2.time, dt, chi, kappa, alpha0)

                    ptm_patch = circuit.ptm.amp_ph_damping_ptm(0, 1-photon_lamda)

                    decay_gate.ptm = np.dot(decay_gate.ptm, ptm_patch)

                    decay_gate.annotation = "*"

            c.add_gate(decay_gate)
