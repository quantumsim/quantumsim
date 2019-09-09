import numpy as np
from itertools import product


def plot(state, ax=None, truncate_levels=None):
    """

    Parameters
    ----------
    state : qs.State
        State to display
    ax : matplotlib.axes.Axes or None
        Axes to plot onto. If None, new figure is created and returned.
    truncate_levels : None or int
        If not None, all the states higher than provided are discarded and a
        identity is added to the state instead, so that total trace is
        preserved. This should emulate behaviour of tomografy in the presence
        of leakage.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
    """
    import matplotlib.pyplot as plt
    # TODO: get rid of qutip, no sence to carry such a huge dependency for
    #  such a small task
    import qutip

    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    else:
        fig = None

    n_qubits = len(state.qubits)
    pv = state.pauli_vector
    rho = pv.to_dm()
    rho /= np.trace(rho)

    if truncate_levels is not None:
        # Tomo emulation: truncate leaked states and add
        rho2 = (rho.reshape(state.pauli_vector.dim_hilbert * 2)
                [(slice(0, truncate_levels),) * (2 * n_qubits)]
                .reshape(truncate_levels ** n_qubits,
                         truncate_levels ** n_qubits))
        trace = np.trace(rho2)
        rho2 += ((1 - trace) * np.identity(2**n_qubits) *
                 truncate_levels ** -n_qubits)
        assert np.allclose(np.trace(rho2), 1)
        dim = [truncate_levels] * n_qubits
    else:
        rho2 = rho
        dim = pv.dim_hilbert

    def tuple_to_string(tup):
        return "".join(str(x) for x in tup)

    labels = [tuple_to_string(x) for x in product(*(range(d) for d in dim))]
    rho_q = qutip.Qobj(rho2, dims=[dim, dim])
    qutip.matrix_histogram_complex(rho_q, ax=ax, xlabels=labels, ylabels=labels)

    return fig
