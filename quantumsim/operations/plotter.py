from itertools import product


def plot(operation, bases_in=None, bases_out=None, ax=None, truncate_levels=None, colorbar=True):
    """
    Parameters
    ----------
    operation : qs.Operation
        Operation to display
    ax : matplotlib.axes.Axes or None
        Axes to plot onto. If None, new figure is created and returned.
    truncate_levels : None or int
        If not None, all the states higher than provided are discarded and a
        identity is added to the state instead, so that total trace is
        preserved. This should emulate behavior of tomography in the presence
        of leakage.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
    """
    import matplotlib.pyplot as plt
    from matplotlib import colorbar

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    dim = operation.dim_hilbert
    num_pauli_elems = (dim**2) * operation.num_qubits

    _bases_in = bases_in or operation.bases_in
    _bases_out = bases_out or operation.bases_out

    ptm = operation.ptm(_bases_in, _bases_out)

    if truncate_levels is not None:
        raise NotImplementedError

    def tuple_to_string(tup):
        pauli_element = ''.join(str(x) for x in tup)
        return r'$%s$' % pauli_element

    _bases_in_labels = (basis.labels for basis in _bases_in)
    x_labels = [tuple_to_string(x) for x in product(*_bases_in_labels)]

    _bases_out_labels = (basis.labels for basis in _bases_out)
    y_labels = [tuple_to_string(x) for x in product(*_bases_out_labels)]

    img = ax.imshow(ptm, cmap='bwr')

    ax.set_xticks(range(num_pauli_elems))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Input basis')

    ax.set_yticks(range(num_pauli_elems))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Output basis')

    if colorbar:
        cax, _ = colorbar.make_axes(ax)
        cbar = colorbar.Colorbar(cax, img)
        cbar.set_ticks((-1, 0, 1))
        cbar.ax.set_ylabel('Amplitude', rotation=270)

    return fig
