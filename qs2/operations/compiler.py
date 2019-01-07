from ..bases import general
from .processes import TracePreservingProcess


class _ProcessBlock:
    def __init__(self, process, *, in_bases=None, out_bases=None):
        """A process block is an internal structure used by the compiler. Each block ultimately implements a process. However in addition a block is aware of it's own bases and indicies, which are then used for the truncation of the processes by the compiler, sparsity analysis and optimization.
        Parameters
        ----------
        in_bases : tuple, optional
            The in bases for the operator (the default is None, which corresponds to unknown bases)
        out_bases : tuple, optional
            The out bases for the operator (the default is None, which corresponds to an unnown bases)

        """
        self.process = process
        self._in_bases = in_bases
        self._out_bases = out_bases

    def __repr__(self):
        raise NotImplementedError


class Compiler:
    """The compiler is a link between the circuit (gates) and the processes. The compiler serves to truncate the processes of the circuit to blocks, while correctly handling specific processes (measurements, resets, etc). The compiler blocks are a wrapper around a process and provide information about the indicies of the process and bases.
    The compiler can then analyze the sparsity of the process operators in order to perform bases optimization for the calculation. These bases are then returned for the processes to be prepared and for the state to use.

    """

    def __init__(self, *operations):
        self.process_inds = [operation[0] for operation in operations]
        self.processes = [operation[1] for operation in operations]

        temp_dims = {}
        for process, inds in zip(self.processes, self.process_inds):
            if isinstance(process, TracePreservingProcess):
                for ind, dim in zip(inds, process.operator.dim_hilbert):
                    if ind not in temp_dims:
                        temp_dims[ind] = dim
                    else:
                        if temp_dims[ind] != dim:
                            raise ValueError('Hilbert dim mismatch')

        self.unique_inds_dims = dict(sorted(temp_dims.items()))

        self.blocks = None

    def create_blocks(self):
        raise NotImplementedError

    def get_optimal_bases(self, initial_bases=None):
        if initial_bases is not None:
            if len(initial_bases) != len(self.unique_inds_dims):
                raise ValueError("Provide a basis for all")
        else:
            initial_bases = tuple(general(dim)
                                  for dim in self.unique_inds_dims.values())
        raise NotImplementedError

    def prepare_blocks(self):
        raise NotImplementedError
