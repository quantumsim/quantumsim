from ..bases import general
from .operation import Transformation, Projection
from .operation import join


class _ProcessBlock:
    """A process block is an internal structure used by the compiler.

    Each block ultimately implements a process. However in addition a block is
    aware of it's own bases and indicies, which are then used for the truncation
    of the processes by the compiler, sparsity analysis and optimization.

    Parameters
    ----------
    in_bases : tuple, optional
        The in bases for the operator (the default is None, which corresponds
        to unknown bases)
    out_bases : tuple, optional
        The out bases for the operator (the default is None, which corresponds
        to an unnown bases)
    """
    def __init__(self, operation, indices, bases_in=None, bases_out=None):
        self.operation = operation
        self.indices = indices
        self.bases_in = bases_in
        self.bases_out = bases_out

    def prepare(self):
        if self.bases_in is None:
            raise ValueError('In bases must be specified')
        if self.bases_out is None:
            raise ValueError('In bases must be specified')

        self.operation.prepare(self.bases_in, self.bases_out)


class _ExplandableBlock:
    """The expandable block is a list of processes, which can be expanded or
    merged with other such blocks. This block is not ready for execution until
    converted to a process block by the compiler.
    """
    def __init__(self, indices):
        self.operations = []
        self.indices = set(indices)

    @property
    def num_bits(self):
        return len(self.indices)

    def merge(self, expandable_block):
        self.operations += expandable_block.processes
        self.indices = self.indices.union(expandable_block.inds)

    def add(self, process, indices):
        self.operations.append(process.at(indices))
        self.indices.update(indices)

    def finalize(self):
        joined_process = join(*self.operations)
        return _ProcessBlock(joined_process, self.indices)


class Compiler:
    """The compiler is a link between the circuit (gates) and the processes.

    The compiler serves to truncate the processes of the circuit to blocks,
    while correctly handling specific processes (measurements, resets, etc).
    The compiler blocks are a wrapper around a process and provide information
    about the indicies of the process and bases.

    The compiler can then analyze the sparsity of the process operators in order
    to perform bases optimization for the calculation. These bases are then
    returned for the processes to be prepared and for the state to use.
    """
    def __init__(self):
        self.proc_blocks = None
        self._blocks = {}

    def create_blocks(self, *operations):
        self.proc_blocks = []
        for process, proc_inds in zip(operations):
            if not isinstance(process, Transformation):
                # Meaning it's either a measurement, initialization or reset.
                # These go in their own blocks
                for proc_ind in proc_inds:
                    if proc_ind in self._blocks:
                        cur_block = self._blocks[proc_ind]
                        self.proc_blocks.append(cur_block.finalize())
                        self._delete_block(proc_ind)
                proc_block = _ProcessBlock(process, proc_inds)
                self.proc_blocks.append(proc_block)
            else:
                self._add_to_blocks(process, proc_inds)

        for block in self._blocks:
            self.proc_blocks.append(block.finalize())

    def get_optimal_bases(self, initial_bases=None):
        if self.proc_blocks is None:
            raise ValueError('Compile operations first')

        last_out_bases = {}

        def get_in_bases(ind, dim):
            if ind in last_out_bases:
                return last_out_bases[ind]
            elif initial_bases is not None:
                return initial_bases[ind]
            return general(dim)

        for block in self.proc_blocks:
            process = block.process
            operator = process.operator
            dims = operator.dim_hilbert

            in_bases = (get_in_bases(ind, dim)
                        for ind, dim in zip(block.inds, dims))
            block.in_bases = in_bases

            if isinstance(process, Projection):
                for ind, basis in zip(block.inds, in_bases):
                    last_out_bases[ind] = basis.computational_subbasis()
            elif isinstance(process, Transformation):
                full_bases = (basis.superbasis() for basis in in_bases)

                # TODO: Add sparsity analysis here to get the real out_basis
                block.out_bases = full_bases
                block.prepare()
                for ind, basis in zip(block.inds, full_bases):
                    last_out_bases[ind] = basis

    def _add_to_blocks(self, process, proc_inds):
        for ind in proc_inds:
            self._register(ind)

        first_block = self._blocks[proc_inds[0]]

        if all(self._blocks[ind] == first_block for ind in proc_inds):
            first_block.add(process, proc_inds)
        else:
            expanded_block = _ExplandableBlock(proc_inds)
            for proc_ind in proc_inds:
                if proc_ind in self._blocks:
                    cur_block = self._blocks[proc_ind]
                    if all(ind in proc_inds for ind in cur_block.inds):
                        expanded_block.merge(cur_block)
                    else:
                        self.proc_blocks.append(cur_block.finalize())

                    self._delete_block(proc_ind)
                    self._blocks[proc_ind] = expanded_block

            expanded_block.add(process, proc_inds)

    def _delete_block(self, ind):
        block_inds = self._blocks[ind].inds
        for block_ind in block_inds:
            del self._blocks[block_ind]

    def _register(self, ind):
        if ind not in self._blocks:
            self._blocks[ind] = _ExplandableBlock(ind)
