# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2018 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

"""Compilers for multiple PTM elements."""

from .pauli import TwoPTMProduct


class CompilerBlock:
    def __init__(
            self,
            bits,
            op,
            index=None,
            bitmap=None,
            in_basis=None,
            out_basis=None):
        self.bits = bits
        self.op = op
        self.bitmap = bitmap
        self.in_basis = in_basis
        self.out_basis = out_basis
        self.ptm = None

    def __repr__(self):
        if self.op == "measure" or self.op == "getdiag":
            opstring = self.op
        else:
            opstring = "ptm"

        bitsstring = ",".join(self.bits)

        basis_string = ""
        if self.in_basis:
            basis_string += "in:" + \
                            "|".join(" ".join(b.basisvector_names) for b in
                                     self.in_basis)
        if self.out_basis:
            basis_string += " out:" + \
                            "|".join(" ".join(b.basisvector_names) for b in
                                     self.out_basis)

        return "<{}: {} on {} {}>".format(
            self.__class__.__name__, opstring, bitsstring, basis_string)


class TwoPTMCompiler:
    def __init__(self, operations, initial_bases=None):
        """Precompiles and optimizes a set of PTM applications for calculation.
        Compilation includes:

        - Contracting single-ptms into adjacent two-qubit ptms
        - choosing basis that facilitate partial tracing:
            - traced qubits in gellmann basis
            - not-traced qubits in general basis
        - examining sparsity and using truncated bases if applicable

        Parameters
        ----------
        operations : list of tuples
            Possible operations are:

            - `([bit], PTM)` for single PTM applications.
            - `([bit0, bit1], TwoPTM)` for two-qubit ptm applications.
            - `([bits], "measure")` for requesting partial trace of all but
              `bits`.

            bit names can be anything hashable.

        initial_bases : None or dict
            If `initial_basis` is None, the initial state is assumed fully
            separable.  Otherwise, its a dict, mapping bits to bases.
        """

        self.operations = operations

        self.bits = set()

        for bs, op in self.operations:
            for b in bs:
                self.bits.add(b)

        self.initial_bases = initial_bases
        if self.initial_bases is None:
            self.initial_bases = []

        self.blocks = None
        self.compiled_blocks = None

    def contract_to_two_ptms(self):

        ctr = 0

        blocks = []
        active_block_idx = {}
        bits_in_block = {}

        for bs, op in self.operations:

            for b in bs:
                if b not in active_block_idx:
                    new_bl = []
                    active_block_idx[b] = len(blocks)
                    blocks.append(new_bl)
                    bits_in_block[b] = [b]

            ctr += 1
            if op == "measure" or op == "getdiag":
                # measurement goes in single block
                measure_block = [(bs, op, ctr)]
                blocks.append(measure_block)
                for b in bs:
                    del active_block_idx[b]
            elif len(bs) == 1:
                blocks[active_block_idx[bs[0]]].append((bs, op, ctr))
            elif len(bs) == 2:
                b0, b1 = bs
                bl_i0, bl_i1 = active_block_idx[b0], active_block_idx[b1]
                if bl_i0 == bl_i1:
                    # qubits are in same block
                    blocks[bl_i0].append((bs, op, ctr))
                else:
                    if len(bits_in_block[b0]) == 2:
                        # b0 was in block with someone else, new block for b0
                        new_bl0 = []
                        bl_i0 = active_block_idx[b0] = len(blocks)
                        blocks.append(new_bl0)
                        bits_in_block[b0] = [b0]
                    if len(bits_in_block[b1]) == 2:
                        # b1 was in block with someone else, new block for b1
                        new_bl1 = []
                        bl_i1 = active_block_idx[b1] = len(blocks)
                        blocks.append(new_bl1)
                        bits_in_block[b1] = [b1]
                    # now we can be sure that both qb are in single block. We
                    # combine them:
                    bits_in_block[b0] = [b0, b1]
                    bits_in_block[b1] = [b0, b1]
                    # append earlier block to later block
                    if bl_i0 < bl_i1:
                        blocks[bl_i1].extend(blocks[bl_i0])
                        blocks[bl_i0] = []
                        blocks[bl_i1].append((bs, op, ctr))
                        active_block_idx[b0] = active_block_idx[b1]
                    else:
                        blocks[bl_i0].extend(blocks[bl_i1])
                        blocks[bl_i1] = []
                        blocks[bl_i0].append((bs, op, ctr))
                        active_block_idx[b1] = active_block_idx[b0]

        # active blocks move to end
        for b, bli in active_block_idx.items():
            bl = blocks[bli]
            blocks[bli] = []
            blocks.append(bl)

        self.blocks = [bl for bl in blocks if bl]
        self.abi = active_block_idx

    def make_block_ptms(self):
        self.compiled_blocks = []

        for bl in self.blocks:
            if bl[0][1] == "measure" or bl[0][1] == "getdiag":
                mbl = CompilerBlock(bits=bl[0][0], op=bl[0][1])
                self.compiled_blocks.append(mbl)
            else:
                product = TwoPTMProduct([])
                bit_map = {}
                for bits, op, i in bl:
                    if len(bits) == 1:
                        b, = bits
                        if b not in bit_map:
                            bit_map[b] = len(bit_map)
                        product.elements.append(([bit_map[b]], op))
                    if len(bits) == 2:
                        b0, b1 = bits
                        if b0 not in bit_map:
                            bit_map[b0] = len(bit_map)
                        if b1 not in bit_map:
                            bit_map[b1] = len(bit_map)
                        product.elements.append(
                            ([bit_map[b0], bit_map[b1]], op))

                # order the bitlist
                bits = list(bit_map.keys())
                if bit_map[bits[0]] == 1:
                    bits = list(reversed(bits))

                ptm_block = CompilerBlock(
                    bits=bits,
                    bitmap=bit_map,
                    op=product)
                self.compiled_blocks.append(ptm_block)

    def basis_choice(self, tol=1e-16):

        # for each block
        #   find previous blocks for involved qubits:
        #   If none: set in_basis from_initial basis
        #   else: Set in_basis from that block
        #
        #   find out_bases from sparsity analysis

        for i, cb in enumerate(self.compiled_blocks):
            cb.in_basis = []
            cb.out_basis = []
            for bit in cb.bits:
                previous = [cb2 for j, cb2 in enumerate(
                    self.compiled_blocks[:i]) if bit in cb2.bits]
                if len(previous) == 0:
                    # we are the first, use init_basis
                    if bit in self.initial_bases:
                        in_basis = self.initial_bases[bit]
                    else:
                        in_basis = GeneralBasis(cb.op.dim_hilbert)
                else:
                    previous = previous[-1]
                    bit_idx_in_previous = previous.bits.index(bit)
                    in_basis = previous.out_basis[bit_idx_in_previous]

                cb.in_basis.append(in_basis)

            # find out_basis and ptm matrix
            if cb.op == "measure":
                cb.out_basis = [b.get_classical_subbasis()
                                for b in cb.in_basis]
            elif cb.op == "getdiag":
                cb.out_basis = cb.in_basis
            elif len(cb.in_basis) == 2:
                full_basis = [b.get_superbasis() for b in cb.in_basis]
                full_mat = cb.op.get_matrix(cb.in_basis, full_basis)
                sparse_out_0 = np.nonzero(np.einsum(
                    "abcd -> a", full_mat ** 2, optimize=True) > tol)[0]
                sparse_out_1 = np.nonzero(np.einsum(
                    "abcd -> b", full_mat ** 2, optimize=True) > tol)[0]
                cb.out_basis = [
                    full_basis[0].get_subbasis(sparse_out_0),
                    full_basis[1].get_subbasis(sparse_out_1)
                ]
                cb.ptm = cb.op.get_matrix(cb.in_basis, cb.out_basis)

    def run(self):
        if self.blocks is None:
            self.make_block_ptms()
        if self.compiled_blocks is None:
            self.contract_to_two_ptms()
            self.basis_choice()
