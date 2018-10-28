# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2018 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import numpy as np

from qs2.compilers import (
    TwoPTMCompiler,
    CompilerBlock)


class TestTwoPTMCompiler:
    def test_init(self):
        operations = [(['q0'],'test_gate')]
        compiler = TwoPTMCompiler(operations)
        assert 'q0' in compiler.bits
        assert len(compiler.initial_bases) == 0
        assert compiler.blocks is None
        assert compiler.compiled_blocks is None
        assert len(compiler.operations) == 1
        assert compiler.operations[0][0][0] == 'q0'
        assert compiler.operations[0][1] == 'test_gate'
