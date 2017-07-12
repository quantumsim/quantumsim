import quantumsim.qasm as qasm

qubit_test_pars = {
    'T1': 30000,
    'T2': 30000,
    'frac1_0': 0.01,
    'frac1_1': 0.99}


class TestQASMParser:

    def test_simple(self):
        simple = """qubit QR
            qubit QL

            init_all
            Y90 QR | Y90 QL
            CZ QR QL
            mY90 QL
            I QL | I QR
            RO QR
        """

        parser = qasm.QASMParser(
            qubit_parameters={
                'QR': qubit_test_pars,
                'QL': qubit_test_pars},
            dt=(
                40,
                200))

        parser.parse(simple)

        assert len(parser.circuits) == 1
        assert len(parser.circuits[0].gates) == 11
        assert len(parser.circuits[0].qubits) == 2


    def test_initial_newlines(self):
        simple = """
            qubit QR
            qubit QL

            init_all
            Y90 QR | Y90 QL
            CZ QR QL
            mY90 QL
            I QL | I QR
            RO QR
        """

        parser = qasm.QASMParser(
            qubit_parameters={
                'QR': qubit_test_pars,
                'QL': qubit_test_pars},
            dt=(
                40,
                200))

        parser.parse(simple)

        assert len(parser.circuits) == 1
        assert len(parser.circuits[0].gates) == 11
        assert len(parser.circuits[0].qubits) == 2

    def test_comments(self):
        simple = """
            # this is where it starts
            qubit QR #qubit 1
            qubit QL # qubit 2

            # first circuit

            init_all
            Y90 QR | Y90 QL
            CZ QR QL
            # ignore this line
            mY90 QL #a comment
            I QL | I QR
            RO QR

            # end
        """

        parser = qasm.QASMParser(
            qubit_parameters={
                'QR': qubit_test_pars,
                'QL': qubit_test_pars},
            dt=(
                40,
                200))

        parser.parse(simple)

        assert len(parser.circuits) == 1
        assert len(parser.circuits[0].gates) == 11
        assert len(parser.circuits[0].qubits) == 2
