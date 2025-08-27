from .gates import SUM as CX, PHASE as S, Hadamard as H
from .circuits import Circuit
from quaos.core.finite_field_solvers import solve_modular_linear_additive
from quaos.core.paulis import PauliString, PauliSum


def add_phase(xz_pauli_sum: PauliSum, qudit_index: int, qudit_index_2: int, phase_key: str) -> Circuit:
    """
    Uses phase key to alter the phase of a Pauli sum based on the Pauli operator at qudit index 2.

    Acts as the identity on qudit index 1.

    Assumes ``pauli_sum`` has the form:

    .. list-table::
       :header-rows: 1

       * - qudit_index_1
         - qudit_index_2
       * - X
         - *
       * - Z
         - *

    The ``phase_key`` is a string where:
    - ``S`` means "same phase"
    - ``D`` means "different phase"

    The order is IXZY.

    Example:
        ``'SDSD'`` keeps the same phase for all but X and Y Paulis on qudit index 2.
    """

    if xz_pauli_sum.dimensions[qudit_index] != 2 or xz_pauli_sum.dimensions[qudit_index] != 2:
        raise ValueError("Pauli dimensions must be equal to 2")

    if phase_key == 'SSSS':
        C = Circuit(dimensions=[2 for i in range(xz_pauli_sum.n_qudits())])
        return C
    elif phase_key == 'DDSS':
        C = Circuit(dimensions=[2 for i in range(xz_pauli_sum.n_qudits())],
                    gates=[CX(qudit_index, qudit_index_2, 2), S(qudit_index_2, 2), H(qudit_index_2, 2),
                           S(qudit_index_2, 2), CX(qudit_index, qudit_index_2, 2)])
        return C
    elif phase_key == 'DSDS':
        C = Circuit(dimensions=[2 for i in range(xz_pauli_sum.n_qudits())],
                    gates=[CX(qudit_index, qudit_index_2, 2), S(qudit_index_2, 2), CX(qudit_index, qudit_index_2, 2),
                           H(qudit_index_2, 2), CX(qudit_index, qudit_index_2, 2), S(qudit_index, 2)])
        return C
    elif phase_key == 'DSSD':
        C = Circuit(dimensions=[2 for i in range(xz_pauli_sum.n_qudits())],
                    gates=[S(qudit_index_2, 2), CX(qudit_index, qudit_index_2, 2), S(qudit_index_2, 2),
                           H(qudit_index_2, 2), S(qudit_index_2, 2), CX(qudit_index, qudit_index_2, 2)])
        return C
    elif phase_key == 'SDDS':
        C = Circuit(dimensions=[2 for i in range(xz_pauli_sum.n_qudits())],
                    gates=[S(qudit_index_2, 2), H(qudit_index_2, 2), CX(qudit_index_2, qudit_index, 2),
                           S(qudit_index, 2), CX(qudit_index_2, qudit_index, 2), H(qudit_index_2, 2),
                           CX(qudit_index, qudit_index_2, 2), S(qudit_index, 2)])
        return C
    elif phase_key == 'SDSD':
        C = Circuit(dimensions=[2 for i in range(xz_pauli_sum.n_qudits())],
                    gates=[CX(qudit_index_2, qudit_index, 2), S(qudit_index, 2), CX(qudit_index_2, qudit_index, 2),
                           H(qudit_index_2, 2), CX(qudit_index, qudit_index_2, 2), S(qudit_index, 2)])
        return C
    elif phase_key == 'SSDD':
        C = Circuit(dimensions=[2 for i in range(xz_pauli_sum.n_qudits())],
                    gates=[H(qudit_index_2, 2), CX(qudit_index_2, qudit_index, 2), S(qudit_index, 2),
                           CX(qudit_index_2, qudit_index, 2), H(qudit_index_2, 2), CX(qudit_index, qudit_index_2, 2),
                           S(qudit_index, 2)])
        return C
    elif phase_key == 'DDDD':
        C = Circuit(dimensions=[2 for i in range(xz_pauli_sum.n_qudits())],
                    gates=[CX(qudit_index_2, qudit_index, 2), S(qudit_index, 2), CX(qudit_index_2, qudit_index, 2),
                           H(qudit_index_2, 2), CX(qudit_index, qudit_index_2, 2), S(qudit_index, 2),
                           CX(qudit_index, qudit_index_2, 2), S(qudit_index_2, 2), H(qudit_index_2, 2),
                           S(qudit_index_2, 2), CX(qudit_index, qudit_index_2, 2)])
        return C
    else:
        raise ValueError(
            "Invalid phase key. Must be one of 'SSSS', 'DDSS', 'DSDS', 'DSSD', 'SDDS', 'SDSD', 'SSDD', 'DDDD'"
        )


def add_s2(pauli_sum: PauliSum, qudit_index_1: int, qudit_index_2: int) -> Circuit:
    """
    xr1zs1 xr2zs2 -> xr1+s2 zs1+s2  *
    """
    C = Circuit(dimensions=[2 for i in range(pauli_sum.n_qudits())],
                gates=[CX(qudit_index_1, qudit_index_2, 2), H(qudit_index_2, 2), CX(qudit_index_2, qudit_index_1, 2),
                       S(qudit_index_2, 2), H(qudit_index_2, 2)])
    return C


def add_r2(pauli_sum: PauliSum, qudit_index_1: int, qudit_index_2: int) -> Circuit:
    """
    xr1zs1 xr2zs2 -> xr1+r2 zs1+r2  *
    """
    C = Circuit(dimensions=[2 for i in range(pauli_sum.n_qudits())],
                gates=[S(qudit_index_1, 2), CX(qudit_index_2, qudit_index_1, 2), S(qudit_index_1, 2)])
    return C


def add_r2s2(pauli_sum: PauliSum, qudit_index_1: int, qudit_index_2: int) -> Circuit:
    """
    xr1zs1 xr2zs2 -> xr1+r2+s2 zs1+r2+s2  *
    """
    C = Circuit(dimensions=[2 for i in range(pauli_sum.n_qudits())],
                gates=[S(qudit_index_2, 2), CX(qudit_index_1, qudit_index_2, 2), H(qudit_index_2, 2),
                       CX(qudit_index_2, qudit_index_1, 2), S(qudit_index_2, 2), H(qudit_index_2, 2)])
    return C


def _find_first_exp(pauli_sum, pauli_index, target_qubit, exp_type):
    """Finds the first qubit >= target_qubit with nonzero x_exp or z_exp for pauli_index."""
    exp = pauli_sum.x_exp if exp_type == 'x' else pauli_sum.z_exp
    for i in range(target_qubit, pauli_sum.n_qudits()):
        if exp[pauli_index, i]:
            return i
    return None


def _apply_h_and_cx(C, pauli_sum, qubit, target_qubit, direction='forward'):
    """Apply H and CX gates and update pauli_sum."""
    g = H(qubit, 2)
    pauli_sum = g.act(pauli_sum)
    C.add_gate(g)
    if direction == 'forward':
        g = CX(target_qubit, qubit, 2)
    else:
        g = CX(qubit, target_qubit, 2)
    C.add_gate(g)
    pauli_sum = g.act(pauli_sum)
    return C, pauli_sum


def _handle_xz_case(C, pauli_sum, pauli_index_x, pauli_index_z, target_qubit):
    px = None
    if pauli_sum.x_exp[pauli_index_x, target_qubit] == 1 and pauli_sum.z_exp[pauli_index_z, target_qubit] == 1:
        px = pauli_index_x
    elif pauli_sum.z_exp[pauli_index_x, target_qubit] == 1 and pauli_sum.x_exp[pauli_index_z, target_qubit] == 1:
        px = pauli_index_z
    return px


def _handle_x_id_or_x_x(C, pauli_sum, pauli_index_x, pauli_index_z, target_qubit):
    px = None
    z_qubit = _find_first_exp(pauli_sum, pauli_index_z, target_qubit, 'z')
    x_qubit = _find_first_exp(pauli_sum, pauli_index_z, target_qubit, 'x')
    if z_qubit is not None:
        g = CX(target_qubit, z_qubit, 2)
    elif x_qubit is not None:
        C, pauli_sum = _apply_h_and_cx(C, pauli_sum, x_qubit, target_qubit, direction='forward')
        z_qubit = _find_first_exp(pauli_sum, pauli_index_z, target_qubit, 'z')
        g = CX(target_qubit, z_qubit, 2)
    C.add_gate(g)
    pauli_sum = g.act(pauli_sum)
    px = pauli_index_x
    return C, pauli_sum, px


def _handle_z_id_or_z_z(C, pauli_sum, pauli_index_z, target_qubit):
    px = None
    x_qubit = _find_first_exp(pauli_sum, pauli_index_z, target_qubit, 'x')
    z_qubit = _find_first_exp(pauli_sum, pauli_index_z, target_qubit, 'z')
    if x_qubit is not None:
        g = CX(x_qubit, target_qubit, 2)
    elif z_qubit is not None:
        C, pauli_sum = _apply_h_and_cx(C, pauli_sum, z_qubit, target_qubit, direction='backward')
        x_qubit = _find_first_exp(pauli_sum, pauli_index_z, target_qubit, 'x')
        g = CX(x_qubit, target_qubit, 2)
    C.add_gate(g)
    pauli_sum = g.act(pauli_sum)
    px = pauli_index_z
    return C, pauli_sum, px


def _handle_id_z(C, pauli_sum, pauli_index_x, target_qubit):
    px = None
    x_qubit = _find_first_exp(pauli_sum, pauli_index_x, target_qubit, 'x')
    z_qubit = _find_first_exp(pauli_sum, pauli_index_x, target_qubit, 'z')
    if x_qubit is not None:
        g = CX(x_qubit, target_qubit, 2)
    elif z_qubit is not None:
        C, pauli_sum = _apply_h_and_cx(C, pauli_sum, z_qubit, target_qubit, direction='backward')
        x_qubit = _find_first_exp(pauli_sum, pauli_index_x, target_qubit, 'x')
        g = CX(x_qubit, target_qubit, 2)
    C.add_gate(g)
    pauli_sum = g.act(pauli_sum)
    px = pauli_index_x
    return C, pauli_sum, px


def _handle_id_x(C, pauli_sum, pauli_index_x, pauli_index_z, target_qubit):
    px = None
    z_qubit = _find_first_exp(pauli_sum, pauli_index_x, target_qubit, 'z')
    x_qubit = _find_first_exp(pauli_sum, pauli_index_x, target_qubit, 'x')
    if z_qubit is not None:
        g = CX(target_qubit, z_qubit, 2)
    elif x_qubit is not None:
        C, pauli_sum = _apply_h_and_cx(C, pauli_sum, x_qubit, target_qubit, direction='forward')
        z_qubit = _find_first_exp(pauli_sum, pauli_index_x, target_qubit, 'z')
        g = CX(target_qubit, z_qubit, 2)
    C.add_gate(g)
    pauli_sum = g.act(pauli_sum)
    px = pauli_index_z
    return C, pauli_sum, px


def _handle_id_id(C, pauli_sum, pauli_index_x, pauli_index_z, target_qubit):
    px = None
    x_qubit_x = _find_first_exp(pauli_sum, pauli_index_x, target_qubit, 'x')
    z_qubit_x = _find_first_exp(pauli_sum, pauli_index_x, target_qubit, 'z')
    x_qubit_z = _find_first_exp(pauli_sum, pauli_index_z, target_qubit, 'x')
    z_qubit_z = _find_first_exp(pauli_sum, pauli_index_z, target_qubit, 'z')
    if x_qubit_x is not None:
        g = CX(x_qubit_x, target_qubit, 2)
        pauli_sum = g.act(pauli_sum)
        C.add_gate(g)
        if z_qubit_z is not None:
            g = CX(target_qubit, z_qubit_z, 2)
        elif x_qubit_z is not None:
            C, pauli_sum = _apply_h_and_cx(C, pauli_sum, x_qubit_z, target_qubit, direction='forward')
            z_qubit_z = _find_first_exp(pauli_sum, pauli_index_z, target_qubit, 'z')
            g = CX(target_qubit, z_qubit_z, 2)
        C.add_gate(g)
        pauli_sum = g.act(pauli_sum)
        px = pauli_index_x
    elif z_qubit_x is not None:
        g = CX(target_qubit, z_qubit_x, 2)
        pauli_sum = g.act(pauli_sum)
        C.add_gate(g)
        if x_qubit_z is not None:
            g = CX(x_qubit_z, target_qubit, 2)
        elif z_qubit_z is not None:
            C, pauli_sum = _apply_h_and_cx(C, pauli_sum, z_qubit_z, target_qubit, direction='backward')
            x_qubit_z = _find_first_exp(pauli_sum, pauli_index_z, target_qubit, 'x')
            g = CX(x_qubit_z, target_qubit, 2)
        C.add_gate(g)
        pauli_sum = g.act(pauli_sum)
        px = pauli_index_z
    return C, pauli_sum, px


def ensure_zx_components(pauli_sum: PauliSum, pauli_index_x: int,
                         pauli_index_z: int, target_qubit: int) -> tuple[Circuit, PauliSum]:
    """
    Assumes anti-commutation between pauli_index_x and pauli_index_z.
    brings pauli_sum to the form:

    target_qubit
    pauli_index_x |  xr1zs1
    pauli_index_z |  xr2zs2
    where r1 and s2 are always non-zero

    """
    if not pauli_sum[pauli_index_x, target_qubit:].commute(pauli_sum[pauli_index_z, target_qubit:]):
        raise ValueError(("ensure_zx_components requires anti-commutation"
                          " between pauli_index_x and pauli_index_z beyond target_qubit"))

    C = Circuit(dimensions=pauli_sum.dimensions)
    px = _handle_xz_case(C, pauli_sum, pauli_index_x, pauli_index_z, target_qubit)
    if px is not None:
        pass
    elif pauli_sum.x_exp[pauli_index_x, target_qubit] == 1 and pauli_sum.z_exp[pauli_index_z, target_qubit] == 0:
        C, pauli_sum, px = _handle_x_id_or_x_x(C, pauli_sum, pauli_index_x, pauli_index_z, target_qubit)
    elif pauli_sum.z_exp[pauli_index_x, target_qubit] == 1 and pauli_sum.x_exp[pauli_index_z, target_qubit] == 0:
        C, pauli_sum, px = _handle_z_id_or_z_z(C, pauli_sum, pauli_index_z, target_qubit)
    elif pauli_sum.x_exp[pauli_index_x, target_qubit] == 0 and pauli_sum.z_exp[pauli_index_z, target_qubit] == 1:
        C, pauli_sum, px = _handle_id_z(C, pauli_sum, pauli_index_x, target_qubit)
    elif pauli_sum.x_exp[pauli_index_x, target_qubit] == 0 and pauli_sum.x_exp[pauli_index_z, target_qubit] == 1:
        C, pauli_sum, px = _handle_id_x(C, pauli_sum, pauli_index_x, pauli_index_z, target_qubit)
    else:
        C, pauli_sum, px = _handle_id_id(C, pauli_sum, pauli_index_x, pauli_index_z, target_qubit)

    if px == pauli_index_z:
        C.add_gate(H(target_qubit, 2))
        pauli_sum = H(target_qubit, 2).act(pauli_sum)

    return C, pauli_sum


def to_ix(pauli_string: PauliString, target_index: int, ignore: int | list[int] | None = None) -> Circuit:
    """Finds a circuit to turn a PauliString to III...IXI...II where the X is at target_index
    for qudits X = xrz0 for any r != 0, I = x0z0

    ignore is a list of qudits to not try to turn into an I
    """

    if ignore is None:
        ignore = []
    if isinstance(ignore, int):
        ignore = [ignore]
    if target_index in ignore:
        raise Exception("target_index must not be in ignore")

    p_string_in = pauli_string.copy()
    # First we make the target qudit an X
    circuit = to_x(pauli_string, target_index, ignore)
    pauli_string = circuit.act(p_string_in)
    n_q = pauli_string.n_qudits()
    for q in range(n_q):
        if q != target_index and q not in ignore:
            if pauli_string[q].x_exp == 0 and pauli_string[q].z_exp != 0:
                circuit.add_gate(H(q, pauli_string.dimensions[q]))
                pauli_string = circuit.act(p_string_in)
            if pauli_string[q].x_exp != 0:
                if pauli_string[q].z_exp != 0:
                    # use the x to cancel the z of q with S gates
                    n_s = solve_modular_linear_additive(pauli_string[q].x_exp, pauli_string[q].z_exp,
                                                        pauli_string.dimensions[q])
                    for i in range(n_s):
                        circuit.add_gate(S(q, pauli_string.dimensions[q]))
                    pauli_string = circuit.act(p_string_in)

                # use cnot to cancel the x of q with the x of target. n_cnot = n where x_q + x+_target) % d = 0
                n_cnot = solve_modular_linear_additive(pauli_string[q].x_exp, pauli_string[target_index].x_exp,
                                                       pauli_string.dimensions[target_index])
                if n_cnot is None:
                    raise Exception("Weird")
                for i in range(n_cnot):
                    circuit.add_gate(CX(target_index, q, pauli_string.dimensions[target_index]))
                pauli_string = circuit.act(p_string_in)

    else:
        return circuit


def _validate_inputs(pauli_string, target_index, ignore):
    if ignore is None:
        ignore = []
    if isinstance(ignore, int):
        ignore = [ignore]
    if target_index in ignore:
        raise Exception("target_index must not be in ignore")

    if target_index < 0:
        target_index += pauli_string.n_qudits()
    if target_index >= pauli_string.n_qudits():
        raise Exception(f"target_index {target_index} out of range {pauli_string.n_qudits()}")
    if PauliString.n_identities == pauli_string.n_qudits():
        raise Exception("PauliString is identity - cannot be converted to X")
    return ignore, target_index


def _single_qudit_x(pauli_string, target_index, circuit):
    dim_target = pauli_string.dimensions[target_index]
    x_exp = pauli_string[target_index].x_exp
    z_exp = pauli_string[target_index].z_exp
    if x_exp != 0 and z_exp == 0:
        return circuit
    elif x_exp != 0 and z_exp != 0:
        n_s_gates = solve_modular_linear_additive(z_exp, x_exp, dim_target)
        for _ in range(n_s_gates):
            circuit.add_gate(S(target_index, dim_target))
        return circuit
    elif x_exp == 0 and z_exp != 0:
        circuit.add_gate(H(target_index, dim_target))
        return circuit
    return None  # Not handled here


def _multi_qudit_x(pauli_string, target_index, ignore, circuit):
    n_q = pauli_string.n_qudits()
    dim_target = pauli_string.dimensions[target_index]
    for q in range(n_q - 1, -1, -1):
        if q != target_index and q not in ignore:
            x_exp = pauli_string[q].x_exp
            z_exp = pauli_string[q].z_exp
            if x_exp != 0:
                circuit.add_gate(CX(q, target_index, dim_target))
                return circuit
            if x_exp == 0 and z_exp != 0:
                circuit.add_gate(CX(target_index, q, dim_target))
                circuit.add_gate(H(target_index, dim_target))
                return circuit
    raise Exception(f"No circuit found to convert {pauli_string} pauli to X")


def to_x(pauli_string: PauliString, target_index: int, ignore: int | list[int] | None = None) -> Circuit:
    """Finds a circuit to turn a PauliString to ****X*** where the X is at target_index"""

    ignore, target_index = _validate_inputs(pauli_string, target_index, ignore)
    circuit = Circuit(dimensions=pauli_string.dimensions)
    result = _single_qudit_x(pauli_string, target_index, circuit)
    if result is not None:
        return result
    # If not handled by single qudit, try multi-qudit
    return _multi_qudit_x(pauli_string, target_index, ignore, circuit)
