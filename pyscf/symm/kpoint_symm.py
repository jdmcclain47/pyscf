import sys
import re
import numpy
import scipy.linalg
from pyscf.gto import mole
from pyscf.lib import logger
from pyscf.lib import norm
from pyscf.lib.numpy_helper import cartesian_prod
import pyscf.symm.geom as geom
from pyscf.symm.geom import SymmSys
from pyscf.symm.geom import argsort_coords
from pyscf.symm.basis import tot_parity_odd
import pyscf.symm.param

TOLERANCE = 1e-5


def _pg_symm_no_translations(translation_vectors, atoms, basis=None):
    unit_cell = numpy.array(cartesian_prod(([0,1], [0,1], [0,1])))
    cell_lattice = numpy.dot(unit_cell, translation_vectors)
    cell_center = mole.charge_center(zip([1] * len(cell_lattice), cell_lattice))
    cell_lattice -= cell_center

    # Create a lattice object to be sent in to the symmetry modules
    lattice_object = zip(['GHOST'] * len(cell_lattice), cell_lattice)
    atomsys = SymmSys(atoms, recenter_coords=True)

    # Find the subgroup of the lattice (for curiosity)
    lattice_topgroup, lattice_orig, lattice_axes = \
       geom.detect_symm(lattice_object, recenter_coords=True)
    lattice_group, lattice_axes = geom.subgroup(lattice_topgroup, lattice_axes)
    print 'kpoint topgroup =', lattice_topgroup
    print 'kpoint subgroup =', lattice_group
    print 'axes', lattice_axes

    # Find the subgroup of the atomic system
    atm_topgroup, atm_orig, atm_axes = \
       geom.detect_symm(atoms)
    atm_group, atm_axes = geom.subgroup(atm_topgroup, atm_axes)
    print 'atomic topgroup =', atm_topgroup
    print 'atomic subgroup =', atm_group
    print 'axes', atm_axes

    # Find the subgroup of the unit cell
    cell_object = []
    #geom.shift_atom(atoms, mole.charge_center(atoms), numpy.eye(3))
    print mole.charge_center(atoms)
    print atoms
    atoms = [[atoms[a][0], atomsys.atoms[a][1:]] for a in range(len(atoms))]
    print atoms
    [cell_object.append(a) for a in atoms]
    [cell_object.append(k) for k in lattice_object]
    cell_topgroup, cell_orig, cell_axes = \
       geom.detect_symm(cell_object, recenter_coords=True)
    print 'cell topgroup =', cell_topgroup
    print 'cell orig = ', cell_orig
    print 'cell axes', cell_axes
    return cell_topgroup, cell_orig, cell_axes


def get_point_group_symmetry(translation_vectors, atoms, basis=None,
                             consider_translations=False):
    if consider_translations:
        raise NotImplementedError('Symmetry that includes translations after operations has not been implemented.')
    else:
        return _pgroup_symm_no_translations(translation_vectors, atoms, basis=None)


def permute_atoms_by_op(ops, atoms, recenter_coords=True):
    """

    For a given symmetry operations `ops`, finds the mapping of original atomic indices
    to the new indices under this operation(s).

    Parameters
    ----------
    ops : list of (3,3) `ndarray`
        Symmetry operation.
    atoms : list of (str, (3,) ndarray)
        List of all atoms in the system, where each atom is defined by a
        string for type and a set of coordinates.
    recenter_coords: bool
        Whether to recenter the coordinates.  For atomic systems, this will generally
        be true.  For finding symmetries of k-points, this may not be desired.

    Returns
    -------
    mapping : list of integers
        Mapping of original system to new system under each operations `ops`.

    """
    assert ops[0].shape == (3,3)

    natms = len(atoms)
    coords = numpy.array([a[1] for a in atoms], dtype=float)
    if recenter_coords:
        center = mole.charge_center([[atoms[i][0], coords[i]] for i in range(natms)])
        coords -= center
    idx = argsort_coords(coords)
    coords_sorted = coords[idx]

    mapping = []
    # Apply operations to all coordinates; these *should* map to other
    # atomic coordinates if the system is invariant under these operations.
    for op in ops:
        newc = numpy.dot(coords, op)
        idx = argsort_coords(newc)
        if not numpy.allclose(coords_sorted, newc[idx], atol=TOLERANCE):
            raise RuntimeError('Symmetry identical atoms not found')

        diff = numpy.linalg.norm(newc[:, None] - coords[None, :], axis=2)
        eq_idx = [numpy.where(diff[ix] < TOLERANCE)[0] for ix in range(diff.shape[0])]
        # Make sure each atom maps to a unique atom
        num_eq = numpy.array([len(eq_idx[ix]) for ix in range(diff.shape[0])])
        if any(num_eq > 1):
            raise RuntimeError('Duplicate points given?\n coords = %s' % coords)

        mapped = [int(eq_idx[ix]) for ix in range(diff.shape[0])]
        mapping.append(mapped)

    return mapping


def reorder_basis_idx(mol, atom_permutation):
    '''

    According to the new atomic mapping `atm_mapping`, gives the indices of the
    basis to map the old basis to the new one.

    Parameters
    ----------
    mol: `gto.Mole`
        Contains atomic system data.
    atm_mapping: list of int or list of list of int
        New mapping of atomic indices.

    Returns
    -------
    mapping: list of int
        New mapping of basis functions.

    '''
    assert isinstance(atom_permutation[0], (list, numpy.ndarray))

    mapping = []
    for permutation in atom_permutation:
        generating_coord = mol.atom_coord(permutation[0])
        aoslice = mol.aoslice_by_atom()

        # Get the offsets of the atomic orbitals for these atoms and rearrange
        ao_slice = [range(*aoslice[i, 2:]) for i in permutation]
        out_idx = [y for x in ao_slice for y in x]  # Flatten array
        mapping.append(out_idx)

    return mapping


def get_inversion_pairs(coords):
    """

    For a given set of coordinates this constructs a list of tuples, where the i'th tuple represents:

        `(point_i, irreducible_point_i, irreducible_is_inverse_of_point_i)`

    Parameters
    ----------
    coords: list of (3,) `ndarray`
        Coordinates of elements to find inversion pairs.

    Returns
    -------
    inv_list : list of `(int, int, bool)` tuples

    Note
    ----
    Does not take basis into account.  This is used for k-points.

    """
    coords = numpy.array(coords)
    nelements = len(coords)

    inversion_pairs = [0] * nelements
    inversion_idx = numpy.arange(nelements)
    found_inversion = numpy.zeros(nelements, dtype=bool)

    for ix in range(nelements):
        if not found_inversion[ix]:
            found_inversion[ix] = True

            # Grab indices of remaining coordinates that have no coordinate related by inversion
            # and see whether they're related by inversion to the current coordinate.
            test_coords_idx = inversion_idx[~found_inversion]
            test_coords = coords[~found_inversion] + coords[ix]
            distance = numpy.sqrt(numpy.einsum('ix,ix->i', test_coords, test_coords))
            inverse_idx = test_coords_idx[numpy.where(distance < TOLERANCE)]

            if len(inverse_idx) == 0:  # No inversion center
                inversion_pairs[ix] = (ix, ix, False)
                found_inversion[ix] = True
            elif len(inverse_idx) == 1:
                inverse_idx = inverse_idx[0]

                inversion_pairs[ix] = (ix, ix, False)
                inversion_pairs[inverse_idx] = (inverse_idx, ix, True)
                found_inversion[ix] = found_inversion[inverse_idx] = True
            else:
                raise RuntimeError('Duplicate points given?\n coords = %s' % coords)

    return inversion_pairs


def transformation_mapping(mol, group_ops, recenter_coords=True):
    nao = mol.nao_nr()
    aoslice = mol.aoslice_by_atom()
    atom_coords = mol.atom_coords()
    if recenter_coords:
        atom_coords -= mole.charge_center([[mol._atom[i][0], atom_coords[i]] for i in range(len(atom_coords))])
    natoms = len(atom_coords)
    idx = argsort_coords(atom_coords)
    atom_coords_sorted = atom_coords[idx]

    nirrep = len(group_ops)

    d2h_operations = pyscf.symm.param.D2H_OPS
    group_ops_symbols = []
    # Check to make sure all `group_ops` are contained within the D2H set of operations
    for op in group_ops:
        found = False
        for group_op_key, group_op_value in d2h_operations.items():
            if numpy.linalg.norm(op - group_op_value) < 1e-14:
                found = True
                group_ops_symbols.append(group_op_key)
        if not found:
            raise ValueError('No operation key found for op = %s' % op)

        newc = numpy.dot(atom_coords, op)
        idx = argsort_coords(newc)
        if not numpy.allclose(atom_coords_sorted, newc[idx], atol=TOLERANCE):
            raise RuntimeError('Symmetry identical atoms not found for op = %s' % op)

    ao_start_idx = numpy.array([aoslice[i, 2] for i in range(natoms)])
    transformation_matrix = [numpy.identity(nao, dtype=numpy.complex128) for i in range(nirrep)]
    # Could be faster if we could group all atoms with the same basis set and applied
    # it to basis sets associated with those atoms.
    for atom_id in range(natoms):
        start_shell_id, end_shell_id = aoslice[atom_id, :2]
        ip = 0
        for shell_id in range(start_shell_id, end_shell_id):
            l = mol.bas_angular(shell_id)
            raise NotImplementedError("Cartesian GTO basis has not been tested")
            if mol.cart:  # Cartesian GTO basis
                degen = (l + 1) * (l + 2) // 2
                for op_id, op in enumerate(group_ops_symbols):
                    n = 0
                    for x in range(l, -1, -1):
                        for y in range(l-x, -1, -1):
                            z = l-x-y
                            idx = ao_start_idx[op_id] + n
                            sign = op[0, 0]**x * op[1, 1]**y * op[2, 2]**z
                            n += 1
            else:
                degen = l * 2 + 1
                for op_id, op_str in enumerate(group_ops_symbols):
                    for n, m in enumerate(range(-l, l+1)):
                        idx = ao_start_idx[atom_id] + n
                        transformation_matrix[op_id][ip + idx, ip + idx] = (-1)**tot_parity_odd(op_str, l, m)

            # Fill these transformations into the other shells
            idx = ao_start_idx[atom_id]
            for i in range(1, mol.bas_nctr(shell_id)):
                idx_start = ao_start_idx[atom_id] + ip + i * degen
                idx_end = ao_start_idx[atom_id] + ip + (i + 1) * degen
                for op_id, op_str in enumerate(group_ops_symbols):
                    transformation_matrix[op_id][idx_start:idx_end, idx_start:idx_end] = \
                            transformation_matrix[op_id][ip:ip + degen, ip:ip + degen]

            for i in range(mol.bas_nctr(shell_id)):
                ip += degen

    return transformation_matrix
