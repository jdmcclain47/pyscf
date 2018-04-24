import numpy
from geom import rotation_mat
from geom import SymmSys
from pyscf.gto.mole import Mole
from pyscf.pbc.gto.cell import Cell
import pyscf.symm.kpoint_symm as kpoint_symm
import param
import geom
import basis as symm_basis

#ROTATION_OPERATOR = {
#    '1': numpy.array([1, 0, 0],
#                     [0, 1, 0],
#                     [0, 0, 1]),
#    '2a': numpy.array([1,  0,  0],
#                      [0, -1,  0],
#                      [0,  0, -1]),
#    '2b': numpy.array([-1,  0,  0],
#                      [ 0,  1,  0],
#                      [ 0,  0, -1]),
#    '2c': numpy.array([-1,  0,  0],
#                      [ 0, -1,  0],
#                      [ 0,  0,  1]),
#    '2d': numpy.array([ 0,  1,  0],
#                      [ 1,  0,  0],
#                      [ 0,  0, -1]),
#    '2e': numpy.array([ 0, -1,  0],
#                      [-1,  0,  0],
#                      [ 0,  0, -1]),
#    '2f': numpy.array([ 1, -1,  0],
#                      [ 0, -1,  0],
#                      [ 0,  0, -1]),
#    '2g': numpy.array([ 1,  0,  0],
#                      [ 1, -1,  0],
#                      [ 0,  0, -1]),
#    '3q': numpy.array([ 0,  0,  1],
#                      [ 1,  0,  0],
#                      [ 0,  1,  0]),
#    '3c': numpy.array([ 0, -1,  0],
#                      [ 1, -1,  0],
#                      [ 0,  0,  1]),
#    '4c': numpy.array([ 0, -1,  0],
#                      [ 1,  0,  0],
#                      [ 0,  0,  1]),
#    '6c': numpy.array([ 1, -1,  0],
#                      [ 1,  0,  0],
#                      [ 0,  0,  1]),
#
#}

TOLERANCE = 1e-8

def get_unit_cell_d2h_subgroup(kpts, atoms, basis):
    kpts_object = zip(['GHOST']*len(kpts), kpts)
    print 'kpoints', kpts
    kptsys = SymmSys(kpts_object, recenter_coords=False)
    atomsys = SymmSys(atoms, recenter_coords=True)

    kpt_topgroup, kpt_orig, kpt_axes = \
       geom.detect_symm(kpts_object, recenter_coords=False)
    kpt_group, kpt_axes = geom.subgroup(kpt_topgroup, kpt_axes)
    print 'kpoint topgroup =', kpt_topgroup
    print 'kpoint subgroup =', kpt_group
    print 'axes', kpt_axes

    atm_topgroup, atm_orig, atm_axes = \
       geom.detect_symm(atoms)
    atm_group, atm_axes = geom.subgroup(atm_topgroup, atm_axes)
    print 'atomic topgroup =', atm_topgroup
    print 'atomic subgroup =', atm_group
    print 'axes', atm_axes

    cell_object = []
    atoms = [[atoms[a][0], atomsys.atoms[a][1:]] for a in range(len(atoms))]
    [cell_object.append(a) for a in atoms]
    [cell_object.append(k) for k in kpts_object]
    cell_topgroup, cell_orig, cell_axes = \
       geom.detect_symm(cell_object, recenter_coords=False)
    print 'cell topgroup =', cell_topgroup
    cell_subgroup, cell_axes = geom.subgroup(cell_topgroup, cell_axes)
    print 'cell subgroup =', cell_subgroup
    print 'cell orig', cell_orig
    print 'cell axes', cell_axes
    return cell_subgroup, cell_orig, cell_axes

def get_unit_cell_point_group(kpts, atoms, basis):
    return get_unit_cell_d2h_subgroup(kpts, atoms, basis)

def get_stars(kpts, atoms, basis, only_inversion=False):
    kpts_object = zip(['GHOST']*len(kpts), kpts)
    cell_subgroup, cell_orig, cell_axes = get_unit_cell_point_group(kpts, atoms, basis)
    if not geom.check_given_symm(cell_subgroup, kpts_object, axis=None):
        print 'symmetry not found!'

    kpt_stars, kpt_ops = geom.symm_identical_atoms(cell_subgroup, kpts_object, recenter_coords=False, return_ops=True, axis=None)

    kpt_inversions = kpoint_symm.get_inversion_pairs(kpts)
    atom_mol = Cell(atom=atoms, basis=basis).build(a=numpy.eye(3)).to_mol()

    inversion = []
    basis_mapping = []
    ao_transformation = []
    for istar in range(len(kpt_stars)):
        ao_transformation.append(kpoint_symm.transformation_mapping(atom_mol, kpt_ops[istar]))
        atm_mapping = kpoint_symm.permute_atoms_by_op(kpt_ops[istar], atoms)
        basis_mapping.append(kpoint_symm.reorder_basis_idx(atom_mol, atm_mapping))

    return kpt_stars, kpt_ops, kpt_inversions, ao_transformation, basis_mapping, cell_axes
