#
# Author: James McClain <jdmcclain47@gmail.com>
#

import unittest
import numpy
from functools import reduce
from pyscf import gto
from pyscf.pbc.gto import Cell
from pyscf import symm
from pyscf.symm import geom
from pyscf.symm import kpoint_symm

# A good reference for reference unit cell symmetries is given at:
# http://som.web.cmu.edu/StructuresAppendix.pdf
def make_fcc_cell():
    cell = Cell()
    cell.unit = 'A'
    cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
    cell.a = [[0.    , 1.7834, 1.7834],
              [1.7834, 0.    , 1.7834],
              [1.7834, 1.7834, 0.    ]]
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.build()
    return cell

def make_zincblende_cell():
    cell = Cell()
    cell.unit = 'B'
    cell.atom = 'Ga 0.,  0.,  0.; As 2.6682931, 2.6682931, 2.6682931'
    cell.a = [[ 0.        ,  5.33658619,  5.33658619],
              [ 5.33658619,  0.        ,  5.33658619],
              [ 5.33658619,  5.33658619,  0.        ]]
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.build()
    return cell

def make_graphene_cell():
    cell = Cell()
    cell.unit = 'B'
    cell.atom = 'C 0.,  0.,  0.; C 0.,   2.68394317,   0.'
    cell.a = [[ 4.64872593,  0.        ,  0.        ],
              [-2.32436297,  4.02591475,  0.        ],
              [ 0.        ,  0.        ,  9.44862995]]
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.build()
    return cell

def make_rocksalt_cell():
    cell = Cell()
    cell.unit = 'B'
    cell.atom = 'Li 0.,  0.,  0.; Cl 4.84714716, 0., 0.'
    cell.a = [[ 0.        ,  4.84714716,  4.84714716],
              [ 4.84714716,  0.        ,  4.84714716],
              [ 4.84714716,  4.84714716,  0.        ]]
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.build()
    return cell

class KnowValues(unittest.TestCase):
    def test_inversion_pairs_223(self):
        # k-points from a 2x2x2 simple cubic cell without the gamma point
        coords = [[-0.27707654, -0.27707654, -0.36943539],
                  [-0.27707654, -0.27707654,  0.        ],
                  [-0.27707654, -0.27707654,  0.36943539],
                  [-0.27707654,  0.27707654, -0.36943539],
                  [-0.27707654,  0.27707654,  0.        ],
                  [-0.27707654,  0.27707654,  0.36943539],
                  [ 0.27707654, -0.27707654, -0.36943539],
                  [ 0.27707654, -0.27707654,  0.        ],
                  [ 0.27707654, -0.27707654,  0.36943539],
                  [ 0.27707654,  0.27707654, -0.36943539],
                  [ 0.27707654,  0.27707654,  0.        ],
                  [ 0.27707654,  0.27707654,  0.36943539]]
        inv_pairs = kpoint_symm.get_inversion_pairs(coords)
        result = [(0, 0, False), (1, 1, False), (2, 2, False), (3, 3, False),
                  (4, 4, False), (5, 5, False), (6, 5, True), (7, 4, True),
                  (8, 3, True), (9, 2, True), (10, 1, True), (11, 0, True)]
        self.assertEqual(inv_pairs, result)

    def test_inversion_pairs_333(self):
        # k-points from a 3x3x3 simple cubic cell
        coords = [[-0.36943539, -0.36943539, -0.36943539],
                  [-0.36943539, -0.36943539,  0.        ],
                  [-0.36943539, -0.36943539,  0.36943539],
                  [-0.36943539,  0.        , -0.36943539],
                  [-0.36943539,  0.        ,  0.        ],
                  [-0.36943539,  0.        ,  0.36943539],
                  [-0.36943539,  0.36943539, -0.36943539],
                  [-0.36943539,  0.36943539,  0.        ],
                  [-0.36943539,  0.36943539,  0.36943539],
                  [ 0.        , -0.36943539, -0.36943539],
                  [ 0.        , -0.36943539,  0.        ],
                  [ 0.        , -0.36943539,  0.36943539],
                  [ 0.        ,  0.        , -0.36943539],
                  [ 0.        ,  0.        ,  0.        ],
                  [ 0.        ,  0.        ,  0.36943539],
                  [ 0.        ,  0.36943539, -0.36943539],
                  [ 0.        ,  0.36943539,  0.        ],
                  [ 0.        ,  0.36943539,  0.36943539],
                  [ 0.36943539, -0.36943539, -0.36943539],
                  [ 0.36943539, -0.36943539,  0.        ],
                  [ 0.36943539, -0.36943539,  0.36943539],
                  [ 0.36943539,  0.        , -0.36943539],
                  [ 0.36943539,  0.        ,  0.        ],
                  [ 0.36943539,  0.        ,  0.36943539],
                  [ 0.36943539,  0.36943539, -0.36943539],
                  [ 0.36943539,  0.36943539,  0.        ],
                  [ 0.36943539,  0.36943539,  0.36943539]]
        inv_pairs = kpoint_symm.get_inversion_pairs(coords)
        result = [(0, 0, False), (1, 1, False), (2, 2, False), (3, 3, False),
                  (4, 4, False), (5, 5, False), (6, 6, False), (7, 7, False),
                  (8, 8, False), (9, 9, False), (10, 10, False), (11, 11, False),
                  (12, 12, False), (13, 13, False), (14, 12, True), (15, 11, True),
                  (16, 10, True), (17, 9, True), (18, 8, True), (19, 7, True),
                  (20, 6, True), (21, 5, True), (22, 4, True), (23, 3, True),
                  (24, 2, True), (25, 1, True), (26, 0, True)]
        self.assertEqual(inv_pairs, result)

    def test_d2h_permute_atoms_by_op(self):
        atoms = [['N', (-0.5, 0.0, -0.5)],
                 ['N', (-0.5, 0.0,  0.5)],
                 ['N', ( 0.5, 0.0,  0.5)],
                 ['N', ( 0.5, 0.0, -0.5)]]
        ops = [numpy.eye(3),
               numpy.diag((-1.,-1., 1.)),
               numpy.diag(( 1.,-1.,-1.)),
               numpy.diag((-1., 1.,-1.)),
               numpy.diag((-1.,-1.,-1.)),
               numpy.diag(( 1., 1.,-1.)),
               numpy.diag((-1., 1., 1.)),
               numpy.diag(( 1.,-1., 1.))]
        permutation = kpoint_symm.permute_atoms_by_op(ops, atoms)
        result = [[0, 1, 2, 3], [3, 2, 1, 0], [1, 0, 3, 2], [2, 3, 0, 1],
                  [2, 3, 0, 1], [1, 0, 3, 2], [3, 2, 1, 0], [0, 1, 2, 3]]
        self.assertEqual(permutation, result)

    def test_d2_permute_atoms_by_op(self):
        atoms = [['N', (-0.5, 0.0, -0.5)],
                 ['N', (-0.5, 0.0,  0.5)],
                 ['N', ( 0.5, 0.0,  0.5)],
                 ['N', ( 0.5, 0.0, -0.5)]]
        coords = [a[1] for a in atoms]
        ops = [numpy.eye(3),
               numpy.diag((-1.,-1., 1.)),
               numpy.diag(( 1.,-1.,-1.)),
               numpy.diag((-1., 1.,-1.))]
        permutation = kpoint_symm.permute_atoms_by_op(ops, atoms)
        result = [[0, 1, 2, 3], [3, 2, 1, 0], [1, 0, 3, 2], [2, 3, 0, 1]]
        self.assertEqual(permutation, result)

    def test_d2h_reorder_basis(self):
        atoms = [['N', (-0.5, 0.0, -0.5)],
                 ['N', (-0.5, 0.0,  0.5)],
                 ['N', ( 0.5, 0.0,  0.5)],
                 ['N', ( 0.5, 0.0, -0.5)]]
        basis = {'N': gto.basis.load('cc_pvqz', 'C'),}
        mol = gto.M(atom=atoms, basis=basis)

        permutation = [[0, 1, 2, 3], [1, 0, 3, 2]]
        idx = kpoint_symm.reorder_basis_idx(mol, permutation)
        orbital_range = numpy.arange(220)
        self.assertEqual(idx[0], list(orbital_range))

        ao0_slice = slice(0, 55)
        ao1_slice = slice(55, 110)
        ao2_slice = slice(110, 165)
        ao3_slice = slice(165, 220)
        result = []
        result.extend(list(orbital_range[ao1_slice]))
        result.extend(list(orbital_range[ao0_slice]))
        result.extend(list(orbital_range[ao3_slice]))
        result.extend(list(orbital_range[ao2_slice]))
        self.assertEqual(idx[1], result)

    def test_d2h_reorder_mixed_basis(self):
        atoms = [['N', (-0.5, 0.0, -0.5)],
                 ['N', (-0.5, 0.0,  0.5)],
                 ['N', ( 0.5, 0.0,  0.5)],
                 ['N1', ( 0.5, 0.0, -0.5)]]
        basis = {'N': gto.basis.load('cc_pvqz', 'C'),
                 'N1': gto.basis.load('cc_pvtz', 'C'),}
        mol = gto.M(atom=atoms, basis=basis)

        permutation = [[0, 1, 2, 3], [1, 0, 3, 2]]
        idx = kpoint_symm.reorder_basis_idx(mol, permutation)
        orbital_range = numpy.arange(195)
        self.assertEqual(idx[0], list(orbital_range))

        ao0_slice = slice(0, 55)
        ao1_slice = slice(55, 110)
        ao2_slice = slice(110, 165)
        ao3_slice = slice(165, 195)
        result = []
        result.extend(list(orbital_range[ao1_slice]))
        result.extend(list(orbital_range[ao0_slice]))
        result.extend(list(orbital_range[ao3_slice]))
        result.extend(list(orbital_range[ao2_slice]))
        self.assertEqual(idx[1], result)

    #def test_transformation_mapping(self):
    #    atoms = [['N', (-0.5, 0.0, -0.5)],
    #             ['N', (-0.5, 0.0,  0.5)],
    #             ['N', ( 0.5, 0.0,  0.5)],
    #             ['N', ( 0.5, 0.0, -0.5)]]
    #    basis = {'N': gto.basis.load('cc_pvqz', 'C'),
    #             'N1': gto.basis.load('cc_pvtz', 'C'),}
    #    mol = gto.M(atom=atoms, basis=basis)

    #    ops = [numpy.eye(3),
    #           numpy.diag((-1.,-1., 1.)),
    #           numpy.diag(( 1.,-1.,-1.)),
    #           numpy.diag((-1., 1.,-1.)),
    #           numpy.diag((-1.,-1.,-1.)),
    #           numpy.diag(( 1., 1.,-1.)),
    #           numpy.diag((-1., 1., 1.)),
    #           numpy.diag(( 1.,-1., 1.))]
    #    kpoint_symm.transformation_mapping(mol, ops)

    def test_get_symmetry_fcc_without_translations(self):
        cell = make_fcc_cell()
        tvec = cell.lattice_vectors()
        group, orig, axes = kpoint_symm.get_point_group_symmetry(tvec, cell._atom, basis=None)

        test_group = 'D3d'
        test_orig = [0., 0., 0.]
        test_axes = [[  7.07106781e-01, 5.55111512e-17, -7.07106781e-01],
                     [ -4.08248290e-01, 8.16496581e-01, -4.08248290e-01],
                     [  5.77350269e-01, 5.77350269e-01,  5.77350269e-01]]
        self.assertEqual(group, test_group)
        self.assertTrue(numpy.allclose(orig, test_orig))
        self.assertTrue(numpy.allclose(axes, test_axes))

    def test_get_symmetry_zincblende_without_translations(self):
        cell = make_zincblende_cell()
        tvec = cell.lattice_vectors()
        group, orig, axes = kpoint_symm.get_point_group_symmetry(tvec, cell._atom, basis=None)

        test_group = 'C3v'
        test_orig = [0., 0., 0.]
        test_axes = [[  7.07106781e-01, 5.55111512e-17, -7.07106781e-01],
                     [ -4.08248290e-01, 8.16496581e-01, -4.08248290e-01],
                     [  5.77350269e-01, 5.77350269e-01,  5.77350269e-01]]
        self.assertEqual(group, test_group)
        self.assertTrue(numpy.allclose(orig, test_orig))
        self.assertTrue(numpy.allclose(axes, test_axes))

    def test_get_symmetry_graphene_without_translations(self):
        cell = make_graphene_cell()
        tvec = cell.lattice_vectors()
        group, orig, axes = kpoint_symm.get_point_group_symmetry(tvec, cell._atom, basis=None)

        test_group = 'C2h'
        test_orig = [0., 0., 0.]
        test_axes = [[  9.49461708e-01,  3.13882885e-01,  0.00000000e+00],
                     [  3.13882885e-01, -9.49461708e-01,  0.00000000e+00],
                     [  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
        self.assertEqual(group, test_group)
        self.assertTrue(numpy.allclose(orig, test_orig))
        self.assertTrue(numpy.allclose(axes, test_axes))

    def test_get_symmetry_rocksalt_without_translations(self):
        cell = make_graphene_cell()
        tvec = cell.lattice_vectors()
        group, orig, axes = kpoint_symm.get_point_group_symmetry(tvec, cell._atom, basis=None)

        test_group = 'C2h'
        test_orig = [0., 0., 0.]
        test_axes = [[  9.49461708e-01,  3.13882885e-01, 0.000000000+00],
                     [  3.13882885e-01, -9.49461708e-01, 0.000000000+00],
                     [  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]
        self.assertEqual(group, test_group)
        self.assertTrue(numpy.allclose(orig, test_orig))
        self.assertTrue(numpy.allclose(axes, test_axes))

if __name__ == '__main__':
    unittest.main()
