#!/usr/bin/env python
#
# Authors: Garnet Chan <gkc1000@gmail.com>
#          Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbchf
from pyscf.pbc.scf import khf
from pyscf.pbc.scf import symm_khf
from pyscf.pbc.scf import kuhf
import pyscf.pbc.tools

TOLERANCE = 1e-8

def finger(a):
    return np.dot(np.cos(np.arange(a.size)), a.ravel())

def make_diamond_primitive(ngs):
    cell = pbcgto.Cell()
    cell.unit = 'A'
    cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
    cell.a = '''0.      1.7834  1.7834
                1.7834  0.      1.7834
                1.7834  1.7834  0.    '''

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.gs = np.array([ngs,ngs,ngs])
    cell.ibz_symmetry = True

    cell.conv_tol = TOLERANCE
    cell.precision = TOLERANCE
    cell.verbose = 9
    #cell.output = '/dev/null'
    cell.build()
    return cell

def make_diamond_primitive2(ngs, with_symmetry):
    cell = pbcgto.Cell()
    cell.unit = 'A'
    cell.atom = 'C 0.,  0.,  -0.772234852555; C 0.0, 0.0, 0.772234852555'
    #cell.a = [[ -5.32204139e-01, -8.46614525e-01,  1.61222390e-03],
    #          [  8.46615626e-01, -5.32204831e-01,  0.000000000000],
    #          [  8.58033350e-04,  1.36493395e-03,  9.99998700e-01]]
    cell.a = np.array(
             [[-2.38304713,  1.3758529,   3.89149967],
              [ 0.        , -2.7517058,   3.89149967],
              [ 2.38304713,  1.3758529,   3.89149967]])
    cell.a /= 1.8897261245654366

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.gs = np.array([ngs,ngs,ngs])
    cell.ibz_symmetry = with_symmetry

    cell.conv_tol = TOLERANCE
    cell.precision = TOLERANCE
    cell.verbose = 9
    #cell.output = '/dev/null'
    cell.build()
    return cell

def make_gaas_primitive(ngs, with_symmetry):
    cell = pbcgto.Cell()
    cell.unit = 'B'
    cell.atom = 'Ga 0.,  0.,  0.; As 2.6682931, 2.6682931, 2.6682931'
    cell.a = [[ 0.        ,  5.33658619,  5.33658619],
              [ 5.33658619,  0.        ,  5.33658619],
              [ 5.33658619,  5.33658619,  0.        ]]

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.gs = np.array([ngs,ngs,ngs])
    cell.ibz_symmetry = with_symmetry

    cell.conv_tol = TOLERANCE
    cell.precision = TOLERANCE
    cell.verbose = 9
    #cell.output = '/dev/null'
    cell.build()
    return cell

def make_n4_cell(ngs):
    cell = pbcgto.Cell()
    cell.atom = '''
    N -0.5 0.0 -0.5
    N -0.5 0.0  0.5
    N  0.5 0.0  0.5
    N  0.5 0.0 -0.5
    '''
    cell.basis = { 'N': 'gth-szv' }
    cell.pseudo = 'gth-pade'
    cell.a = np.eye(3) * 3
    cell.gs = [ngs] * 3
    cell.conv_tol = TOLERANCE
    cell.precision = TOLERANCE
    cell.verbose = 9
    cell.build()
    return cell

def make_n4_perturbed_cell(ngs):
    cell = pbcgto.Cell()
    cell.atom = '''
    N -0.5 -0.15 -0.5
    N -0.5  0.15  0.5
    N  0.5 -0.15  0.5
    N  0.5  0.15 -0.5
    '''
    cell.basis = { 'N': 'gth-szv' }
    cell.pseudo = 'gth-pade'
    cell.a = np.eye(3) * 3
    cell.gs = [ngs] * 3
    cell.conv_tol = TOLERANCE
    cell.precision = TOLERANCE
    cell.verbose = 9
    cell.build()
    return cell

class KnowValues(unittest.TestCase):
    def _test_d2h(self, nk=[2,2,2], with_gamma_point=True, with_symmetry=False):
        cell = make_n4_cell(24)

        my_kpts = cell.make_kpts(nk, with_gamma_point=with_gamma_point)
        if with_symmetry:
            mf = symm_khf.KRHF(cell, my_kpts)
        else:
            mf = khf.KRHF(cell, my_kpts)
        mf.direct_scf_tol = TOLERANCE
        mf.conv_tol_grad = TOLERANCE * 1000
        e = mf.kernel()

        #print e
        #for ikpt, kpt, en in zip(range(np.prod(nk)), my_kpts, mf.mo_energy_kpts):
        #    print 'kpt', ikpt, '(', kpt, ')', en
        return e

    def _test_d2h_perturbed(self, nk=[2,2,2], with_gamma_point=True, with_symmetry=False):
        cell = make_n4_perturbed_cell(24)

        my_kpts = cell.make_kpts(nk, with_gamma_point=with_gamma_point)
        if with_symmetry:
            mf = symm_khf.KRHF(cell, my_kpts)
        else:
            mf = khf.KRHF(cell, my_kpts)
        mf.direct_scf_tol = TOLERANCE
        mf.conv_tol_grad = TOLERANCE * 1000
        e = mf.kernel()

        #print e
        #for ikpt, kpt, en in zip(range(np.prod(nk)), my_kpts, mf.mo_energy_kpts):
        #    print 'kpt', ikpt, '(', kpt, ')', en
        return e

    def _test_fcc(self, nk=[2,2,2], with_gamma_point=True, with_symmetry=False):
        cell = make_diamond_primitive2(24, with_symmetry)

        my_kpts = cell.make_kpts(nk, with_gamma_point=with_gamma_point)
        if with_symmetry:
            mf = symm_khf.KRHF(cell, my_kpts)
        else:
            mf = khf.KRHF(cell, my_kpts)
        mf.direct_scf_tol = TOLERANCE
        mf.conv_tol_grad = TOLERANCE * 1000
        e = mf.kernel()

        #print e
        #for ikpt, kpt, en in zip(range(np.prod(nk)), my_kpts, mf.mo_energy_kpts):
        #    print 'kpt', ikpt, '(', kpt, ')', en
        return e

    def _test_zincblende(self, nk=[2,2,2], with_gamma_point=True, with_symmetry=False):
        cell = make_gaas_primitive(24, with_symmetry)

        my_kpts = cell.make_kpts(nk, with_gamma_point=with_gamma_point)
        if with_symmetry:
            mf = symm_khf.KRHF(cell, my_kpts)
        else:
            mf = khf.KRHF(cell, my_kpts)
        mf.direct_scf_tol = TOLERANCE
        mf.conv_tol_grad = TOLERANCE * 1000
        e = mf.kernel()

        #print e
        #for ikpt, kpt, en in zip(range(np.prod(nk)), my_kpts, mf.mo_energy_kpts):
        #    print 'kpt', ikpt, '(', kpt, ')', en
        return e


    def test_zincblende_222_without_gamma(self):
        nk = [2, 2, 2]
        e = -78.4906715912
        e_symm = self._test_zincblende(nk=nk, with_gamma_point=False, with_symmetry=True)
        print 'energy (no symmetry) = ', e
        print 'energy (symmetry) = ', e_symm

    def test_zincblende_333_without_gamma(self):
        nk = [3, 3, 3]
        e = -78.4417504051
        e_symm = self._test_zincblende(nk=nk, with_gamma_point=False, with_symmetry=True)
        print 'energy (no symmetry) = ', e
        print 'energy (symmetry) = ', e_symm

    def test_fcc_222_without_gamma(self):
        nk = [2, 2, 2]
        e = -11.0711250775
        e_symm = self._test_fcc(nk=nk, with_gamma_point=False, with_symmetry=True)
        self.assertAlmostEqual(e_symm, e, 8)
        print 'energy (no symmetry) = ', e
        print 'energy (symmetry) = ', e_symm

    def test_fcc_333_without_gamma(self):
        nk = [3, 3, 3]
        e = -11.0002012359
        e_symm = self._test_fcc(nk=nk, with_gamma_point=False, with_symmetry=True)
        self.assertAlmostEqual(e_symm, e, 8)
        print 'energy (no symmetry) = ', e
        print 'energy (symmetry) = ', e_symm

    def test_d2h_222_without_gamma(self):
        nk = [2, 2, 2]
        e = -36.5507045293
        e_symm = self._test_d2h(nk=nk, with_gamma_point=False, with_symmetry=True)
        print 'energy (no symmetry) = ', e
        print 'energy (symmetry) = ', e_symm

    def test_d2h_333_without_gamma(self):
        nk = [3, 3, 3]
        e = -36.5011269864
        e_symm = self._test_d2h(nk=nk, with_gamma_point=False, with_symmetry=True)
        print 'energy (no symmetry) = ', e
        print 'energy (symmetry) = ', e_symm


if __name__ == '__main__':
    print("Full Tests for pbc.scf.khf")
    unittest.main()
