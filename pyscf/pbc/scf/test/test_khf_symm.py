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
from pyscf.pbc.scf import khf_symm
from pyscf.pbc.scf import kuhf
import pyscf.pbc.tools

TOLERANCE = 1e-8

def make_diamond_primitive(ngs, with_symmetry):
    cell = pbcgto.Cell()
    cell.unit = 'A'
    cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
    cell.a = '''0.      1.7834  1.7834
                1.7834  0.      1.7834
                1.7834  1.7834  0.    '''

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.gs = np.array([ngs,ngs,ngs])
    cell.ibz_symmetry = with_symmetry

    cell.precision = TOLERANCE
    cell.verbose = 9
    cell.output = '/dev/null'
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

    cell.precision = TOLERANCE
    cell.verbose = 9
    cell.output = '/dev/null'
    cell.build()
    return cell

def make_d2h_primitive(ngs, with_symmetry):
    cell = pbcgto.Cell()
    cell.atom = '''
    N -0.5 0.0 -0.5
    N -0.5 0.0  0.5
    N  0.5 0.0  0.5
    N  0.5 0.0 -0.5
    '''
    cell.a = np.eye(3) * 3

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.gs = np.array([ngs,ngs,ngs])
    cell.ibz_symmetry = with_symmetry

    cell.precision = TOLERANCE
    cell.verbose = 9
    cell.output = '/dev/null'
    cell.build()
    return cell

class KnowValues(unittest.TestCase):
    def _test_d2h(self, nk=[2,2,2], with_gamma_point=True, with_symmetry=False, only_coulomb=False):
        cell = make_d2h_primitive(24, with_symmetry)

        my_kpts = cell.make_kpts(nk, with_gamma_point=with_gamma_point)
        if with_symmetry:
            mf = khf_symm.KRHF(cell, my_kpts)
        else:
            mf = khf.KRHF(cell, my_kpts)

        if only_coulomb:
            def get_jk(*args):
                vj = mf.get_j(*args)
                vk = 0.0 * vj
                return vj, vk
            mf.get_jk = get_jk

        mf.direct_scf_tol = TOLERANCE
        mf.conv_tol_grad = TOLERANCE * 1000
        e = mf.kernel()
        return e

    def _test_fcc(self, ng=24, nk=[2,2,2], with_gamma_point=True, with_symmetry=False,
                  only_coulomb=False):
        cell = make_diamond_primitive(ng, with_symmetry)

        my_kpts = cell.make_kpts(nk, with_gamma_point=with_gamma_point)
        if with_symmetry:
            mf = khf_symm.KRHF(cell, my_kpts)
        else:
            mf = khf.KRHF(cell, my_kpts)

        if only_coulomb:
            def get_jk(*args):
                vj = mf.get_j(*args)
                vk = 0.0 * vj
                return vj, vk
            mf.get_jk = get_jk

        mf.direct_scf_tol = TOLERANCE
        mf.conv_tol_grad = TOLERANCE * 1000
        e = mf.kernel()
        return e

    def _test_zincblende(self, nk=[2,2,2], with_gamma_point=True, with_symmetry=False, only_coulomb=False):
        cell = make_gaas_primitive(24, with_symmetry)

        my_kpts = cell.make_kpts(nk, with_gamma_point=with_gamma_point)
        if with_symmetry:
            mf = khf_symm.KRHF(cell, my_kpts)
        else:
            mf = khf.KRHF(cell, my_kpts)

        if only_coulomb:
            def get_jk(*args):
                vj = mf.get_j(*args)
                vk = 0.0 * vj
                return vj, vk
            mf.get_jk = get_jk

        mf.direct_scf_tol = TOLERANCE
        mf.conv_tol_grad = TOLERANCE * 1000
        e = mf.kernel()
        return e

    def test_zincblende_222_without_gamma(self):
        nk = [2, 2, 2]
        e = -78.4906715912
        e_symm = self._test_zincblende(nk=nk, with_gamma_point=False, with_symmetry=True)
        #print 'energy (symmetry) = ', e_symm

    def test_zincblende_333_without_gamma(self):
        nk = [3, 3, 3]
        e = -78.4417504051
        e_symm = self._test_zincblende(nk=nk, with_gamma_point=False, with_symmetry=True)
        #print 'energy (symmetry) = ', e_symm

    def test_fcc_222_without_gamma_only_coulomb(self):
        nk = [2, 2, 2]
        e = self._test_fcc(nk=nk, with_gamma_point=False, with_symmetry=True, only_coulomb=False)
        e_symm = self._test_fcc(nk=nk, with_gamma_point=False, with_symmetry=True, only_coulomb=True)
        self.assertAlmostEqual(e_symm, e, 8)

    def test_fcc_222_without_gamma(self):
        nk = [2, 2, 2]
        e = -11.0711250775
        e_symm = self._test_fcc(nk=nk, with_gamma_point=False, with_symmetry=True)
        self.assertAlmostEqual(e_symm, e, 8)
        #print 'energy (symmetry) = ', e_symm

    def test_fcc_333_without_gamma(self):
        nk = [3, 3, 3]
        e = -11.0002012359
        e_symm = self._test_fcc(nk=nk, with_gamma_point=False, with_symmetry=True)
        self.assertAlmostEqual(e_symm, e, 8)
        #print 'energy (symmetry) = ', e_symm

    def test_d2h_222_without_gamma(self):
        nk = [2, 2, 2]
        e = -36.5507045293
        e_symm = self._test_d2h(nk=nk, with_gamma_point=False, with_symmetry=True)
        #print 'energy (symmetry) = ', e_symm

    def test_d2h_333_without_gamma(self):
        nk = [3, 3, 3]
        e = -36.5011269864
        e_symm = self._test_d2h(nk=nk, with_gamma_point=False, with_symmetry=True)
        #print 'energy (symmetry) = ', e_symm


if __name__ == '__main__':
    print("Full Tests for pbc.scf.khf_symm")
    unittest.main()
