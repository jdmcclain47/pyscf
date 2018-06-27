#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: James D. McClain
#          Timothy Berkelbach <tim.berkelbach@gmail.com>
#

import unittest
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf

from pyscf import scf
import pyscf.cc
import pyscf.pbc.cc
import pyscf.pbc.cc.kccsd_rhf
import pyscf.pbc.cc.ccsd
import make_test_cell
from pyscf.pbc.tools import pbc

import pyscf.pbc.cc.kccsd_t_rhf as kccsd_t_rhf

def run_kcell(cell, n, nk):
    #############################################
    # Do a k-point calculation                  #
    #############################################
    abs_kpts = cell.make_kpts(nk, wrap_around=True)

    #############################################
    # Running HF                                #
    #############################################
    kmf = pbcscf.KRHF(cell, abs_kpts, exxdiv=None)
    kmf.conv_tol = 1e-14
    #kmf.verbose = 7
    ekpt = kmf.scf()


    cc = pyscf.pbc.cc.kccsd_rhf.RCCSD(kmf)
    cc.conv_tol=1e-8
    cc.verbose = 7
    ecc, t1, t2 = cc.kernel()
    return ekpt, ecc

class KnownValues(unittest.TestCase):
    def test_311_n1_high_cost(self):
        L = 7.0
        n = 9
        cell = make_test_cell.test_cell_n1(L,[n]*3)
        nk = (3, 1, 1)
        hf_311 = -0.92687629918229486
        cc_311 = -0.042702177586414237
        escf, ecc = run_kcell(cell,n,nk)
        self.assertAlmostEqual(escf,hf_311, 9)
        self.assertAlmostEqual(ecc, cc_311, 6)

    def test_ipccsd_311_n1_high_cost(self):
        L = 7.0
        n = 9
        cell = make_test_cell.test_cell_n1(L,[n]*3)
        nk = (3, 1, 1)

        abs_kpts = cell.make_kpts(nk, wrap_around=True)
        kmf = pbcscf.KRHF(cell, abs_kpts, exxdiv=None)
        kmf.conv_tol = 1e-14
        escf = kmf.scf()

        cc = pyscf.pbc.cc.kccsd_rhf.RCCSD(kmf)
        cc.conv_tol=1e-14
        cc.verbose = 7
        ecc, t1, t2 = cc.kernel()

        hf_311 = -0.92687629918229486
        cc_311 = -0.042702177586414237
        self.assertAlmostEqual(escf,hf_311, 9)
        self.assertAlmostEqual(ecc, cc_311, 6)

        ew, ev = cc.ipccsd(nroots=1, kptlist=[0], partition='mp')
        self.assertAlmostEqual(ew[0], 0.1961932627711932, 6)
        lew, lev = cc.lipccsd(nroots=3, kptlist=[0], partition='mp')
        self.assertAlmostEqual(lew[0], 0.1961932627711932, 6)

        ew, ev = cc.ipccsd(nroots=3, kptlist=[0])
        self.assertAlmostEqual(ew[0], 0.1858896225849556, 6)
        self.assertAlmostEqual(ew[1], 0.3079196211332019, 6)
        self.assertAlmostEqual(ew[2], 0.3206246035042978, 6)
        lew, lev = cc.lipccsd(nroots=3, kptlist=[0])
        self.assertAlmostEqual(lew[0], 0.1858896225849556, 6)
        self.assertAlmostEqual(lew[1], 0.3079196211332019, 6)
        self.assertAlmostEqual(lew[2], 0.3206246035042978, 6)

        ew_star = cc.ipccsd_star(ew, ev, lev, kptlist=[0])
        self.assertAlmostEqual(ew_star[0][0], 0.18211313999415324, 6)

    def test_ipccsd_311_n3_high_cost(self):
        n = 9
        cell = make_test_cell.test_cell_n3([n]*3)
        nk = (2, 1, 1)

        abs_kpts = cell.make_kpts(nk, wrap_around=True)
        kmf = pbcscf.KRHF(cell, abs_kpts, exxdiv=None)
        escf = kmf.scf()
        self.assertAlmostEqual(escf, -4.55108889617723, 6)

        cc = pyscf.pbc.cc.kccsd_rhf.RCCSD(kmf)
        ecc, t1, t2 = cc.kernel()
        self.assertAlmostEqual(ecc, -0.3019060312035857, 6)

        delta_ccsd, pt1, pt2 = cc.get_t3p2_amplitude_contribution(cc.t1, cc.t2)
        self.assertAlmostEqual(delta_ccsd, -4.34163688971623e-03, 6)

        cc.t1 = pt1
        cc.t2 = pt2
        ew, ev = cc.ipccsd(nroots=3, kptlist=[0])
        self.assertAlmostEqual(ew[0], -1.233240987367493, 6)
        self.assertAlmostEqual(ew[1], -1.212807603472019, 6)
        self.assertAlmostEqual(ew[2], -1.180617648232528, 6)
        lew, lev = cc.lipccsd(nroots=3, kptlist=[0])
        self.assertAlmostEqual(ew[0], lew[0], 6)

        ew_star = cc.ipccsd_star(ew, ev, lev, kptlist=[0])
        self.assertAlmostEqual(ew_star[0][0], -1.234322995604, 6)

        # Correction to CCSD with supercell (energy per cell)
        # mesh=[25,25,25]: -4.290875592567e-03

        #n = 25
        #cell = make_test_cell.test_cell_n3([n]*3)
        #nk = (2, 1, 1)

        #scell = pbc.super_cell(cell, nk)
        #smf = pbcscf.RHF(scell, exxdiv=None)
        #smf.conv_tol_grad = 1e-9
        #smf.conv_tol = 1e-14
        #escf = smf.scf()

        #cc = pyscf.cc.RCCSD(smf)
        #cc.conv_tol = 1e-14
        #cc.conv_tol_normt = 1e-14
        #cc.max_cycle = 100
        #cc.verbose = 7
        #ecc, t1, t2 = cc.kernel()

        #from pyscf.cc import eom_rccsd
        #myeom = eom_rccsd.EOMIP(cc)
        #delta_ccsd, pt1, pt2 = eom_rccsd.get_t3p2_amplitude_contribution(cc,          cc.t1, cc.t2, eris=cc.ao2mo())
        #print('Correction to CCSD = %20.14e' % (delta_ccsd / np.prod(nk)))

    def test_eaccsd_311_n1_high_cost(self):
        L = 7.0
        n = 9
        cell = make_test_cell.test_cell_n1(L,[n]*3)
        nk = (3, 1, 1)

        abs_kpts = cell.make_kpts(nk, wrap_around=True)
        kmf = pbcscf.KRHF(cell, abs_kpts, exxdiv=None)
        kmf.conv_tol = 1e-14
        escf = kmf.scf()

        cc = pyscf.pbc.cc.kccsd_rhf.RCCSD(kmf)
        cc.conv_tol=1e-14
        cc.verbose = 7
        ecc, t1, t2 = cc.kernel()

        hf_311 = -0.92687629918229486
        cc_311 = -0.042702177586414237
        self.assertAlmostEqual(escf,hf_311, 9)
        self.assertAlmostEqual(ecc, cc_311, 6)

        ew, ev = cc.eaccsd(nroots=1, kptlist=[0], partition='mp')
        self.assertAlmostEqual(ew[0], 0.2824268531553734, 6)
        lew, lev = cc.leaccsd(nroots=1, kptlist=[0], partition='mp')
        self.assertAlmostEqual(lew[0], 0.2824268531553734, 6)

        ew, ev = cc.eaccsd(nroots=3, kptlist=[0])
        self.assertAlmostEqual(ew[0], 0.2637778931683522, 6)
        self.assertAlmostEqual(ew[1], 0.2891321218458044, 6)
        self.assertAlmostEqual(ew[2], 0.2891321218458051, 6)
        lew, lev = cc.leaccsd(nroots=3, kptlist=[0])
        self.assertAlmostEqual(lew[0], 0.2637778931683522, 6)
        self.assertAlmostEqual(lew[1], 0.2891321218458044, 6)
        self.assertAlmostEqual(lew[2], 0.2891321218458051, 6)

    def test_single_kpt(self):
        cell = pbcgto.Cell()
        cell.atom = '''
        H 0 0 0
        H 1 0 0
        H 0 1 0
        H 0 1 1
        '''
        cell.a = np.eye(3)*2
        cell.basis = [[0, [1.2, 1]], [1, [1.0, 1]]]
        cell.verbose = 0
        cell.build()

        kpts = cell.get_abs_kpts([.5,.5,.5]).reshape(1,3)
        mf = pbcscf.KRHF(cell, kpts=kpts).run(conv_tol=1e-9)
        kcc = pyscf.pbc.cc.kccsd_rhf.RCCSD(mf)
        e0 = kcc.kernel()[0]

        mf = pbcscf.RHF(cell, kpt=kpts[0]).run()
        mycc = pyscf.pbc.cc.RCCSD(mf)
        e1 = mycc.kernel()[0]
        self.assertAlmostEqual(e0, e1, 7)

    def test_frozen_n3(self):
        mesh = 5
        cell = make_test_cell.test_cell_n3([mesh]*3)
        nk = (1, 1, 2)
        ehf_bench = -8.348616843863795
        ecc_bench = -0.037920339437169

        abs_kpts = cell.make_kpts(nk, with_gamma_point=True)

        # RHF calculation
        kmf = pbcscf.KRHF(cell, abs_kpts, exxdiv=None)
        kmf.conv_tol = 1e-9
        ehf = kmf.scf()

        # KRCCSD calculation, equivalent to running supercell
        # calculation with frozen=[0,1,2] (if done with larger mesh)
        cc = pyscf.pbc.cc.kccsd_rhf.RCCSD(kmf, frozen=[[0],[0,1]])
        cc.diis_start_cycle = 1
        ecc, t1, t2 = cc.kernel()
        self.assertAlmostEqual(ehf, ehf_bench, 9)
        self.assertAlmostEqual(ecc, ecc_bench, 9)

    def _test_cu_metallic_nonequal_occ(self, kmf, cell, nk=[1,1,1]):
        assert cell.mesh == [7, 7, 7]
        ecc1_bench = -0.9646107739333411
        max_cycle = 5  # Too expensive to do more

        # The following calculation at full convergence gives -0.711071910294612
        # for a cell.mesh = [25, 25, 25].
        mycc = pyscf.pbc.cc.kccsd_rhf.RCCSD(kmf, frozen=0)
        mycc.diis_start_cycle = 1
        mycc.iterative_damping = 0.05
        mycc.max_cycle = max_cycle
        ecc1, t1, t2 = mycc.kernel()

        self.assertAlmostEqual(ecc1, ecc1_bench, 6)

    def _test_cu_metallic_frozen_occ(self, kmf, cell, nk=[1,1,1]):
        assert cell.mesh == [7, 7, 7]
        ecc2_bench = -0.7651806468801496
        max_cycle = 5

        # The following calculation at full convergence gives -0.6440448716452378
        # for a cell.mesh = [25, 25, 25].  It is equivalent to an RHF supercell [1, 1, 2]
        # calculation with frozen = [0, 3].
        mycc = pyscf.pbc.cc.kccsd_rhf.RCCSD(kmf, frozen=[[2, 3], [0, 1]])
        mycc.diis_start_cycle = 1
        mycc.iterative_damping = 0.05
        mycc.max_cycle = max_cycle
        ecc2, t1, t2 = mycc.kernel()

        self.assertAlmostEqual(ecc2, ecc2_bench, 6)

    def _test_cu_metallic_frozen_vir(self, kmf, cell, nk=[1,1,1]):
        assert cell.mesh == [7, 7, 7]
        ecc3_bench = -0.76794053711557086
        max_cycle = 200

        # The following calculation at full convergence gives -0.58688462599474
        # for a cell.mesh = [25, 25, 25].  It is equivalent to a supercell [1, 1, 2]
        # calculation with frozen = [0, 3, 35].
        mycc = pyscf.pbc.cc.kccsd_rhf.RCCSD(kmf, frozen=[[1, 17], [0]])
        mycc.diis_start_cycle = 1
        mycc.max_cycle = max_cycle
        mycc.iterative_damping = 0.05
        ecc3, t1, t2 = mycc.kernel()

        self.assertAlmostEqual(ecc3, ecc3_bench, 6)

        ew, ev = mycc.ipccsd(nroots=3, koopmans=True, kptlist=[1])
        self.assertAlmostEqual(ew[0][0], -3.028339571372944, 6)
        self.assertAlmostEqual(ew[0][1], -2.850636489429295, 6)
        self.assertAlmostEqual(ew[0][2], -2.801491561537961, 6)

        ew, ev = mycc.eaccsd(nroots=3, koopmans=True, kptlist=[1])
        self.assertAlmostEqual(ew[0][0], 3.266064683223669, 6)
        self.assertAlmostEqual(ew[0][1], 3.281390137070985, 6)
        self.assertAlmostEqual(ew[0][2], 3.426297911456726, 6)

        check_gamma = False  # Turn me on to run the supercell calculation!

        if check_gamma:
            from pyscf.pbc.tools.pbc import super_cell
            supcell = super_cell(cell, nk)
            kmf = pbcscf.RHF(supcell, exxdiv=None)
            ehf = kmf.scf()

            mycc = pyscf.pbc.cc.RCCSD(kmf, frozen=[0, 3, 35])
            mycc.max_cycle = max_cycle
            mycc.iterative_damping = 0.04
            ecc, t1, t2 = mycc.kernel()

            print('Gamma energy =', ecc/np.prod(nk))
            print('K-point energy =', ecc3_bench)

            ew, ev = mycc.ipccsd(nroots=5)
            # For cell mesh of [25, 25, 25], we get:
            #
            # EOM-CCSD root 0 E = -3.052456841625895
            # EOM-CCSD root 1 E = -2.989798972232893
            # EOM-CCSD root 2 E = -2.839646545189692
            # EOM-CCSD root 3 E = -2.836645046801352
            # EOM-CCSD root 4 E = -2.831020659800223

            ew, ev = mycc.eaccsd(nroots=5)
            # For cell mesh of [25, 25, 25], we get:
            #
            # EOM-CCSD root 0 E = 3.049774979170073
            # EOM-CCSD root 1 E = 3.104127952392612
            # EOM-CCSD root 2 E = 3.109435080273549
            # EOM-CCSD root 3 E = 3.139400145624026
            # EOM-CCSD root 4 E = 3.151896524990866

    def test_cu_metallic_high_cost(self):
        mesh = 7
        cell = make_test_cell.test_cell_cu_metallic([mesh]*3)
        nk = [1,1,2]
        ehf_bench = -52.5393701339723

        # KRHF calculation
        kmf = pbcscf.KRHF(cell, exxdiv=None)
        kmf.kpts = cell.make_kpts(nk, scaled_center=[0.0, 0.0, 0.0], wrap_around=True)
        kmf.conv_tol_grad = 1e-6  # Stricter tol needed for answer to agree with supercell
        ehf = kmf.scf()

        self.assertAlmostEqual(ehf, ehf_bench, 6)

        # Run CC calculations
        self._test_cu_metallic_nonequal_occ(kmf, cell, nk=nk)
        self._test_cu_metallic_frozen_occ(kmf, cell, nk=nk)
        self._test_cu_metallic_frozen_vir(kmf, cell, nk=nk)

    def test_ccsd_t_high_cost(self):
        n = 14
        cell = make_test_cell.test_cell_n3([n]*3)

        kpts = cell.make_kpts([1, 1, 2])
        kpts -= kpts[0]
        kmf = pbcscf.KRHF(cell, kpts=kpts, exxdiv=None)
        ehf = kmf.kernel()

        mycc = pyscf.pbc.cc.KRCCSD(kmf)
        ecc, t1, t2 = mycc.kernel()

        energy_t = kccsd_t_rhf.kernel(mycc)
        energy_t_bench = -0.00191443154358
        self.assertAlmostEqual(energy_t, energy_t_bench, 6)

if __name__ == '__main__':
    print("Full kpoint_rhf test")
    unittest.main()

