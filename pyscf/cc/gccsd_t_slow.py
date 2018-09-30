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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
GHF-CCSD(T) with spin-orbital integrals
'''

import time
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import gccsd

# spin-orbital formula
# JCP, 98, 8718
def kernel(cc, eris, t1=None, t2=None, max_memory=2000, verbose=logger.INFO):
    assert(isinstance(eris, gccsd._PhysicistsERIs))
    if t1 is None or t2 is None:
        t1, t2 = cc.t1, cc.t2

    nocc, nvir = t1.shape
    bcei = numpy.asarray(eris.ovvv).conj().transpose(3,2,1,0)
    majk = numpy.asarray(eris.ooov).conj().transpose(2,3,0,1)
    bcjk = numpy.asarray(eris.oovv).conj().transpose(2,3,0,1)

    mo_e = eris.fock.diagonal().real
    eia = mo_e[:nocc,None] - mo_e[nocc:]
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eia, eia, eia)

    t3c =(numpy.einsum('jkae,bcei->ijkabc', t2, bcei)
        - numpy.einsum('imbc,majk->ijkabc', t2, majk))
    t3c = t3c - t3c.transpose(0,1,2,4,3,5) - t3c.transpose(0,1,2,5,4,3)
    t3c = t3c - t3c.transpose(1,0,2,3,4,5) - t3c.transpose(2,1,0,3,4,5)
    t3c /= d3
#    e4 = numpy.einsum('ijkabc,ijkabc,ijkabc', t3c.conj(), d3, t3c) / 36
#    sia = numpy.einsum('jkbc,ijkabc->ia', eris.oovv, t3c) * .25
#    e5 = numpy.einsum('ia,ia', sia, t1.conj())
#    et = e4 + e5
#    return et
    t3d = numpy.einsum('ia,bcjk->ijkabc', t1, bcjk)
    t3d += numpy.einsum('ai,jkbc->ijkabc', eris.fock[nocc:,:nocc], t2)
    t3d = t3d - t3d.transpose(0,1,2,4,3,5) - t3d.transpose(0,1,2,5,4,3)
    t3d = t3d - t3d.transpose(1,0,2,3,4,5) - t3d.transpose(2,1,0,3,4,5)
    t3d /= d3
    et = numpy.einsum('ijkabc,ijkabc,ijkabc', (t3c+t3d).conj(), d3, t3c) / 36
    return et


def t3p2_corrected_amplitudes(cc, t1, t2, eris):
    """Calculates T1, T2 amplitudes corrected by second-order T3 contribution."""
    fock = eris.fock
    nocc, nvir = t1.shape

    fov = fock[:nocc, nocc:]
    foo = fock[:nocc, :nocc].diagonal()
    fvv = fock[nocc:, nocc:].diagonal()

    oovv = numpy.asarray(eris.oovv)
    ovvv = numpy.asarray(eris.ovvv)
    ooov = numpy.asarray(eris.ooov)
    vooo = numpy.asarray(ooov).conj().transpose(3, 2, 1, 0)
    vvvo = numpy.asarray(ovvv).conj().transpose(3, 2, 1, 0)

    ccsd_energy = gccsd.energy(cc, t1, t2, eris)

    # Slow, memory-intensize
    t3 = lib.einsum('bcdk,ijad->ijkabc', vvvo, t2)
    t3 -= lib.einsum('cmkj,imab->ijkabc', vooo, t2)
    # P(ijk)
    t3 = (t3 + t3.transpose(1,2,0,3,4,5) +
               t3.transpose(2,0,1,3,4,5))
    # P(abc)
    t3 = (t3 + t3.transpose(0,1,2,4,5,3) +
               t3.transpose(0,1,2,5,3,4))
    eia = foo[:,None] - fvv[None,:]
    eijab = eia[:,None,:,None] + eia[None,:,None,:]
    eijkabc = eijab[:,:,None,:,:,None] + eia[None,None,:,None,None,:]
    t3 /= eijkabc

    eijk = foo[:, None, None] + foo[None, :, None] + foo[None, None, :]
    eia = foo[:, None] - fvv[None, :]
    eijab = eia[:, None, :, None] + eia[None, :, None, :]

    # Correction to t1
    pt1 = 0.25 * lib.einsum('mnef,imnaef->ia', oovv, t3)

    # Correction to t2
    pt2 = lib.einsum('ijmabe,me->ijab', t3, fov)
    tmp = 0.5 * lib.einsum('ijmaef,mbfe->ijab', t3, ovvv)
    tmp = tmp - tmp.transpose(0, 1, 3, 2)  # P(ab)
    pt2 += tmp
    tmp = - 0.5 * lib.einsum('imnabe,mnje->ijab', t3, ooov)
    tmp = tmp - tmp.transpose(1, 0, 2, 3)  # P(ij)
    pt2 += tmp

    eia = foo[:, None] - fvv[None, :]
    eijab = eia[:, None, :, None] + eia[None, :, None, :]

    pt1 /= eia
    pt2 /= eijab

    pt1 += t1
    pt2 += t2

    delta_ccsd_energy = gccsd.energy(cc, pt1, pt2, eris) - ccsd_energy
    return delta_ccsd_energy, pt1, pt2


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import cc

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -.957 , .587)],
        [1 , (0.2,  .757 , .487)]]

    mol.basis = '631g'
    mol.build()
    mf = scf.RHF(mol).run(conv_tol=1e-1)
    mycc = cc.CCSD(mf).set(conv_tol=1e-11).run()
    et = mycc.ccsd_t()

    mycc = cc.GCCSD(scf.addons.convert_to_ghf(mf)).set(conv_tol=1e-11).run()
    eris = mycc.ao2mo()
    print(kernel(mycc, eris) - et)

    gccsd_t_a = t3p2_corrected_amplitudes(mycc, mycc.t1, mycc.t2, eris)[0]
