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

import time
from functools import reduce
import numpy as np
import h5py

from pyscf import lib
import pyscf.ao2mo
from pyscf.lib import logger
import pyscf.cc
import pyscf.cc.ccsd
from pyscf.pbc import scf
from pyscf.pbc.mp.kmp2 import (get_frozen_mask, get_nocc, get_nmo,
                               padded_mo_coeff, padding_k_idx)
from pyscf.pbc.cc import kintermediates_rhf as imdk
from pyscf.lib.parameters import LOOSE_ZERO_TOL, LARGE_DENOM
from pyscf.lib import linalg_helper
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.lib.kpts_helper import member, gamma_point
from pyscf import __config__
from itertools import product

# einsum = np.einsum
einsum = lib.einsum

# This is restricted (R)CCSD
# Ref: Hirata, et al., J. Chem. Phys. 120, 2581 (2004)

kernel = pyscf.cc.ccsd.kernel


def get_normt_diff(cc, t1, t2, t1new, t2new):
    '''Calculates norm(t1 - t1new) + norm(t2 - t2new).'''
    return np.linalg.norm(t1new - t1) + np.linalg.norm(t2new - t2)

def atleast_3d(array):
    if len(array.shape) == 0:
        result = array.reshape(1,1,1)
    elif len(array.shape) == 1:
        result = array[None, None, :]
    elif len(array.shape) == 2:
        result = array[None, :, :]
    else:
        result = array
    return result

def update_amps(cc, t1, t2, eris):
    time0 = time1 = time.clock(), time.time()
    log = logger.Logger(cc.stdout, cc.verbose)
    nkpts, nocc, nvir = t1.shape
    fock = eris.fock
    mo_e_o = [e[:nocc] for e in eris.mo_energy]
    mo_e_v = [e[nocc:] + cc.level_shift for e in eris.mo_energy]

    # Get location of padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(cc, kind="split")

    fov = fock[:, :nocc, nocc:]
    foo = fock[:, :nocc, :nocc]
    fvv = fock[:, nocc:, nocc:]

    kconserv = cc.khelper.kconserv

    Foo = imdk.cc_Foo(t1, t2, eris, kconserv)
    Fvv = imdk.cc_Fvv(t1, t2, eris, kconserv)
    Fov = imdk.cc_Fov(t1, t2, eris, kconserv)
    Loo = imdk.Loo(t1, t2, eris, kconserv)
    Lvv = imdk.Lvv(t1, t2, eris, kconserv)

    # Move energy terms to the other side
    for k in range(nkpts):
        Foo[k][np.diag_indices(nocc)] -= mo_e_o[k]
        Fvv[k][np.diag_indices(nvir)] -= mo_e_v[k]
        Loo[k][np.diag_indices(nocc)] -= mo_e_o[k]
        Lvv[k][np.diag_indices(nvir)] -= mo_e_v[k]
    time1 = log.timer_debug1('intermediates', *time1)

    # T1 equation
    t1new = np.array(fov).astype(t1.dtype).conj()

    for ka in range(nkpts):
        ki = ka
        # kc == ki; kk == ka
        t1new[ka] += -2. * einsum('kc,ka,ic->ia', fov[ki], t1[ka], t1[ki])
        t1new[ka] += einsum('ac,ic->ia', Fvv[ka], t1[ki])
        t1new[ka] += -einsum('ki,ka->ia', Foo[ki], t1[ka])

        tau_term = np.empty((nkpts, nocc, nocc, nvir, nvir), dtype=t1.dtype)
        for kk in range(nkpts):
            tau_term[kk] = 2 * t2[kk, ki, kk] - t2[ki, kk, kk].transpose(1, 0, 2, 3)
        tau_term[ka] += einsum('ic,ka->kica', t1[ki], t1[ka])

        for kk in range(nkpts):
            kc = kk
            t1new[ka] += einsum('kc,kica->ia', Fov[kc], tau_term[kk])

            t1new[ka] += einsum('akic,kc->ia', 2 * eris.voov[ka, kk, ki], t1[kc])
            t1new[ka] += einsum('kaic,kc->ia', -eris.ovov[kk, ka, ki], t1[kc])

            for kc in range(nkpts):
                kd = kconserv[ka, kc, kk]

                Svovv = 2 * eris.vovv[ka, kk, kc] - eris.vovv[ka, kk, kd].transpose(0, 1, 3, 2)
                tau_term_1 = t2[ki, kk, kc].copy()
                if ki == kc and kk == kd:
                    tau_term_1 += einsum('ic,kd->ikcd', t1[ki], t1[kk])
                t1new[ka] += einsum('akcd,ikcd->ia', Svovv, tau_term_1)

                # kk - ki + kl = kc
                #  => kl = ki - kk + kc
                kl = kconserv[ki, kk, kc]
                Sooov = 2 * eris.ooov[kk, kl, ki] - eris.ooov[kl, kk, ki].transpose(1, 0, 2, 3)
                tau_term_1 = t2[kk, kl, ka].copy()
                if kk == ka and kl == kc:
                    tau_term_1 += einsum('ka,lc->klac', t1[ka], t1[kc])
                t1new[ka] += -einsum('klic,klac->ia', Sooov, tau_term_1)
    time1 = log.timer_debug1('t1', *time1)

    # T2 equation
    t2new = np.empty_like(t2)
    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        t2new[ki, kj, ka] = eris.oovv[ki, kj, ka].conj()

    mem_now = lib.current_memory()[0]
    if (nocc ** 4 * nkpts ** 3) * 16 / 1e6 + mem_now < cc.max_memory * .9:
        Woooo = imdk.cc_Woooo(t1, t2, eris, kconserv)
    else:
        fimd = lib.H5TmpFile()
        Woooo = fimd.create_dataset('oooo', (nkpts, nkpts, nkpts, nocc, nocc, nocc, nocc), t1.dtype.char)
        Woooo = imdk.cc_Woooo(t1, t2, eris, kconserv, Woooo)

    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        # Chemist's notation for momentum conserving t2(ki,kj,ka,kb)
        kb = kconserv[ki, ka, kj]

        t2new_tmp = np.zeros((nocc, nocc, nvir, nvir), dtype=t2.dtype)
        for kl in range(nkpts):
            kk = kconserv[kj, kl, ki]
            tau_term = t2[kk, kl, ka].copy()
            if kl == kb and kk == ka:
                tau_term += einsum('ic,jd->ijcd', t1[ka], t1[kb])
            t2new_tmp += 0.5 * einsum('klij,klab->ijab', Woooo[kk, kl, ki], tau_term)
        t2new[ki, kj, ka] += t2new_tmp
        t2new[kj, ki, kb] += t2new_tmp.transpose(1, 0, 3, 2)
    Woooo = None
    fimd = None
    time1 = log.timer_debug1('t2 oooo', *time1)

    # einsum('abcd,ijcd->ijab', Wvvvv, tau)
    add_vvvv_(cc, t2new, t1, t2, eris)
    time1 = log.timer_debug1('t2 vvvv', *time1)

    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki, ka, kj]

        t2new_tmp = einsum('ac,ijcb->ijab', Lvv[ka], t2[ki, kj, ka])
        t2new_tmp += einsum('ki,kjab->ijab', -Loo[ki], t2[ki, kj, ka])

        kc = kconserv[ka, ki, kb]
        tmp2 = np.asarray(eris.vovv[kc, ki, kb]).transpose(3, 2, 1, 0).conj() \
               - einsum('kbic,ka->abic', eris.ovov[ka, kb, ki], t1[ka])
        t2new_tmp += einsum('abic,jc->ijab', tmp2, t1[kj])

        kk = kconserv[ki, ka, kj]
        tmp2 = np.asarray(eris.ooov[kj, ki, kk]).transpose(3, 2, 1, 0).conj() \
               + einsum('akic,jc->akij', eris.voov[ka, kk, ki], t1[kj])
        t2new_tmp -= einsum('akij,kb->ijab', tmp2, t1[kb])
        t2new[ki, kj, ka] += t2new_tmp
        t2new[kj, ki, kb] += t2new_tmp.transpose(1, 0, 3, 2)

    mem_now = lib.current_memory()[0]
    if (nocc ** 2 * nvir ** 2 * nkpts ** 3) * 16 / 1e6 * 2 + mem_now < cc.max_memory * .9:
        Wvoov = imdk.cc_Wvoov(t1, t2, eris, kconserv)
        Wvovo = imdk.cc_Wvovo(t1, t2, eris, kconserv)
    else:
        fimd = lib.H5TmpFile()
        Wvoov = fimd.create_dataset('voov', (nkpts, nkpts, nkpts, nvir, nocc, nocc, nvir), t1.dtype.char)
        Wvovo = fimd.create_dataset('vovo', (nkpts, nkpts, nkpts, nvir, nocc, nvir, nocc), t1.dtype.char)
        Wvoov = imdk.cc_Wvoov(t1, t2, eris, kconserv, Wvoov)
        Wvovo = imdk.cc_Wvovo(t1, t2, eris, kconserv, Wvovo)

    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki, ka, kj]
        t2new_tmp = np.zeros((nocc, nocc, nvir, nvir), dtype=t2.dtype)
        for kk in range(nkpts):
            kc = kconserv[ka, ki, kk]
            tmp_voov = 2. * Wvoov[ka, kk, ki] - Wvovo[ka, kk, kc].transpose(0, 1, 3, 2)
            t2new_tmp += einsum('akic,kjcb->ijab', tmp_voov, t2[kk, kj, kc])

            kc = kconserv[ka, ki, kk]
            t2new_tmp -= einsum('akic,kjbc->ijab', Wvoov[ka, kk, ki], t2[kk, kj, kb])

            kc = kconserv[kk, ka, kj]
            t2new_tmp -= einsum('bkci,kjac->ijab', Wvovo[kb, kk, kc], t2[kk, kj, ka])

        t2new[ki, kj, ka] += t2new_tmp
        t2new[kj, ki, kb] += t2new_tmp.transpose(1, 0, 3, 2)
    Wvoov = Wvovo = None
    fimd = None
    time1 = log.timer_debug1('t2 voov', *time1)

    for ki in range(nkpts):
        ka = ki
        # Remove zero/padded elements from denominator
        eia = LARGE_DENOM * np.ones((nocc, nvir), dtype=eris.mo_energy[0].dtype)
        n0_ovp_ia = np.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
        eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[ka])[n0_ovp_ia]
        t1new[ki] /= eia

    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki, ka, kj]
        # For LARGE_DENOM, see t1new update above
        eia = LARGE_DENOM * np.ones((nocc, nvir), dtype=eris.mo_energy[0].dtype)
        n0_ovp_ia = np.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
        eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[ka])[n0_ovp_ia]

        ejb = LARGE_DENOM * np.ones((nocc, nvir), dtype=eris.mo_energy[0].dtype)
        n0_ovp_jb = np.ix_(nonzero_opadding[kj], nonzero_vpadding[kb])
        ejb[n0_ovp_jb] = (mo_e_o[kj][:,None] - mo_e_v[kb])[n0_ovp_jb]
        eijab = eia[:, None, :, None] + ejb[:, None, :]

        t2new[ki, kj, ka] /= eijab

    time0 = log.timer_debug1('update t1 t2', *time0)

    return t1new, t2new


# TODO: pull these 3 methods to pyscf.util and make tests
def describe_nested(data):
    """
    Retrieves the description of a nested array structure.
    Args:
        data (iterable): a nested structure to describe;

    Returns:
        - A nested structure where numpy arrays are replaced by their shapes;
        - The overall number of scalar elements;
        - The common data type;
    """
    if isinstance(data, np.ndarray):
        return data.shape, data.size, data.dtype
    elif isinstance(data, (list, tuple)):
        total_size = 0
        struct = []
        dtype = None
        for i in data:
            i_struct, i_size, i_dtype = describe_nested(i)
            struct.append(i_struct)
            total_size += i_size
            if dtype is not None and i_dtype is not None and i_dtype != dtype:
                raise ValueError("Several different numpy dtypes encountered: %s and %s" %
                                 (str(dtype), str(i_dtype)))
            dtype = i_dtype
        return struct, total_size, dtype
    else:
        raise ValueError("Unknown object to describe: %s" % str(data))


def nested_to_vector(data, destination=None, offset=0):
    """
    Puts any nested iterable into a vector.
    Args:
        data (Iterable): a nested structure of numpy arrays;
        destination (array): array to store the data to;
        offset (int): array offset;

    Returns:
        If destination is not specified, returns a vectorized data and the original nested structure to restore the data
        into its original form. Otherwise returns a new offset.
    """
    if destination is None:
        struct, total_size, dtype = describe_nested(data)
        destination = np.empty(total_size, dtype=dtype)
        rtn = True
    else:
        rtn = False

    if isinstance(data, np.ndarray):
        destination[offset:offset + data.size] = data.ravel()
        offset += data.size
    elif isinstance(data, (list, tuple)):
        for i in data:
            offset = nested_to_vector(i, destination, offset)
    else:
        raise ValueError("Unknown object to vectorize: %s" % str(data))

    if rtn:
        return destination, struct
    else:
        return offset


def vector_to_nested(vector, struct, copy=True, ensure_size_matches=True):
    """
    Retrieves the original nested structure from the vector.
    Args:
        vector (array): a vector to decompose;
        struct (Iterable): a nested structure with arrays' shapes;
        copy (bool): whether to copy arrays;
        ensure_size_matches (bool): if True, ensures all elements from the vector are used;

    Returns:
        A nested structure with numpy arrays and, if `ensure_size_matches=False`, the number of vector elements used.
    """
    if len(vector.shape) != 1:
        raise ValueError("Only vectors accepted, got: %s" % repr(vector.shape))

    if isinstance(struct, tuple):
        expected_size = np.prod(struct)
        if ensure_size_matches:
            if vector.size != expected_size:
                raise ValueError("Structure size mismatch: expected %s = %d, found %d" %
                                 (repr(struct), expected_size, vector.size,))
        if len(vector) < expected_size:
            raise ValueError("Additional %d = (%d = %s) - %d vector elements are required" %
                             (expected_size - len(vector), expected_size,
                              repr(struct), len(vector),))
        a = vector[:expected_size].reshape(struct)
        if copy:
            a = a.copy()

        if ensure_size_matches:
            return a
        else:
            return a, expected_size

    elif isinstance(struct, list):
        offset = 0
        result = []
        for i in struct:
            nested, size = vector_to_nested(vector[offset:], i, copy=copy, ensure_size_matches=False)
            offset += size
            result.append(nested)

        if ensure_size_matches:
            if vector.size != offset:
                raise ValueError("%d additional elements found" % (vector.size - offset))
            return result
        else:
            return result, offset

    else:
        raise ValueError("Unknown object to compose: %s" % (str(struct)))


def energy_full(cc, t1, t2, eris):
    '''Energy function for full-rank t2.'''
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.khelper.kconserv
    fock = eris.fock
    e = 0.0 + 1j * 0.0
    for ki in range(nkpts):
        e += 2 * einsum('ia,ia', fock[ki, :nocc, nocc:], t1[ki])
    tau = t1t1 = np.zeros(shape=t2.shape, dtype=t2.dtype)
    for ki in range(nkpts):
        ka = ki
        for kj in range(nkpts):
            # kb = kj
            t1t1[ki, kj, ka] = einsum('ia,jb->ijab', t1[ki], t1[kj])
    tau += t2
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki, ka, kj]
                e += 2 * einsum('ijab,ijab', tau[ki, kj, ka], eris.oovv[ki, kj, ka])
                e += -einsum('ijab,ijba', tau[ki, kj, ka], eris.oovv[ki, kj, kb])
    e /= nkpts
    if abs(e.imag) > 1e-4:
        logger.warn(cc, 'Non-zero imaginary part found in KRCCSD energy %s', e)
    return e.real


def energy_tril(cc, t1, t2, eris):
    '''Energy function for lower-triangular t2.'''
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.khelper.kconserv
    fock = eris.fock
    e = 0.0 + 1j * 0.0
    for ki in range(nkpts):
        e += 2 * einsum('ia,ia', fock[ki, :nocc, nocc:], t1[ki])
    tau = t1t1 = np.zeros(shape=t2.shape, dtype=t2.dtype)
    for kj in range(nkpts):
        for ki in range(kj+1):
            ka = ki  # kb = kj
            idx = (kj*(kj+1)//2) + ki
            t1t1[idx, ka] = einsum('ia,jb->ijab', t1[ki], t1[kj])
    tau += t2
    for kj in range(nkpts):
        for ki in range(kj+1):
            idx = (kj*(kj+1))//2 + ki
            for ka in range(nkpts):
                kb = kconserv[ki, ka, kj]
                e += 2 * einsum('ijab,ijab', tau[idx, ka], eris.oovv[ki, kj, ka])
                e += -einsum('ijab,ijba', tau[idx, ka], eris.oovv[ki, kj, kb])

                if ki != kj:
                    e += 2 * einsum('ijba,ijba', tau[idx, kb], eris.oovv[ki, kj, kb])
                    e += -einsum('ijba,ijab', tau[idx, kb], eris.oovv[ki, kj, ka])
    e /= nkpts
    if abs(e.imag) > 1e-4:
        logger.warn(cc, 'Non-zero imaginary part found in KRCCSD energy %s', e)
    return e.real


def energy(cc, t1, t2, eris):
    nkpts, nocc, nvir = t1.shape
    if len(t2.shape) == 7 and t2.shape[:2] == (nkpts,nkpts):
        return energy_full(cc, t1, t2, eris)
    elif len(t2.shape) == 6 and t2.shape[:2] == (nkpts*(nkpts+1)//2,nkpts):
        return energy_tril(cc, t1, t2, eris)
    else:
        raise ValueError('No conversion known for shape = %s' % t2.shape)


def add_vvvv_(cc, Ht2, t1, t2, eris):
    nocc = cc.nocc
    nmo = cc.nmo
    nvir = nmo - nocc
    nkpts = cc.nkpts
    kconserv = cc.khelper.kconserv

    mem_now = lib.current_memory()[0]
    if cc.direct and hasattr(eris, 'Lpv'):
        #: If memory is not enough to hold eris.Lpv
        #:def get_Wvvvv(ka, kb, kc):
        #:    kd = kconserv[ka,kc,kb]
        #:    v = cc._scf.with_df.ao2mo([eris.mo_coeff[k] for k in [ka,kc,kb,kd]],
        #:                              cc.kpts[[ka,kc,kb,kd]]).reshape([nmo]*4)
        #:    Wvvvv  = lib.einsum('kcbd,ka->abcd', v[:nocc,nocc:,nocc:,nocc:], -t1[ka])
        #:    Wvvvv += lib.einsum('ackd,kb->abcd', v[nocc:,nocc:,:nocc,nocc:], -t1[kb])
        #:    Wvvvv += v[nocc:,nocc:,nocc:,nocc:].transpose(0,2,1,3)
        #:    Wvvvv *= (1./nkpts)
        #:    return Wvvvv
        def get_Wvvvv(ka, kb, kc):
            kd = kconserv[ka, kc, kb]
            Lbd = (eris.Lpv[kb, kd, :, nocc:] -
                   lib.einsum('Lkd,kb->Lbd', eris.Lpv[kb, kd, :, :nocc], t1[kb]))
            Wvvvv = lib.einsum('Lac,Lbd->abcd', eris.Lpv[ka, kc, :, nocc:], Lbd)
            Lbd = None
            kcbd = lib.einsum('Lkc,Lbd->kcbd', eris.Lpv[ka, kc, :, :nocc],
                              eris.Lpv[kb, kd, :, nocc:])
            Wvvvv -= lib.einsum('kcbd,ka->abcd', kcbd, t1[ka])
            Wvvvv *= (1. / nkpts)
            return Wvvvv

    elif (nvir ** 4 * nkpts ** 3) * 16 / 1e6 + mem_now < cc.max_memory * .9:
        _Wvvvv = imdk.cc_Wvvvv(t1, t2, eris, kconserv)

        def get_Wvvvv(ka, kb, kc):
            return _Wvvvv[ka, kb, kc]
    else:
        fimd = lib.H5TmpFile()
        _Wvvvv = fimd.create_dataset('vvvv', (nkpts,nkpts,nkpts,nvir,nvir,nvir,nvir), t1.dtype.char)
        _Wvvvv = imdk.cc_Wvvvv(t1, t2, eris, kconserv, _Wvvvv)
        def get_Wvvvv(ka, kb, kc):
            return _Wvvvv[ka, kb, kc]

    #:Ps = kconserve_pmatrix(cc.nkpts, cc.khelper.kconserv)
    #:Wvvvv = einsum('xyzakcd,ykb->xyzabcd', eris.vovv, -t1)
    #:Wvvvv = Wvvvv + einsum('xyzabcd,xyzw->yxwbadc', Wvvvv, Ps)
    #:Wvvvv += eris.vvvv
    #:
    #:tau = t2.copy()
    #:idx = np.arange(nkpts)
    #:tau[idx,:,idx] += einsum('xic,yjd->xyijcd', t1, t1)
    #:Ht2 += einsum('xyuijcd,zwuabcd,xyuv,zwuv->xyzijab', tau, Wvvvv, Ps, Ps)
    for ka, kb, kc in kpts_helper.loop_kkk(nkpts):
        kd = kconserv[ka, kc, kb]
        Wvvvv = get_Wvvvv(ka, kb, kc)
        for ki in range(nkpts):
            kj = kconserv[ka, ki, kb]
            tau = t2[ki, kj, kc].copy()
            if ki == kc and kj == kd:
                tau += np.einsum('ic,jd->ijcd', t1[ki], t1[kj])
            Ht2[ki, kj, ka] += lib.einsum('abcd,ijcd->ijab', Wvvvv, tau)
    fimd = None
    return Ht2


# Ps is Permutation transformation matrix
# The physical meaning of Ps matrix is the conservation of moment.
# Given the four indices in Ps, the element shows whether moment conservation
# holds (1) or not (0)
def kconserve_pmatrix(nkpts, kconserv):
    Ps = np.zeros((nkpts, nkpts, nkpts, nkpts))
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                # Chemist's notation for momentum conserving t2(ki,kj,ka,kb)
                kb = kconserv[ki, ka, kj]
                Ps[ki, kj, ka, kb] = 1
    return Ps


class RCCSD(pyscf.cc.ccsd.CCSD):
    max_space = getattr(__config__, 'pbc_cc_kccsd_rhf_KRCCSD_max_space', 20)

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        assert (isinstance(mf, scf.khf.KSCF))
        pyscf.cc.ccsd.CCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.kpts = mf.kpts
        self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts)
        self.made_ee_imds = False
        self.made_ip_imds = False
        self.made_ea_imds = False
        self.ip_partition = None
        self.ea_partition = None
        self.direct = True  # If possible, use GDF to compute Wvvvv on-the-fly

        keys = set(['kpts', 'khelper', 'made_ee_imds',
                    'made_ip_imds', 'made_ea_imds', 'ip_partition',
                    'ea_partition', 'max_space', 'direct'])
        self._keys = self._keys.union(keys)

    @property
    def nkpts(self):
        return len(self.kpts)

    get_normt_diff = get_normt_diff
    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    def dump_flags(self):
        return pyscf.cc.ccsd.CCSD.dump_flags(self)

    @property
    def ccsd_vector_desc(self):
        """Description of the ground-state vector."""
        nvir = self.nmo - self.nocc
        return [(self.nkpts, self.nocc, nvir), (self.nkpts,) * 3 + (self.nocc,) * 2 + (nvir,) * 2]

    def amplitudes_to_vector(self, t1, t2):
        """Ground state amplitudes to a vector."""
        return nested_to_vector((t1, t2))[0]

    def vector_to_amplitudes(self, vec):
        """Ground state vector to apmplitudes."""
        return vector_to_nested(vec, self.ccsd_vector_desc)

    @property
    def ip_vector_desc(self):
        """Description of the IP vector."""
        return [(self.nocc,), (self.nkpts, self.nkpts, self.nocc, self.nocc, self.nmo - self.nocc)]

    def ip_amplitudes_to_vector(self, t1, t2):
        """Ground state amplitudes to a vector."""
        return nested_to_vector((t1, t2))[0]

    def ip_vector_to_amplitudes(self, vec):
        """Ground state vector to apmplitudes."""
        return vector_to_nested(vec, self.ip_vector_desc)

    @property
    def ea_vector_desc(self):
        """Description of the EA vector."""
        nvir = self.nmo - self.nocc
        return [(nvir,), (self.nkpts, self.nkpts, self.nocc, nvir, nvir)]

    def ea_amplitudes_to_vector(self, t1, t2):
        """Ground state amplitudes to a vector."""
        return nested_to_vector((t1, t2))[0]

    def ea_vector_to_amplitudes(self, vec):
        """Ground state vector to apmplitudes."""
        return vector_to_nested(vec, self.ea_vector_desc)

    def init_amps(self, eris):
        time0 = time.clock(), time.time()
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        t1 = np.zeros((nkpts,nocc,nvir), dtype=eris.fock.dtype)
        t2 = np.empty((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=eris.fock.dtype)
        mo_e_o = [eris.mo_energy[k][:nocc] for k in range(nkpts)]
        mo_e_v = [eris.mo_energy[k][nocc:] for k in range(nkpts)]

        # Get location of padded elements in occupied and virtual space
        nonzero_opadding, nonzero_vpadding = padding_k_idx(self, kind="split")

        emp2 = 0
        kconserv = self.khelper.kconserv
        touched = np.zeros((nkpts, nkpts, nkpts), dtype=bool)
        for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
            if touched[ki, kj, ka]:
                continue

            kb = kconserv[ki, ka, kj]
            # For discussion of LARGE_DENOM, see t1new update above
            eia = LARGE_DENOM * np.ones((nocc, nvir), dtype=eris.mo_energy[0].dtype)
            n0_ovp_ia = np.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
            eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[ka])[n0_ovp_ia]

            ejb = LARGE_DENOM * np.ones((nocc, nvir), dtype=eris.mo_energy[0].dtype)
            n0_ovp_jb = np.ix_(nonzero_opadding[kj], nonzero_vpadding[kb])
            ejb[n0_ovp_jb] = (mo_e_o[kj][:,None] - mo_e_v[kb])[n0_ovp_jb]
            eijab = eia[:, None, :, None] + ejb[:, None, :]

            eris_ijab = eris.oovv[ki, kj, ka]
            eris_ijba = eris.oovv[ki, kj, kb]
            t2[ki, kj, ka] = eris_ijab.conj() / eijab
            woovv = 2 * eris_ijab - eris_ijba.transpose(0, 1, 3, 2)
            emp2 += np.einsum('ijab,ijab', t2[ki, kj, ka], woovv)

            if ka != kb:
                eijba = eijab.transpose(0, 1, 3, 2)
                t2[ki, kj, kb] = eris_ijba.conj() / eijba
                woovv = 2 * eris_ijba - eris_ijab.transpose(0, 1, 3, 2)
                emp2 += np.einsum('ijab,ijab', t2[ki, kj, kb], woovv)

            touched[ki, kj, ka] = touched[ki, kj, kb] = True

        self.emp2 = emp2.real / nkpts
        logger.info(self, 'Init t2, MP2 energy (with fock eigenvalue shift) = %.15g', self.emp2)
        logger.timer(self, 'init mp2', *time0)
        return self.emp2, t1, t2

    energy = energy
    update_amps = update_amps

    def kernel(self, t1=None, t2=None, eris=None, mbpt2=False):
        return self.ccsd(t1, t2, eris, mbpt2=mbpt2)

    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False):
        '''Ground-state CCSD.

        Kwargs:
            mbpt2 : bool
                Use one-shot MBPT2 approximation to CCSD.
        '''
        self.dump_flags()
        if eris is None:
            # eris = self.ao2mo()
            eris = self.ao2mo(self.mo_coeff)
        self.eris = eris
        if mbpt2:
            cctyp = 'MBPT2'
            self.e_corr, self.t1, self.t2 = self.init_amps(eris)
            return self.e_corr, self.t1, self.t2

        cctyp = 'CCSD'
        self.converged, self.e_corr, self.t1, self.t2 = \
            kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                   tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                   verbose=self.verbose)
        self._finalize()
        return self.e_corr, self.t1, self.t2

    def ao2mo(self, mo_coeff=None):
        return _ERIS(self, mo_coeff)

    def vector_size_ip(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts

        size = nocc + nkpts ** 2 * nocc ** 2 * nvir
        return size

    def ipccsd(self, nroots=1, left=False, koopmans=False, guess=None, partition=None,
               kptlist=None, **kwargs):
        '''Calculate (N-1)-electron charged excitations via IP-EOM-CCSD.

        Kwargs:
            nroots : int
                Number of roots (eigenvalues) requested per k-point
            left : bool
                Whether to calculate the left amplitudes.  Defaults to calculating right
                amplitudes.
            koopmans : bool
                Calculate Koopmans'-like (quasiparticle) excitations only, targeting via
                overlap.
            guess : list of ndarray
                List of guess vectors to use for targeting via overlap.
            partition : bool or str
                Use a matrix-partitioning for the doubles-doubles block.
                Can be None, 'mp' (Moller-Plesset, i.e. orbital energies on the diagonal),
                or 'full' (full diagonal elements).
            kptlist : list
                List of k-point indices for which eigenvalues are requested.
        '''
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        if kptlist is None:
            kptlist = range(nkpts)
        size = self.vector_size_ip()
        for k, kshift in enumerate(kptlist):
            nfrozen = np.sum(self.mask_frozen_ip(np.zeros(size, dtype=int), kshift, const=1))
            nroots = min(nroots, size - nfrozen)
        if partition:
            partition = partition.lower()
            assert partition in ['mp', 'full']
        self.ip_partition = partition
        evals = np.zeros((len(kptlist), nroots), np.float)
        evecs = np.zeros((len(kptlist), nroots, size), np.complex)

        if left:
            matvec = lambda _arg: self.lipccsd_matvec(_arg, kshift, **kwargs)
        else:
            matvec = lambda _arg: self.ipccsd_matvec(_arg, kshift, **kwargs)

        ## Initialize intermediates
        #with_t3p2 = kwargs.pop('with_t3p2', False)
        #copy_amps_t3p2 = kwargs.pop('copy_amps_t3p2', True)
        #with_t3p2_imds = kwargs.pop('with_t3p2_imds', False)
        #if kwargs:
        #    raise TypeError('Unexpected **kwargs: %r' % kwargs)
        #if not hasattr(self, 'imds'):
        #    self.imds = _IMDS(self, with_t3p2=with_t3p2, with_t3p2_imds=with_t3p2_imds)
        if not hasattr(self, 'imds'):
            self.imds = _IMDS(self, with_t3p2=kwargs.get('with_t3p2', False),
                                    with_t3p2_imds=kwargs.get('with_t3p2_imds', False))

        for k, kshift in enumerate(kptlist):
            adiag = self.ipccsd_diag(kshift, **kwargs)
            adiag = self.mask_frozen_ip(adiag, kshift, const=LARGE_DENOM)
            if partition == 'full':
                self._ipccsd_diag_matrix2 = self.ip_vector_to_amplitudes(adiag)[1]

            user_guess = False
            if guess:
                user_guess = True
                assert len(guess[k]) == nroots
                for g in guess[k]:
                    assert g.size == size
            else:
                guess = []
                if koopmans:
                    # Get location of padded elements in occupied and virtual space
                    nonzero_opadding = padding_k_idx(self, kind="split")[0][kshift]

                    for n in nonzero_opadding[::-1][:nroots]:
                        g = np.zeros(size)
                        g[n] = 1.0
                        g = self.mask_frozen_ip(g, kshift, const=0.0)
                        guess.append(g)
                else:
                    idx = adiag.argsort()[:nroots]
                    for i in idx:
                        g = np.zeros(size)
                        g[i] = 1.0
                        g = self.mask_frozen_ip(g, kshift, const=0.0)
                        guess.append(g)

            def precond(r, e0, x0):
                return r / (e0 - adiag + 1e-12)

            eig = linalg_helper.eig
            if user_guess or koopmans:
                def pickeig(w, v, nr, envs):
                    x0 = linalg_helper._gen_x0(envs['v'], envs['xs'])
                    idx = np.argmax(np.abs(np.dot(np.array(guess).conj(), np.array(x0).T)), axis=1)
                    return lib.linalg_helper._eigs_cmplx2real(w, v, idx)

                evals_k, evecs_k = eig(matvec, guess, precond, pick=pickeig,
                                       tol=self.conv_tol, max_cycle=self.max_cycle,
                                       max_space=self.max_space, nroots=nroots, verbose=self.verbose)
            else:
                evals_k, evecs_k = eig(matvec, guess, precond,
                                       tol=self.conv_tol, max_cycle=self.max_cycle,
                                       max_space=self.max_space, nroots=nroots, verbose=self.verbose)
            if not user_guess:
                guess = None

            evals_k = evals_k.real
            evals[k] = evals_k
            evecs[k] = evecs_k

            if nroots == 1:
                evals_k, evecs_k = [evals_k], [evecs_k]

            for n, en, vn in zip(range(nroots), evals_k, evecs_k):
                r1, r2 = self.ip_vector_to_amplitudes(vn)
                qp_weight = np.linalg.norm(r1) ** 2
                logger.info(self, 'EOM root %d E = %.16g  qpwt = %0.6g',
                            n, en, qp_weight)
        log.timer('EOM-CCSD', *cput0)
        self.eip = evals
        return self.eip, evecs

    def ipccsd_matvec(self, vector, kshift, imds=None, diag=None, with_t3p2=False, with_t3p2_imds=False):
        '''2ph operators are of the form s_{ij}^{ b}, i.e. 'jb' indices are coupled.'''
        # Ref: Nooijen and Snijders, J. Chem. Phys. 102, 1681 (1995) Eqs.(8)-(9)
        if not hasattr(self, 'imds'):
            if imds is None:
                self.imds = _IMDS(self, with_t3p2=with_t3p2, with_t3p2_imds=with_t3p2_imds)
            else:
                self.imds = imds
        self.imds.ip_partition = self.ip_partition
        if not self.imds.made_ip_imds:
            self.imds.make_ip(self.ip_partition, with_t3p2=with_t3p2, with_t3p2_imds=with_t3p2_imds)
        imds = self.imds

        vector = self.mask_frozen_ip(vector, kshift, const=0.0)
        r1, r2 = self.ip_vector_to_amplitudes(vector)

        t1, t2 = self.t1, self.t2
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        kconserv = self.khelper.kconserv

        # 1h-1h block
        Hr1 = -einsum('ki,k->i', imds.Loo[kshift], r1)
        # 1h-2h1p block
        for kl in range(nkpts):
            Hr1 += 2. * einsum('ld,ild->i', imds.Fov[kl], r2[kshift, kl])
            Hr1 += -einsum('ld,lid->i', imds.Fov[kl], r2[kl, kshift])
            for kk in range(nkpts):
                kd = kconserv[kk, kshift, kl]
                Hr1 += -2. * einsum('klid,kld->i', imds.Wooov[kk, kl, kshift], r2[kk, kl])
                Hr1 += einsum('lkid,kld->i', imds.Wooov[kl, kk, kshift], r2[kk, kl])

        Hr2 = np.zeros(r2.shape, dtype=np.common_type(imds.Wovoo[0, 0, 0], r1))
        # 2h1p-1h block
        for ki in range(nkpts):
            for kj in range(nkpts):
                kb = kconserv[ki, kshift, kj]
                Hr2[ki, kj] -= einsum('kbij,k->ijb', imds.Wovoo[kshift, kb, ki], r1)
                if with_t3p2_imds:
                    Hr2[ki, kj] += -np.einsum('kbij,k->ijb', imds.Wovoo_t3p2[kshift, kb, ki], r1)
        # 2h1p-2h1p block
        if self.ip_partition == 'mp':
            nkpts, nocc, nvir = self.t1.shape
            fock = self.eris.fock
            foo = fock[:, :nocc, :nocc]
            fvv = fock[:, nocc:, nocc:]
            for ki in range(nkpts):
                for kj in range(nkpts):
                    kb = kconserv[ki, kshift, kj]
                    Hr2[ki, kj] += einsum('bd,ijd->ijb', fvv[kb], r2[ki, kj])
                    Hr2[ki, kj] -= einsum('li,ljb->ijb', foo[ki], r2[ki, kj])
                    Hr2[ki, kj] -= einsum('lj,ilb->ijb', foo[kj], r2[ki, kj])
        elif self.ip_partition == 'full':
            Hr2 += self._ipccsd_diag_matrix2 * r2
        else:
            for ki in range(nkpts):
                for kj in range(nkpts):
                    kb = kconserv[ki, kshift, kj]
                    Hr2[ki, kj] += einsum('bd,ijd->ijb', imds.Lvv[kb], r2[ki, kj])
                    Hr2[ki, kj] -= einsum('li,ljb->ijb', imds.Loo[ki], r2[ki, kj])
                    Hr2[ki, kj] -= einsum('lj,ilb->ijb', imds.Loo[kj], r2[ki, kj])
                    for kl in range(nkpts):
                        kk = kconserv[ki, kl, kj]
                        Hr2[ki, kj] += einsum('klij,klb->ijb', imds.Woooo[kk, kl, ki], r2[kk, kl])
                        kd = kconserv[kl, kj, kb]
                        Hr2[ki, kj] += 2. * einsum('lbdj,ild->ijb', imds.Wovvo[kl, kb, kd], r2[ki, kl])
                        Hr2[ki, kj] += -einsum('lbdj,lid->ijb', imds.Wovvo[kl, kb, kd], r2[kl, ki])
                        Hr2[ki, kj] += -einsum('lbjd,ild->ijb', imds.Wovov[kl, kb, kj], r2[ki, kl])  # typo in Ref
                        kd = kconserv[kl, ki, kb]
                        Hr2[ki, kj] += -einsum('lbid,ljd->ijb', imds.Wovov[kl, kb, ki], r2[kl, kj])
            tmp = (2. * einsum('xyklcd,xykld->c', imds.Woovv[:, :, kshift], r2[:, :])
                      - einsum('yxlkcd,xykld->c', imds.Woovv[:, :, kshift], r2[:, :]))
            Hr2[:, :] += -einsum('c,xyijcb->xyijb', tmp, t2[:, :, kshift])

        return self.mask_frozen_ip(self.ip_amplitudes_to_vector(Hr1, Hr2), kshift, const=0.0)

    def lipccsd_matvec(self, vector, kshift, imds=None, diag=None, with_t3p2=False, with_t3p2_imds=False):
        # Ref: Nooijen and Snijders, J. Chem. Phys. 102, 1681 (1995) Eqs.(8)-(9)
        if not hasattr(self, 'imds'):
            if imds is None:
                self.imds = _IMDS(self, with_t3p2=with_t3p2, with_t3p2_imds=with_t3p2_imds)
            else:
                self.imds = imds
        self.imds.ip_partition = self.ip_partition
        if not self.imds.made_ip_imds:
            self.imds.make_ip(self.ip_partition, with_t3p2=with_t3p2, with_t3p2_imds=with_t3p2_imds)
        imds = self.imds

        vector = self.mask_frozen_ip(vector, kshift, const=0.0)
        r1, r2 = self.ip_vector_to_amplitudes(vector)

        t1, t2 = self.t1, self.t2
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        kconserv = self.khelper.kconserv

        Hr1 = -einsum('ki,i->k',imds.Loo[kshift],r1)
        for ki, kb in product(range(nkpts), repeat=2):
            kj = kconserv[kshift,ki,kb]
            Hr1 -= einsum('kbij,ijb->k',imds.Wovoo[kshift,kb,ki],r2[ki,kj])
            if with_t3p2_imds:
                Hr1 += -np.einsum('kbij,ijb->k', imds.Wovoo_t3p2[kshift,kb,ki], r2[ki,kj])

        Hr2 = np.zeros(r2.shape, dtype=np.complex128)
        for kl, kk in product(range(nkpts), repeat=2):
            kd = kconserv[kk,kshift,kl]
            SWooov = (2. * imds.Wooov[kk,kl,kshift] -
                           imds.Wooov[kl,kk,kshift].transpose(1, 0, 2, 3))
            Hr2[kk,kl] -= einsum('klid,i->kld',SWooov,r1)

            Hr2[kk,kshift] -= (kk==kd)*einsum('kd,l->kld',imds.Fov[kk],r1)
            Hr2[kshift,kl] += (kl==kd)*2.*einsum('ld,k->kld',imds.Fov[kl],r1)

        if self.ip_partition == 'mp':
            fock = self.eris.fock
            foo = fock[:,:nocc,:nocc]
            fvv = fock[:,nocc:,nocc:]
            for kl, kk in product(range(nkpts), repeat=2):
                kd = kconserv[kk,kshift,kl]
                Hr2[kk,kl] -= einsum('ki,ild->kld',foo[kk],r2[kk,kl])
                Hr2[kk,kl] -= einsum('lj,kjd->kld',foo[kl],r2[kk,kl])
                Hr2[kk,kl] += einsum('bd,klb->kld',fvv[kd],r2[kk,kl])
        elif self.ip_partition == 'full':
            Hr2 += self._ipccsd_diag_matrix2*r2
        else:
            r2t2_tmp = np.zeros((nvir),dtype=t1.dtype)
            for ki, kj in product(range(nkpts), repeat=2):
                kc = kshift
                r2t2_tmp += einsum('ijcb,ijb->c',t2[ki, kj, kc],r2[ki, kj])

            for kl, kk in product(range(nkpts), repeat=2):
                kd = kconserv[kk,kshift,kl]
                Hr2[kk,kl] -= einsum('ki,ild->kld',imds.Loo[kk],r2[kk,kl])
                Hr2[kk,kl] -= einsum('lj,kjd->kld',imds.Loo[kl],r2[kk,kl])
                Hr2[kk,kl] += einsum('bd,klb->kld',imds.Lvv[kd],r2[kk,kl])

                SWoovv = (2. * imds.Woovv[kl, kk, kd] -
                               imds.Woovv[kk, kl, kd].transpose(1, 0, 2, 3))
                Hr2[kk, kl] -= einsum('lkdc,c->kld',SWoovv,r2t2_tmp)

                for kj in range(nkpts):
                    kb = kconserv[kd, kl, kj]

                    SWovvo = (2. * imds.Wovvo[kl,kb,kd] -
                                   imds.Wovov[kl,kb,kj].transpose(0, 1, 3, 2))
                    Hr2[kk,kl] += einsum('lbdj,kjb->kld',SWovvo,r2[kk,kj])

                    kb = kconserv[kd, kk, kj]
                    Hr2[kk,kl] -= einsum('kbdj,ljb->kld',imds.Wovvo[kk,kb,kd],r2[kl,kj])
                    Hr2[kk,kl] -= einsum('kbjd,jlb->kld',imds.Wovov[kk,kb,kj],r2[kj,kl])

                    ki = kconserv[kk,kj,kl]
                    Hr2[kk,kl] += einsum('klji,jid->kld',imds.Woooo[kk,kl,kj],r2[kj,ki])

        return self.mask_frozen_ip(self.ip_amplitudes_to_vector(Hr1, Hr2), kshift, const=0.0)

    def _ipccsd_star(self, ipccsd_evals, ipccsd_evecs, lipccsd_evecs, kptlist):
        """Calculates perturbative correction IP-CCSD*

        Args:
            eom (:obj:`EOMIP`):
                Object containing coupled-cluster results.
            ipccsd_evals (array-like):
                Right EOM-IP-CCSD eigenvalues; should be same as left eigenvalues.
            ipccsd_evecs (array-like):
                List of right EOM-IP-CCSD eigenvectors.
            lipccsd_evecs (array-like):
                List of left EOM-IP-CCSD eigenvectors.
            kptlist (list):
                List of k-point indices for which eigenvalue perturbations are requested.

        Returns:
            e_star (list of :obj:`ndarray`):
                The IP-CCSD* energies at each kpoint.

        Notes:
            The user should check to make sure the right and left eigenvalues agree
            before running the perturbative correction.

            The 2hp right amplitudes are assumed to be of the form s^{a }_{ij}, i.e.
            the (ia) indices are coupled while the left are assumed to be of the form
            s^{ b}_{ij}, i.e. the (jb) indices are coupled.

        Reference:
            Saeh, Stanton "...energy surfaces of radicals" JCP 111, 8275 (1999)

        """
        from itertools import product
        from pyscf.pbc.lib import kpts_helper
        time0 = time.clock(), time.time()

        assert(self.ip_partition == None)
        eris = self.eris
        #assert(isinstance(eris, gccsd._PhysicistsERIs))
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        size = nocc + nkpts*nkpts*nocc*nocc*nvir
        kconserv = self.khelper.kconserv

        ipccsd_evecs, lipccsd_evecs = atleast_3d(ipccsd_evecs), atleast_3d(lipccsd_evecs)
        ipccsd_evals = np.atleast_2d(ipccsd_evals)
        # TODO: generate checks for input array sizes
        assert(ipccsd_evecs.shape[-1] == size)
        assert(ipccsd_evecs.shape == lipccsd_evecs.shape)

        t1, t2 = self.t1, self.t2
        foo = [eris.fock[ikpt,:nocc,:nocc].diagonal() for ikpt in range(nkpts)]
        fvv = [eris.fock[ikpt,nocc:,nocc:].diagonal() for ikpt in range(nkpts)]

        ipccsd_evecs  = np.array(ipccsd_evecs)
        lipccsd_evecs = np.array(lipccsd_evecs)
        e_star = []
        for ikshift, kshift in enumerate(kptlist):
            e_star_kpt = []
            for ip_eval, ip_evec, ip_levec in zip(ipccsd_evals[ikshift],
                ipccsd_evecs[ikshift], lipccsd_evecs[ikshift]):

                # Enforcing <L|R> = 1
                l1, l2 = self.ip_vector_to_amplitudes(ip_levec)
                r1, r2 = self.ip_vector_to_amplitudes(ip_evec)
                ldotr = np.dot(l1, r1) + np.dot(l2.ravel(), r2.ravel())

                logger.info(self, 'Left-right amplitude overlap : %14.8e', ldotr.real)
                if abs(ldotr) < 1e-7:
                    logger.warn(self, 'Small %s left-right amplitude overlap. Results '
                                     'may be inaccurate.', ldotr.real)

                l1 /= ldotr
                l2 /= ldotr

                deltaE = 0.0 + 0.0*1j
                for ki, kj, ka, kb in product(range(nkpts), repeat=4):
                    fac = 1.
                    if ka > kb:
                        continue
                    fac = (2.0 - float(ka==kb))

                    if ka == kb:
                        if ki > kj:
                            continue
                        else:
                            fac *= (2.0 - float(ki==kj))

                    kk = kpts_helper.get_kconserv3(self._scf.cell, self.kpts,
                                                   [ka, kb ,kshift, ki, kj])
                    lijkab_tmp = np.zeros((nocc,nocc,nocc,nvir,nvir), dtype=t2.dtype)
                    rijkab_tmp = np.zeros((nocc,nocc,nocc,nvir,nvir), dtype=t2.dtype)

                    # (i,j,k) -> (i,j,k)
                    if kk == kshift and kb == kconserv[ki,ka,kj]:
                        tmp = 0.5*einsum('ijab,k->ijkab',eris.oovv[ki,kj,ka],l1)
                        lijkab_tmp += 8.* tmp

                    # (j,i,k) -> (i,j,k)
                    if kk == kshift and kb == kconserv[ki,ka,kj]:
                        tmp = 0.5*einsum('jiab,k->ijkab',eris.oovv[kj,ki,ka],l1)
                        lijkab_tmp -= 4.* tmp

                    # (k,j,i) -> (i,j,k)
                    if ki == kshift and kb == kconserv[kj,ka,kk]:
                        tmp = 0.5*einsum('kjab,i->ijkab',eris.oovv[kk,kj,ka],l1)
                        lijkab_tmp -= 4.* tmp

                    # (i,k,j) -> (i,j,k)
                    if kj == kshift and kb == kconserv[ki,ka,kk]:
                        tmp = 0.5*einsum('ikab,j->ijkab',eris.oovv[ki,kk,ka],l1)
                        lijkab_tmp -= 4.* tmp

                    # (j,k,i) -> (i,j,k)
                    if kj == kshift and kb == kconserv[ki,ka,kk]:
                        tmp = 0.5*einsum('kiab,j->ijkab',eris.oovv[kk,ki,ka],l1)
                        lijkab_tmp += 2.* tmp

                    # (k,i,j) -> (i,j,k)
                    if ki == kshift and kb == kconserv[kj,ka,kk]:
                        tmp = 0.5*einsum('jkab,i->ijkab',eris.oovv[kj,kk,ka],l1)
                        lijkab_tmp += 2.* tmp

                    # Beginning of vovv terms

                    # (i,j,k) -> (i,j,k)
                    ke = kconserv[ka,ki,kb]
                    tmp = einsum('eiba,kje->ijkab',eris.vovv[ke,ki,kb], l2[kk,kj])
                    lijkab_tmp += 2.*tmp

                    ke = kconserv[ka,kj,kb]
                    tmp = einsum('ejab,kie->ijkab',eris.vovv[ke,kj,ka], l2[kk,ki])
                    lijkab_tmp += 2.*tmp

                    # (j,i,k) -> (i,j,k)
                    ke = kconserv[ka,kj,kb]
                    tmp = einsum('ejba,kie->ijkab',eris.vovv[ke,kj,kb], l2[kk,ki])
                    lijkab_tmp -= 1.*tmp

                    ke = kconserv[ka,ki,kb]
                    tmp = einsum('eiab,kje->ijkab',eris.vovv[ke,ki,ka], l2[kk, kj])
                    lijkab_tmp -= 1.*tmp

                    # (k,j,i) -> (i,j,k)
                    ke = kconserv[ki,kshift,kj]
                    tmp = einsum('ekba,ije->ijkab',eris.vovv[ke,kk,kb], l2[ki, kj])
                    lijkab_tmp -= 1.*tmp

                    ke = kconserv[ki,kshift,kj]
                    tmp = einsum('ekab,jie->ijkab',eris.vovv[ke,kk,ka], l2[kj, ki])
                    lijkab_tmp -= 1.*tmp

                    # ooov part 1

                    # (i,j,k) -> (i,j,k)
                    km = kconserv[kshift,ki,ka]
                    tmp = -einsum('kjmb,mia->ijkab',eris.ooov[kk,kj,km], l2[km, ki])
                    lijkab_tmp += 2.*tmp

                    km = kconserv[kshift,kj,kb]
                    tmp = -einsum('kima,mjb->ijkab',eris.ooov[kk,ki,km], l2[km, kj])
                    lijkab_tmp += 2.*tmp

                    # (j,i,k) -> (i,j,k)
                    km = kconserv[kshift,kj,ka]
                    tmp = -einsum('kimb,mja->ijkab',eris.ooov[kk,ki,km], l2[km, kj])
                    lijkab_tmp -= 1.*tmp

                    km = kconserv[kshift,ki,kb]
                    tmp = -einsum('kjma,mib->ijkab',eris.ooov[kk,kj,km], l2[km, ki])
                    lijkab_tmp -= 1.*tmp

                    # (k,j,i) -> (i,j,k)
                    km = kconserv[kshift,kj,kb]
                    tmp = -einsum('ikma,mjb->ijkab',eris.ooov[ki,kk,km], l2[km, kj])
                    lijkab_tmp -= 1.*tmp

                    km = kconserv[kshift,ki,ka]
                    tmp = -einsum('jkmb,mia->ijkab',eris.ooov[kj,kk,km], l2[km, ki])
                    lijkab_tmp -= 1.*tmp

                    # ooov part 2

                    # (i,j,k) -> (i,j,k)
                    km = kconserv[ki,kb,kj]
                    tmp = -einsum('ijmb,kma->ijkab',eris.ooov[ki,kj,km], l2[kk, km])
                    lijkab_tmp += 2.*tmp

                    km = kconserv[kj,ka,ki]
                    tmp = -einsum('jima,kmb->ijkab',eris.ooov[kj,ki,km], l2[kk, km])
                    lijkab_tmp += 2.*tmp

                    # (j,i,k) -> (i,j,k)
                    km = kconserv[ki,kb,kj]
                    tmp = -einsum('jimb,kma->ijkab',eris.ooov[kj,ki,km], l2[kk, km])
                    lijkab_tmp -= 1.*tmp

                    km = kconserv[kj,ka,ki]
                    tmp = -einsum('ijma,kmb->ijkab',eris.ooov[ki,kj,km], l2[kk, km])
                    lijkab_tmp -= 1.*tmp

                    # (k,j,i) -> (i,j,k)
                    km = kconserv[kshift,ki,kb]
                    tmp = -einsum('jkma,imb->ijkab',eris.ooov[kj,kk,km], l2[ki, km])
                    lijkab_tmp -= 1.*tmp

                    km = kconserv[kshift,kj,ka]
                    tmp = -einsum('ikmb,jma->ijkab',eris.ooov[ki,kk,km], l2[kj, km])
                    lijkab_tmp -= 1.*tmp

                    # Starting the right amplitude equations #

                    # (ia,jb) -> (ia,jb)
                    km = kconserv[ka,ki,kb]
                    tmp = einsum('mnjk,n->mjk',eris.oooo[km,kshift,kj],r1)
                    tmp2 = einsum('mjk,imab->ijkab',tmp,t2[ki, km, ka])
                    rijkab_tmp += tmp2

                    km = kshift
                    ke = kconserv[ki,ka,kj]
                    tmp = einsum('mbke,m->bke',eris.ovov[km,kb,kk],r1)
                    tmp2 = -einsum('bke,ijae->ijkab',tmp,t2[ki, kj, ka])
                    rijkab_tmp += tmp2

                    km = kshift
                    ke = kconserv[km,kj,kb]
                    tmp = einsum('bmje,m->bej',eris.voov[kb,km,kj],r1)
                    tmp2 = -einsum('bej,ikae->ijkab',tmp,t2[ki, kk, ka])
                    rijkab_tmp += tmp2

                    ke = kconserv[ka,ki,kb]
                    tmp2 = einsum('eiba,kje->ijkab',eris.vovv[ke,ki,kb].conj(),r2[kk, kj])
                    rijkab_tmp += tmp2

                    km = kconserv[kshift,ki,ka]
                    tmp2 = -einsum('kjmb,mia->ijkab',eris.ooov[kk,kj,km].conj(),r2[km, ki])
                    rijkab_tmp += tmp2

                    km = kconserv[ki,kb,kj]
                    tmp2 = -einsum('ijmb,kma->ijkab',eris.ooov[ki,kj,km].conj(),r2[kk,km])
                    rijkab_tmp += tmp2

                    # (ia,jb) -> (jb,ia)
                    km = kconserv[kb,kj,ka]
                    tmp = einsum('mnik,n->mik',eris.oooo[km,kshift,ki],r1)
                    tmp2 = einsum('mik,jmba->ijkab',tmp,t2[kj, km, kb])
                    rijkab_tmp += tmp2

                    km = kshift
                    ke = kconserv[ki,kb,kj]
                    tmp = einsum('make,m->ake',eris.ovov[km,ka,kk],r1)
                    tmp2 = -einsum('ake,jibe->ijkab',tmp,t2[kj, ki, kb])
                    rijkab_tmp += tmp2

                    km = kshift
                    ke = kconserv[km,ki,ka]
                    tmp = einsum('amie,m->aei',eris.voov[ka,km,ki],r1)
                    tmp2 = -einsum('aei,jkbe->ijkab',tmp,t2[kj, kk, kb])
                    rijkab_tmp += tmp2

                    ke = kconserv[kb,kj,ka]
                    tmp2 = einsum('ejab,kie->ijkab',eris.vovv[ke,kj,ka].conj(),r2[kk,   ki])
                    rijkab_tmp += tmp2

                    km = kconserv[kshift,kj,kb]
                    tmp2 = -einsum('kima,mjb->ijkab',eris.ooov[kk,ki,km].conj(),r2[km,  kj])
                    rijkab_tmp += tmp2

                    km = kconserv[kj,ka,ki]
                    tmp2 = -einsum('jima,kmb->ijkab',eris.ooov[kj,ki,km].conj(),r2[kk,km])
                    rijkab_tmp += tmp2

                    eijk = (foo[ki][:, None, None] + foo[kj][None, :, None] +
                            foo[kk][None, None, :])
                    eab = fvv[ka][:, None] + fvv[kb][None, :]
                    eijkab = eijk[:, :, :, None, None] - eab[None, None, None, :, :]
                    denom = eijkab + ip_eval
                    denom = 1. / denom

                    deltaE += fac*0.5*einsum('ijkab,ijkab,ijkab',lijkab_tmp,rijkab_tmp,denom)

                logger.info(self, "Exc. energy, delta energy = %16.12f, %16.12f",
                            ip_eval + deltaE.real, deltaE.real)
                e_star_kpt.append(ip_eval + deltaE.real)

            e_star.append(e_star_kpt)
        return e_star

    def ipccsd_diag(self, kshift, with_t3p2=False, with_t3p2_imds=False):
        if not hasattr(self, 'imds'):
            self.imds = _IMDS(self, with_t3p2=with_t3p2, with_t3p2_imds=with_t3p2_imds)
        self.imds.ip_partition = self.ip_partition
        if not self.imds.made_ip_imds:
            self.imds.make_ip(self.ip_partition, with_t3p2=with_t3p2, with_t3p2_imds=with_t3p2_imds)
        imds = self.imds

        t1, t2 = self.t1, self.t2
        nkpts, nocc, nvir = t1.shape
        kconserv = self.khelper.kconserv

        Hr1 = -np.diag(imds.Loo[kshift])

        Hr2 = np.zeros((nkpts, nkpts, nocc, nocc, nvir), dtype=t1.dtype)
        if self.ip_partition == 'mp':
            foo = self.eris.fock[:, :nocc, :nocc]
            fvv = self.eris.fock[:, nocc:, nocc:]
            for ki in range(nkpts):
                for kj in range(nkpts):
                    kb = kconserv[ki, kshift, kj]
                    Hr2[ki, kj] = fvv[kb].diagonal()
                    Hr2[ki, kj] -= foo[ki].diagonal()[:, None, None]
                    Hr2[ki, kj] -= foo[kj].diagonal()[:, None]
        else:
            idx = np.arange(nocc)
            for ki in range(nkpts):
                for kj in range(nkpts):
                    kb = kconserv[ki, kshift, kj]
                    Hr2[ki, kj] = imds.Lvv[kb].diagonal()
                    Hr2[ki, kj] -= imds.Loo[ki].diagonal()[:, None, None]
                    Hr2[ki, kj] -= imds.Loo[kj].diagonal()[:, None]

                    if ki == kconserv[ki, kj, kj]:
                        Hr2[ki, kj] += np.einsum('ijij->ij', imds.Woooo[ki, kj, ki])[:, :, None]

                    Hr2[ki, kj] -= np.einsum('jbjb->jb', imds.Wovov[kj, kb, kj])

                    Wovvo = np.einsum('jbbj->jb', imds.Wovvo[kj, kb, kb])
                    Hr2[ki, kj] += 2. * Wovvo
                    if ki == kj:  # and i == j
                        Hr2[ki, ki, idx, idx] -= Wovvo

                    Hr2[ki, kj] -= np.einsum('ibib->ib', imds.Wovov[ki, kb, ki])[:, None, :]

                    kd = kconserv[kj, kshift, ki]
                    Hr2[ki, kj] -= 2. * np.einsum('ijcb,jibc->ijb', t2[ki, kj, kshift], imds.Woovv[kj, ki, kd])
                    Hr2[ki, kj] += np.einsum('ijcb,ijbc->ijb', t2[ki, kj, kshift], imds.Woovv[ki, kj, kd])

        return self.ip_amplitudes_to_vector(Hr1, Hr2)

    def mask_frozen_ip(self, vector, kshift, const=LARGE_DENOM):
        '''Replaces all frozen orbital indices of `vector` with the value `const`.'''
        r1, r2 = self.ip_vector_to_amplitudes(vector)
        nkpts, nocc, nvir = self.t1.shape
        kconserv = self.khelper.kconserv

        # Get location of padded elements in occupied and virtual space
        nonzero_opadding, nonzero_vpadding = padding_k_idx(self, kind="split")

        new_r1 = const * np.ones_like(r1)
        new_r2 = const * np.ones_like(r2)

        new_r1[nonzero_opadding[kshift]] = r1[nonzero_opadding[kshift]]
        for ki in range(nkpts):
            for kj in range(nkpts):
                kb = kconserv[ki, kshift, kj]
                idx = np.ix_([ki], [kj], nonzero_opadding[ki], nonzero_opadding[kj], nonzero_vpadding[kb])
                new_r2[idx] = r2[idx]

        return self.ip_amplitudes_to_vector(new_r1, new_r2)

    def vector_size_ea(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts

        size = nvir + nkpts ** 2 * nvir ** 2 * nocc
        return size

    def eaccsd(self, nroots=1, left=False, koopmans=False, guess=None, partition=None,
               kptlist=None, **kwargs):
        '''Calculate (N+1)-electron charged excitations via EA-EOM-CCSD.

        Kwargs:
            See ipccd()
        '''
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        if kptlist is None:
            kptlist = range(nkpts)
        size = self.vector_size_ea()
        for k, kshift in enumerate(kptlist):
            nfrozen = np.sum(self.mask_frozen_ea(np.zeros(size, dtype=int), kshift, const=1))
            nroots = min(nroots, size - nfrozen)
        if partition:
            partition = partition.lower()
            assert partition in ['mp', 'full']
        self.ea_partition = partition
        evals = np.zeros((len(kptlist), nroots), np.float)
        evecs = np.zeros((len(kptlist), nroots, size), np.complex)

        # Initialize intermediates
        if not hasattr(self, 'imds'):
            self.imds = _IMDS(self, with_t3p2=kwargs.get('with_t3p2', False),
                                    with_t3p2_imds=kwargs.get('with_t3p2_imds', False))

        for k, kshift in enumerate(kptlist):
            if left:
                matvec = lambda _arg: self.leaccsd_matvec(_arg, kshift, **kwargs)
            else:
                matvec = lambda _arg: self.eaccsd_matvec(_arg, kshift, **kwargs)

            adiag = self.eaccsd_diag(kshift)
            adiag = self.mask_frozen_ea(adiag, kshift, const=LARGE_DENOM)
            if partition == 'full':
                self._eaccsd_diag_matrix2 = self.ea_vector_to_amplitudes(adiag)[1]

            user_guess = False
            if guess:
                user_guess = True
                assert len(guess[k]) == nroots
                for g in guess[k]:
                    assert g.size == size
            else:
                guess = []
                if koopmans:
                    # Get location of padded elements in occupied and virtual space
                    nonzero_vpadding = padding_k_idx(self, kind="split")[1][kshift]

                    for n in nonzero_vpadding[:nroots]:
                        g = np.zeros(size)
                        g[n] = 1.0
                        g = self.mask_frozen_ea(g, kshift, const=0.0)
                        guess.append(g)
                else:
                    idx = adiag.argsort()[:nroots]
                    for i in idx:
                        g = np.zeros(size)
                        g[i] = 1.0
                        g = self.mask_frozen_ea(g, kshift, const=0.0)
                        guess.append(g)

            def precond(r, e0, x0):
                return r / (e0 - adiag + 1e-12)

            eig = linalg_helper.eig
            if user_guess or koopmans:
                def pickeig(w, v, nr, envs):
                    x0 = linalg_helper._gen_x0(envs['v'], envs['xs'])
                    idx = np.argmax(np.abs(np.dot(np.array(guess).conj(), np.array(x0).T)), axis=1)
                    return lib.linalg_helper._eigs_cmplx2real(w, v, idx)

                evals_k, evecs_k = eig(matvec, guess, precond, pick=pickeig,
                                       tol=self.conv_tol, max_cycle=self.max_cycle,
                                       max_space=self.max_space, nroots=nroots, verbose=self.verbose)
            else:
                evals_k, evecs_k = eig(matvec, guess, precond,
                                       tol=self.conv_tol, max_cycle=self.max_cycle,
                                       max_space=self.max_space, nroots=nroots, verbose=self.verbose)
            if not user_guess:
                guess = None

            evals_k = evals_k.real
            evals[k] = evals_k
            evecs[k] = evecs_k

            if nroots == 1:
                evals_k, evecs_k = [evals_k], [evecs_k]

            for n, en, vn in zip(range(nroots), evals_k, evecs_k):
                r1, r2 = self.ea_vector_to_amplitudes(vn)
                qp_weight = np.linalg.norm(r1) ** 2
                logger.info(self, 'EOM root %d E = %.16g  qpwt = %0.6g',
                            n, en, qp_weight)
        log.timer('EOM-CCSD', *cput0)
        self.eea = evals
        return self.eea, evecs

    def eaccsd_matvec(self, vector, kshift, imds=None, diag=None, with_t3p2=False, with_t3p2_imds=False):
        # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
        if not hasattr(self, 'imds'):
            if imds is None:
                self.imds = _IMDS(self, with_t3p2=with_t3p2, with_t3p2_imds=with_t3p2_imds)
            else:
                self.imds = imds
        if not self.imds.made_ea_imds:
            self.imds.make_ea(self.ea_partition, with_t3p2=with_t3p2, with_t3p2_imds=with_t3p2_imds)
        imds = self.imds

        vector = self.mask_frozen_ea(vector, kshift, const=0.0)
        r1, r2 = self.ea_vector_to_amplitudes(vector)

        t1, t2 = self.t1, self.t2
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        kconserv = self.khelper.kconserv

        # Eq. (30)
        # 1p-1p block
        Hr1 = einsum('ac,c->a', imds.Lvv[kshift], r1)
        # 1p-2p1h block
        for kl in range(nkpts):
            Hr1 += 2. * einsum('ld,lad->a', imds.Fov[kl], r2[kl, kshift])
            Hr1 += -einsum('ld,lda->a', imds.Fov[kl], r2[kl, kl])
            for kc in range(nkpts):
                kd = kconserv[kshift, kc, kl]
                Hr1 += 2. * einsum('alcd,lcd->a', imds.Wvovv[kshift, kl, kc], r2[kl, kc])
                Hr1 += -einsum('aldc,lcd->a', imds.Wvovv[kshift, kl, kd], r2[kl, kc])

        # Eq. (31)
        # 2p1h-1p block
        Hr2 = np.zeros(r2.shape, dtype=np.common_type(imds.Wvvvo[0, 0, 0], r1))
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[kshift,ka,kj]
                Hr2[kj,ka] += einsum('abcj,c->jab',imds.Wvvvo[ka,kb,kshift],r1)
                if with_t3p2_imds:
                    Hr2[kj,ka] += einsum('abcj,c->jab',imds.Wvvvo_t3p2[ka,kb,kshift],r1)

        # 2p1h-2p1h block
        if self.ea_partition == 'mp':
            fock = self.eris.fock
            foo = fock[:, :nocc, :nocc]
            fvv = fock[:, nocc:, nocc:]
            for kj in range(nkpts):
                for ka in range(nkpts):
                    kb = kconserv[kshift, ka, kj]
                    Hr2[kj, ka] -= einsum('lj,lab->jab', foo[kj], r2[kj, ka])
                    Hr2[kj, ka] += einsum('ac,jcb->jab', fvv[ka], r2[kj, ka])
                    Hr2[kj, ka] += einsum('bd,jad->jab', fvv[kb], r2[kj, ka])
        elif self.ea_partition == 'full':
            Hr2 += self._eaccsd_diag_matrix2 * r2
        else:
            for kj in range(nkpts):
                for ka in range(nkpts):
                    kb = kconserv[kshift, ka, kj]
                    Hr2[kj, ka] -= einsum('lj,lab->jab', imds.Loo[kj], r2[kj, ka])
                    Hr2[kj, ka] += einsum('ac,jcb->jab', imds.Lvv[ka], r2[kj, ka])
                    Hr2[kj, ka] += einsum('bd,jad->jab', imds.Lvv[kb], r2[kj, ka])
                    for kd in range(nkpts):
                        kc = kconserv[ka, kd, kb]
                        Hr2[kj, ka] += einsum('abcd,jcd->jab', imds.Wvvvv[ka, kb, kc], r2[kj, kc])
                        kl = kconserv[kd, kb, kj]
                        Hr2[kj, ka] += 2. * einsum('lbdj,lad->jab', imds.Wovvo[kl, kb, kd], r2[kl, ka])
                        # imds.Wvovo[kb,kl,kd,kj] <= imds.Wovov[kl,kb,kj,kd].transpose(1,0,3,2)
                        Hr2[kj, ka] += -einsum('bldj,lad->jab', imds.Wovov[kl, kb, kj].transpose(1, 0, 3, 2),
                                               r2[kl, ka])
                        # imds.Wvoov[kb,kl,kj,kd] <= imds.Wovvo[kl,kb,kd,kj].transpose(1,0,3,2)
                        Hr2[kj, ka] += -einsum('bljd,lda->jab', imds.Wovvo[kl, kb, kd].transpose(1, 0, 3, 2),
                                               r2[kl, kd])
                        kl = kconserv[kd, ka, kj]
                        # imds.Wvovo[ka,kl,kd,kj] <= imds.Wovov[kl,ka,kj,kd].transpose(1,0,3,2)
                        Hr2[kj, ka] += -einsum('aldj,ldb->jab', imds.Wovov[kl, ka, kj].transpose(1, 0, 3, 2),
                                               r2[kl, kd])
            tmp = (2. * einsum('xyklcd,xylcd->k', imds.Woovv[kshift, :, :], r2[:, :])
                      - einsum('xylkcd,xylcd->k', imds.Woovv[:, kshift, :], r2[:, :]))
            Hr2[:, :] += -einsum('k,xykjab->xyjab', tmp, t2[kshift, :, :])

        return self.mask_frozen_ea(self.ea_amplitudes_to_vector(Hr1, Hr2), kshift, const=0.0)

    def leaccsd_matvec(self, vector, kshift, imds=None, diag=None, with_t3p2=False, with_t3p2_imds=False):
        # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
        if not hasattr(self, 'imds'):
            if imds is None:
                self.imds = _IMDS(self, with_t3p2=with_t3p2, with_t3p2_imds=with_t3p2_imds)
            else:
                self.imds = imds
        if not self.imds.made_ea_imds:
            self.imds.make_ea(self.ea_partition, with_t3p2=with_t3p2, with_t3p2_imds=with_t3p2_imds)
        imds = self.imds

        vector = self.mask_frozen_ea(vector, kshift, const=0.0)
        r1, r2 = self.ea_vector_to_amplitudes(vector)

        t1, t2 = self.t1, self.t2
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        kconserv = self.khelper.kconserv

        # 1p-1p block
        Hr1 = np.einsum('ac,a->c', imds.Lvv[kshift], r1)
        # 1p-2p1h block
        for kj, ka in product(range(nkpts), repeat=2):
            kb = kconserv[kj, ka, kshift]
            Hr1 += np.einsum('abcj,jab->c', imds.Wvvvo[ka, kb, kshift], r2[kj, ka])
            if with_t3p2_imds:
                Hr1 += np.einsum('abcj,jab->c', imds.Wvvvo_t3p2[ka, kb, kshift], r2[kj, ka])

        # 2p1h-1p block
        Hr2 = np.zeros((nkpts, nkpts, nocc, nvir, nvir), dtype=np.complex128)
        for kl, kc in product(range(nkpts), repeat=2):
            kd = kconserv[kl, kc, kshift]
            Hr2[kl, kc] += 2. * (kl==kd) * np.einsum('c,ld->lcd', r1, imds.Fov[kd])
            Hr2[kl, kc] += - (kl==kc) * np.einsum('d,lc->lcd', r1, imds.Fov[kl])

            SWvovv = (2. * imds.Wvovv[kshift, kl, kc] -
                           imds.Wvovv[kshift, kl, kd].transpose(0, 1, 3, 2))
            Hr2[kl, kc] += np.einsum('a,alcd->lcd', r1, SWvovv)

        # 2p1h-2p1h block
        if self.ea_partition == 'mp':
            fock = self.eris.fock
            foo = fock[:,:nocc,:nocc]
            fvv = fock[:,nocc:,nocc:]
            for kl, kc in product(range(nkpts), repeat=2):
                kd = kconserv[kl, kc, kshift]
                Hr2[kl, kc] += lib.einsum('lad,ac->lcd', r2[kl, kc], fvv[kc])
                Hr2[kl, kc] += lib.einsum('lcb,bd->lcd', r2[kl, kc], fvv[kd])
                Hr2[kl, kc] += -lib.einsum('jcd,lj->lcd', r2[kl, kc], foo[kl])
        elif self.ea_partition == 'full':
            Hr2 += self._eaccsd_diag_matrix2*r2
        else:
            r2t2_tmp = np.zeros((nocc),dtype=t1.dtype)
            for ki, kc in product(range(nkpts), repeat=2):
                kb = kconserv[ki, kc, kshift]
                r2t2_tmp += np.einsum('ijcb,ibc->j', imds.t2[ki, kshift, kc], r2[ki, kb])

            for kl, kc in product(range(nkpts), repeat=2):
                kd = kconserv[kl, kc, kshift]
                Hr2[kl, kc] += lib.einsum('lad,ac->lcd', r2[kl, kc], imds.Lvv[kc])
                Hr2[kl, kc] += lib.einsum('lcb,bd->lcd', r2[kl, kc], imds.Lvv[kd])
                Hr2[kl, kc] += -lib.einsum('jcd,lj->lcd', r2[kl, kc], imds.Loo[kl])

                SWoovv = (2. * imds.Woovv[kl, kshift, kd] -
                               imds.Woovv[kl, kshift, kc].transpose(0, 1, 3, 2))
                Hr2 += -np.einsum('ljdc,j->lcd', SWoovv, r2t2_tmp)

                for kb in range(nkpts):
                    kj = kconserv[kl, kd, kb]
                    SWovvo = (2. * imds.Wovvo[kl, kb, kd] -
                                   imds.Wovov[kl, kb, kj].transpose(0, 1, 3, 2))
                    Hr2[kl, kc] += lib.einsum('jcb,lbdj->lcd', r2[kj, kc], SWovvo)
                    kj = kconserv[kl, kc, kb]
                    Hr2[kl, kc] += -lib.einsum('lbjc,jbd->lcd', imds.Wovov[kl, kb, kj], r2[kj, kb])
                    Hr2[kl, kc] += -lib.einsum('lbcj,jdb->lcd', imds.Wovvo[kl, kb, kc], r2[kj, kd])

                    ka = kconserv[kc, kb, kd]
                    Hr2[kl, kc] += lib.einsum('lab,abcd->lcd', r2[kl, ka], imds.Wvvvv[ka, kb, kc])

        return self.mask_frozen_ea(self.ea_amplitudes_to_vector(Hr1, Hr2), kshift, const=0.0)

    def eaccsd_diag(self, kshift, with_t3p2=False, with_t3p2_imds=False):
        if not hasattr(self, 'imds'):
            self.imds = _IMDS(self, with_t3p2=with_t3p2, with_t3p2_imds=with_t3p2_imds)
        self.imds.ea_partition = self.ea_partition
        if not self.imds.made_ea_imds:
            self.imds.make_ea(self.ea_partition, with_t3p2=with_t3p2, with_t3p2_imds=with_t3p2_imds)
        imds = self.imds

        t1, t2 = self.t1, self.t2
        nkpts, nocc, nvir = t1.shape
        kconserv = self.khelper.kconserv

        Hr1 = np.diag(imds.Lvv[kshift])

        Hr2 = np.zeros((nkpts, nkpts, nocc, nvir, nvir), dtype=t2.dtype)
        if self.ea_partition == 'mp':
            foo = self.eris.fock[:, :nocc, :nocc]
            fvv = self.eris.fock[:, nocc:, nocc:]
            for kj in range(nkpts):
                for ka in range(nkpts):
                    kb = kconserv[kshift, ka, kj]
                    Hr2[kj, ka] -= foo[kj].diagonal()[:, None, None]
                    Hr2[kj, ka] += fvv[ka].diagonal()[None, :, None]
                    Hr2[kj, ka] += fvv[kb].diagonal()
        else:
            idx = np.eye(nvir, dtype=bool)
            for kj in range(nkpts):
                for ka in range(nkpts):
                    kb = kconserv[kshift, ka, kj]
                    Hr2[kj, ka] -= imds.Loo[kj].diagonal()[:, None, None]
                    Hr2[kj, ka] += imds.Lvv[ka].diagonal()[None, :, None]
                    Hr2[kj, ka] += imds.Lvv[kb].diagonal()

                    Hr2[kj, ka] += np.einsum('abab->ab', imds.Wvvvv[ka, kb, ka])

                    Hr2[kj, ka] -= np.einsum('jbjb->jb', imds.Wovov[kj, kb, kj])[:, None, :]
                    Wovvo = np.einsum('jbbj->jb', imds.Wovvo[kj, kb, kb])
                    Hr2[kj, ka] += 2. * Wovvo[:, None, :]
                    if ka == kb:
                        for a in range(nvir):
                            Hr2[kj, ka, :, a, a] -= Wovvo[:, a]

                    Hr2[kj, ka] -= np.einsum('jaja->ja', imds.Wovov[kj, ka, kj])[:, :, None]

                    Hr2[kj, ka] -= 2 * np.einsum('ijab,ijab->jab', t2[kshift, kj, ka], imds.Woovv[kshift, kj, ka])
                    Hr2[kj, ka] += np.einsum('ijab,ijba->jab', t2[kshift, kj, ka], imds.Woovv[kshift, kj, kb])

        return self.ea_amplitudes_to_vector(Hr1, Hr2)

    def _eaccsd_star(self, eaccsd_evals, eaccsd_evecs, leaccsd_evecs, kptlist):
        """Calculates perturbative correction EA-CCSD*

        Args:
            eom (:obj:`EOMEA`):
                Object containing coupled-cluster results.
            eaccsd_evals (array-like):
                Right EOM-EA-CCSD eigenvalues; should be same as left eigenvalues.
            eaccsd_evecs (array-like):
                List of right EOM-EA-CCSD eigenvectors.
            leaccsd_evecs (array-like):
                List of left EOM-EA-CCSD eigenvectors.
            kptlist (list):
                List of k-point indices for which eigenvalue perturbations are requested.

        Returns:
            e_star (list of :obj:`ndarray`):
                The EA-CCSD* energies at each kpoint.

        Notes:
            The user should check to make sure the right and left eigenvalues agree
            before running the perturbative correction.

            The 2hp right amplitudes are assumed to be of the form s^{a }_{ij}, i.e.
            the (ia) indices are coupled while the left are assumed to be of the form
            s^{ b}_{ij}, i.e. the (jb) indices are coupled.

        Reference:
            Saeh, Stanton "...energy surfaces of radicals" JCP 111, 8275 (1999)

        """
        from itertools import product
        from pyscf.pbc.lib import kpts_helper
        time0 = time.clock(), time.time()

        assert(self.ea_partition == None)
        eris = self.eris
        #assert(isinstance(eris, gccsd._PhysicistsERIs))
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        size = nvir + nkpts*nkpts*nocc*nvir*nvir
        kconserv = self.khelper.kconserv

        eaccsd_evecs, leaccsd_evecs = atleast_3d(eaccsd_evecs), atleast_3d(leaccsd_evecs)
        eaccsd_evals = np.atleast_2d(eaccsd_evals)
        # TODO: generate checks for input array sizes
        assert(eaccsd_evecs.shape[-1] == size)
        assert(eaccsd_evecs.shape == leaccsd_evecs.shape)

        t1, t2 = self.t1, self.t2
        foo = [eris.fock[ikpt,:nocc,:nocc].diagonal() for ikpt in range(nkpts)]
        fvv = [eris.fock[ikpt,nocc:,nocc:].diagonal() for ikpt in range(nkpts)]

        eaccsd_evecs  = np.array(eaccsd_evecs)
        leaccsd_evecs = np.array(leaccsd_evecs)
        e_star = []
        for ikshift, kshift in enumerate(kptlist):
            e_star_kpt = []
            for ea_eval, ea_evec, ea_levec in zip(eaccsd_evals[ikshift],
                eaccsd_evecs[ikshift], leaccsd_evecs[ikshift]):

                # Enforcing <L|R> = 1
                l1, l2 = self.ea_vector_to_amplitudes(ea_levec)
                r1, r2 = self.ea_vector_to_amplitudes(ea_evec)
                ldotr = np.dot(l1, r1) + np.dot(l2.ravel(), r2.ravel())

                logger.info(self, 'Left-right amplitude overlap : %14.8e', ldotr.real)
                if abs(ldotr) < 1e-7:
                    logger.warn(self, 'Small %s left-right amplitude overlap. Results '
                                     'may be inaccurate.', ldotr.real)

                l1 /= ldotr
                l2 /= ldotr

                deltaE = 0.0 + 0.0*1j
                for ki, kj, ka, kb in product(range(nkpts), repeat=4):
                    fac = 1.
                    if ka > kb:
                        continue
                    fac = (2.0 - float(ka==kb))

                    if ka == kb:
                        if ki > kj:
                            continue
                        else:
                            fac *= (2.0 - float(ki==kj))

                    kc = kpts_helper.get_kconserv3(self._scf.cell, self.kpts,
                                                   [ki, kj ,kshift, ka, kb])
                    lijabc_tmp = np.zeros((nocc,nocc,nvir,nvir,nvir), dtype=t2.dtype)
                    rijabc_tmp = np.zeros((nocc,nocc,nvir,nvir,nvir), dtype=t2.dtype)

                    #(a,b,c) -> (a,b,c)
                    if kc == kshift and kb == kconserv[ki,ka,kj]:
                        tmp = -0.5*einsum('ijab,c->ijabc',eris.oovv[ki,kj,ka],l1)
                        lijabc_tmp += 8.*tmp

                    #(b,a,c) -> (a,b,c)
                    if kc == kshift and ka == kconserv[ki,kb,kj]:
                        tmp = -0.5*einsum('ijba,c->ijabc',eris.oovv[ki,kj,kb],l1)
                        lijabc_tmp += -4.* tmp

                    #(c,b,a) -> (a,b,c)
                    if ka == kshift and kb == kconserv[ki,kc,kj]:
                        tmp = -0.5*einsum('ijcb,a->ijabc',eris.oovv[ki,kj,kc],l1)
                        lijabc_tmp += -4.* tmp

                    #(a,c,b) -> (a,b,c)
                    if kb == kshift and kc == kconserv[ki,ka,kj]:
                        tmp = -0.5*einsum('ijac,b->ijabc',eris.oovv[ki,kj,ka],l1)
                        lijabc_tmp += -4.* tmp

                    #(b,c,a) -> (a,b,c)
                    if kb == kshift and ka == kconserv[ki,kc,kj]:
                        tmp = -0.5*einsum('ijca,b->ijabc',eris.oovv[ki,kj,kc],l1)
                        lijabc_tmp += 2.* tmp

                    #(c,a,b) -> (a,b,c)
                    if ka == kshift and kc == kconserv[ki,kb,kj]:
                        tmp = -0.5*einsum('ijbc,a->ijabc',eris.oovv[ki,kj,kb],l1)
                        lijabc_tmp += 2.* tmp

                    # Beginning of ooov terms

                    #(a,b,c) -> (a,b,c)
                    km = kconserv[ki,ka,kj]
                    tmp = einsum('jima,mcb->ijabc',eris.ooov[kj,ki,km],l2[km, kc])
                    lijabc_tmp += 2.* tmp

                    km = kconserv[kj,kb,ki]
                    tmp = einsum('ijmb,mca->ijabc',eris.ooov[ki,kj,km],l2[km, kc])
                    lijabc_tmp += 2.* tmp

                    #(b,a,c) -> (a,b,c)
                    km = kconserv[ki,kb,kj]
                    tmp = einsum('jimb,mca->ijabc',eris.ooov[kj,ki,km],l2[km, kc])
                    lijabc_tmp += -1.* tmp

                    km = kconserv[kj,ka,ki]
                    tmp = einsum('ijma,mcb->ijabc',eris.ooov[ki,kj,km],l2[km, kc])
                    lijabc_tmp += -1.* tmp

                    #(c,b,a) -> (a,b,c)
                    km = kconserv[ki,kc,kj]
                    tmp = einsum('jimc,mab->ijabc',eris.ooov[kj,ki,km],l2[km, ka])
                    lijabc_tmp += -1.* tmp

                    km = kconserv[kj,kc,ki]
                    tmp = einsum('ijmc,mba->ijabc',eris.ooov[ki,kj,km],l2[km, kb])
                    lijabc_tmp += -1.* tmp

                    # vovv part 1

                    #(c,b,a) -> (a,b,c)
                    ke = kconserv[kshift,ka,ki]
                    tmp = -einsum('ejcb,iea->ijabc',eris.vovv[ke,kj,kc],l2[ki,ke])
                    lijabc_tmp += 2.* tmp

                    ke = kconserv[kshift,kb,kj]
                    tmp = -einsum('eica,jeb->ijabc',eris.vovv[ke,ki,kc],l2[kj,ke])
                    lijabc_tmp += 2.* tmp

                    #(b,c,a) -> (a,b,c)
                    ke = kconserv[kshift,kb,kj]
                    tmp = -einsum('eiac,jeb->ijabc',eris.vovv[ke,ki,ka],l2[kj,ke])
                    lijabc_tmp += -1.* tmp

                    ke = kconserv[kshift,kb,ki]
                    tmp = -einsum('ejca,ieb->ijabc',eris.vovv[ke,kj,kc],l2[ki,ke])
                    lijabc_tmp += -1.* tmp

                    #(c,a,b) -> (a,b,c)
                    ke = kconserv[kshift,ka,kj]
                    tmp = -einsum('eicb,jea->ijabc',eris.vovv[ke,ki,kc],l2[kj,ke])
                    lijabc_tmp += -1.* tmp

                    ke = kconserv[kshift,ka,ki]
                    tmp = -einsum('ejbc,iea->ijabc',eris.vovv[ke,kj,kb],l2[ki,ke])
                    lijabc_tmp += -1.* tmp

                    # vovv part 2

                    #(c,b,a) -> (a,b,c)
                    ke = kconserv[kshift,kc,ki]
                    tmp = -einsum('ejab,ice->ijabc',eris.vovv[ke,kj,ka],l2[ki,kc])
                    lijabc_tmp += 2.* tmp

                    ke = kconserv[kshift,kc,kj]
                    tmp = -einsum('eiba,jce->ijabc',eris.vovv[ke,ki,kb],l2[kj,kc])
                    lijabc_tmp += 2.* tmp

                    #(b,c,a) -> (a,b,c)
                    ke = kconserv[kshift,kc,ki]
                    tmp = -einsum('ejba,ice->ijabc',eris.vovv[ke,kj,kb],l2[ki,kc])
                    lijabc_tmp += -1.* tmp

                    ke = kconserv[kshift,ka,kj]
                    tmp = -einsum('eibc,jae->ijabc',eris.vovv[ke,ki,kb],l2[kj,ka])
                    lijabc_tmp += -1.* tmp

                    #(c,a,b) -> (a,b,c)
                    ke = kconserv[kshift,kb,ki]
                    tmp = -einsum('ejac,ibe->ijabc',eris.vovv[ke,kj,ka],l2[ki,kb])
                    lijabc_tmp += -1.* tmp

                    ke = kconserv[kshift,kc,kj]
                    tmp = -einsum('eiab,jce->ijabc',eris.vovv[ke,ki,ka],l2[kj,kc])
                    lijabc_tmp += -1.* tmp

                    # Starting the right amplitude equations

                    # (ia,jb) -> (ia,jb)
                    ke = kconserv[ki,ka,kj]
                    tmp2 = einsum('bcef,f->bce',eris.vvvv[kb,kc,ke],r1)
                    tmp = - einsum('bce,ijae->ijabc',tmp2,t2[ki, kj, ka])
                    rijabc_tmp += tmp

                    km = kconserv[kshift,kc,kj]
                    tmp2 = einsum('mcje,e->mcj',eris.ovov[km,kc,kj],r1)
                    tmp = einsum('mcj,imab->ijabc',tmp2,t2[ki, km, ka])
                    rijabc_tmp += tmp

                    km = kconserv[kc,ki,ka]
                    ke = kshift
                    tmp2 = einsum('bmje,e->mbj',eris.voov[kb,km,kj],r1)
                    tmp = einsum('mbj,imac->ijabc',tmp2,t2[ki, km, ka])
                    rijabc_tmp += tmp

                    km = kconserv[ki,ka,kj]
                    tmp = einsum('jima,mcb->ijabc',eris.ooov[kj,ki,km].conj(),r2[km,kc])
                    rijabc_tmp += tmp

                    ke = kconserv[kshift,kb,kj]
                    tmp = -einsum('eica,jeb->ijabc',eris.vovv[ke,ki,kc].conj(),r2[kj,ke])
                    rijabc_tmp += tmp

                    ke = kconserv[kshift,kc,kj]
                    tmp = -einsum('eiba,jce->ijabc',eris.vovv[ke,ki,kb].conj(),r2[kj,kc])
                    rijabc_tmp += tmp

                    # (ia,jb) -> (jb,ia)
                    ke = kconserv[kj,kb,ki]
                    tmp2 = einsum('acef,f->ace',eris.vvvv[ka,kc,ke],r1)
                    tmp = - einsum('ace,jibe->ijabc',tmp2,t2[kj, ki, kb])
                    rijabc_tmp += tmp

                    km = kconserv[kshift,kc,ki]
                    tmp2 = einsum('mcie,e->mci',eris.ovov[km,kc,ki],r1)
                    tmp = einsum('mci,jmba->ijabc',tmp2,t2[kj, km, kb])
                    rijabc_tmp += tmp

                    km = kconserv[kc,kj,kb]
                    ke = kshift
                    tmp2 = einsum('amie,e->mai',eris.voov[ka,km,ki],r1)
                    tmp = einsum('mai,jmbc->ijabc',tmp2,t2[kj, km, kb])
                    rijabc_tmp += tmp

                    km = kconserv[kj,kb,ki]
                    tmp = einsum('ijmb,mca->ijabc',eris.ooov[ki,kj,km].conj(),r2[km,kc])
                    rijabc_tmp += tmp

                    ke = kconserv[kshift,ka,ki]
                    tmp = -einsum('ejcb,iea->ijabc',eris.vovv[ke,kj,kc].conj(),r2[ki,ke])
                    rijabc_tmp += tmp

                    ke = kconserv[kshift,kc,ki]
                    tmp = -einsum('ejab,ice->ijabc',eris.vovv[ke,kj,ka].conj(),r2[ki,kc])
                    rijabc_tmp += tmp

                    eij = foo[ki][:, None] + foo[kj][None, :]
                    eabc = (fvv[ka][:, None, None] + fvv[kb][None, :, None] +
                            fvv[kc][None, None, :])
                    eijabc = eij[:, :, None, None, None] - eabc[None, None, :, :, :]
                    denom = eijabc + ea_eval
                    denom = 1. / denom

                    deltaE += fac*0.5*einsum('ijabc,ijabc,ijabc',lijabc_tmp,rijabc_tmp,denom)

                logger.info(self, "Exc. energy, delta energy = %16.12f, %16.12f",
                            ea_eval + deltaE.real, deltaE.real)
                e_star_kpt.append(ea_eval + deltaE.real)

            e_star.append(e_star_kpt)
        return e_star

    def mask_frozen_ea(self, vector, kshift, const=LARGE_DENOM):
        '''Replaces all frozen orbital indices of `vector` with the value `const`.'''
        r1, r2 = self.ea_vector_to_amplitudes(vector)
        nkpts, nocc, nvir = self.t1.shape
        kconserv = self.khelper.kconserv

        # Get location of padded elements in occupied and virtual space
        nonzero_opadding, nonzero_vpadding = padding_k_idx(self, kind="split")

        new_r1 = const * np.ones_like(r1)
        new_r2 = const * np.ones_like(r2)

        new_r1[nonzero_vpadding[kshift]] = r1[nonzero_vpadding[kshift]]
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[kshift, ka, kj]
                idx = np.ix_([kj], [ka], nonzero_opadding[kj], nonzero_vpadding[ka], nonzero_vpadding[kb])
                new_r2[idx] = r2[idx]

        return self.ea_amplitudes_to_vector(new_r1, new_r2)


KRCCSD = RCCSD

class _ERIS:  # (pyscf.cc.ccsd._ChemistsERIs):
    def __init__(self, cc, mo_coeff=None, method='incore'):
        from pyscf.pbc import df
        from pyscf.pbc import tools
        from pyscf.pbc.cc.ccsd import _adjust_occ
        log = logger.Logger(cc.stdout, cc.verbose)
        cput0 = (time.clock(), time.time())
        moidx = get_frozen_mask(cc)
        cell = cc._scf.cell
        kpts = cc.kpts
        nkpts = cc.nkpts
        nocc = cc.nocc
        nmo = cc.nmo
        nvir = nmo - nocc

        # if any(nocc != np.count_nonzero(cc._scf.mo_occ[k]>0)
        #       for k in range(nkpts)):
        #    raise NotImplementedError('Different occupancies found for different k-points')

        if mo_coeff is None:
            mo_coeff = cc.mo_coeff
        dtype = mo_coeff[0].dtype

        mo_coeff = self.mo_coeff = padded_mo_coeff(cc, mo_coeff)

        # Re-make our fock MO matrix elements from density and fock AO
        dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
        with lib.temporary_env(cc._scf, exxdiv=None):
            # _scf.exxdiv affects eris.fock. HF exchange correction should be
            # excluded from the Fock matrix.
            fockao = cc._scf.get_hcore() + cc._scf.get_veff(cell, dm)
        self.fock = np.asarray([reduce(np.dot, (mo.T.conj(), fockao[k], mo))
                                for k, mo in enumerate(mo_coeff)])

        self.mo_energy = [self.fock[k].diagonal().real for k in range(nkpts)]
        # Add HFX correction in the self.mo_energy to improve convergence in
        # CCSD iteration. It is useful for the 2D systems since their occupied and
        # the virtual orbital energies may overlap which may lead to numerical
        # issue in the CCSD iterations.
        # FIXME: Whether to add this correction for other exxdiv treatments?
        # Without the correction, MP2 energy may be largely off the correct value.
        madelung = tools.madelung(cell, kpts)
        self.mo_energy = [_adjust_occ(mo_e, nocc, -madelung)
                          for k, mo_e in enumerate(self.mo_energy)]

        # Get location of padded elements in occupied and virtual space.
        nocc_per_kpt = get_nocc(cc, per_kpoint=True)
        nonzero_padding = padding_k_idx(cc, kind="joint")

        # Check direct and indirect gaps for possible issues with CCSD convergence.
        mo_e = [self.mo_energy[kp][nonzero_padding[kp]] for kp in range(nkpts)]
        mo_e = np.sort([y for x in mo_e for y in x])  # Sort de-nested array
        gap = mo_e[np.sum(nocc_per_kpt)] - mo_e[np.sum(nocc_per_kpt)-1]
        if gap < 1e-5:
            logger.warn(cc, 'HOMO-LUMO gap %s too small for KCCSD. '
                            'May cause issues in convergence.', gap)

        mem_incore, mem_outcore, mem_basic = _mem_usage(nkpts, nocc, nvir)
        mem_now = lib.current_memory()[0]
        fao2mo = cc._scf.with_df.ao2mo

        kconserv = cc.khelper.kconserv
        khelper = cc.khelper
        orbo = np.asarray(mo_coeff[:,:,:nocc], order='C')
        orbv = np.asarray(mo_coeff[:,:,nocc:], order='C')

        if (method == 'incore' and (mem_incore + mem_now < cc.max_memory)
                or cell.incore_anyway):
            log.info('using incore ERI storage')
            self.oooo = np.empty((nkpts,nkpts,nkpts,nocc,nocc,nocc,nocc), dtype=dtype)
            self.ooov = np.empty((nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir), dtype=dtype)
            self.oovv = np.empty((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=dtype)
            self.ovov = np.empty((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=dtype)
            self.voov = np.empty((nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), dtype=dtype)
            self.vovv = np.empty((nkpts,nkpts,nkpts,nvir,nocc,nvir,nvir), dtype=dtype)
            #self.vvvv = np.empty((nkpts,nkpts,nkpts,nvir,nvir,nvir,nvir), dtype=dtype)
            self.vvvv = cc._scf.with_df.ao2mo_7d(orbv, factor=1./nkpts).transpose(0,2,1,3,5,4,6)

            for (ikp,ikq,ikr) in khelper.symm_map.keys():
                iks = kconserv[ikp,ikq,ikr]
                eri_kpt = fao2mo((mo_coeff[ikp],mo_coeff[ikq],mo_coeff[ikr],mo_coeff[iks]),
                                 (kpts[ikp],kpts[ikq],kpts[ikr],kpts[iks]), compact=False)
                if dtype == np.float: eri_kpt = eri_kpt.real
                eri_kpt = eri_kpt.reshape(nmo, nmo, nmo, nmo)
                for (kp, kq, kr) in khelper.symm_map[(ikp, ikq, ikr)]:
                    eri_kpt_symm = khelper.transform_symm(eri_kpt, kp, kq, kr).transpose(0, 2, 1, 3)
                    self.oooo[kp, kr, kq] = eri_kpt_symm[:nocc, :nocc, :nocc, :nocc] / nkpts
                    self.ooov[kp, kr, kq] = eri_kpt_symm[:nocc, :nocc, :nocc, nocc:] / nkpts
                    self.oovv[kp, kr, kq] = eri_kpt_symm[:nocc, :nocc, nocc:, nocc:] / nkpts
                    self.ovov[kp, kr, kq] = eri_kpt_symm[:nocc, nocc:, :nocc, nocc:] / nkpts
                    self.voov[kp, kr, kq] = eri_kpt_symm[nocc:, :nocc, :nocc, nocc:] / nkpts
                    self.vovv[kp, kr, kq] = eri_kpt_symm[nocc:, :nocc, nocc:, nocc:] / nkpts
                    #self.vvvv[kp, kr, kq] = eri_kpt_symm[nocc:, nocc:, nocc:, nocc:] / nkpts

            self.dtype = dtype
        else:
            log.info('using HDF5 ERI storage')
            self.feri1 = lib.H5TmpFile()

            self.oooo = self.feri1.create_dataset('oooo', (nkpts, nkpts, nkpts, nocc, nocc, nocc, nocc), dtype.char)
            self.ooov = self.feri1.create_dataset('ooov', (nkpts, nkpts, nkpts, nocc, nocc, nocc, nvir), dtype.char)
            self.oovv = self.feri1.create_dataset('oovv', (nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype.char)
            self.ovov = self.feri1.create_dataset('ovov', (nkpts, nkpts, nkpts, nocc, nvir, nocc, nvir), dtype.char)
            self.voov = self.feri1.create_dataset('voov', (nkpts, nkpts, nkpts, nvir, nocc, nocc, nvir), dtype.char)
            self.vovv = self.feri1.create_dataset('vovv', (nkpts, nkpts, nkpts, nvir, nocc, nvir, nvir), dtype.char)

            if (not (cc.direct and type(cc._scf.with_df) is df.GDF)
                or cell.dimension == 2):
                self.vvvv = self.feri1.create_dataset('vvvv', (nkpts,nkpts,nkpts,nvir,nvir,nvir,nvir), dtype.char)

            # <ij|pq>  = (ip|jq)
            cput1 = time.clock(), time.time()
            for kp in range(nkpts):
                for kq in range(nkpts):
                    for kr in range(nkpts):
                        ks = kconserv[kp, kq, kr]
                        orbo_p = mo_coeff[kp][:, :nocc]
                        orbo_r = mo_coeff[kr][:, :nocc]
                        buf_kpt = fao2mo((orbo_p, mo_coeff[kq], orbo_r, mo_coeff[ks]),
                                         (kpts[kp], kpts[kq], kpts[kr], kpts[ks]), compact=False)
                        if mo_coeff[0].dtype == np.float: buf_kpt = buf_kpt.real
                        buf_kpt = buf_kpt.reshape(nocc, nmo, nocc, nmo).transpose(0, 2, 1, 3)
                        self.dtype = buf_kpt.dtype
                        self.oooo[kp, kr, kq, :, :, :, :] = buf_kpt[:, :, :nocc, :nocc] / nkpts
                        self.ooov[kp, kr, kq, :, :, :, :] = buf_kpt[:, :, :nocc, nocc:] / nkpts
                        self.oovv[kp, kr, kq, :, :, :, :] = buf_kpt[:, :, nocc:, nocc:] / nkpts
            cput1 = log.timer_debug1('transforming oopq', *cput1)

            # <ia|pq> = (ip|aq)
            cput1 = time.clock(), time.time()
            for kp in range(nkpts):
                for kq in range(nkpts):
                    for kr in range(nkpts):
                        ks = kconserv[kp, kq, kr]
                        orbo_p = mo_coeff[kp][:, :nocc]
                        orbv_r = mo_coeff[kr][:, nocc:]
                        buf_kpt = fao2mo((orbo_p, mo_coeff[kq], orbv_r, mo_coeff[ks]),
                                         (kpts[kp], kpts[kq], kpts[kr], kpts[ks]), compact=False)
                        if mo_coeff[0].dtype == np.float: buf_kpt = buf_kpt.real
                        buf_kpt = buf_kpt.reshape(nocc, nmo, nvir, nmo).transpose(0, 2, 1, 3)
                        self.ovov[kp, kr, kq, :, :, :, :] = buf_kpt[:, :, :nocc, nocc:] / nkpts
                        # TODO: compute vovv on the fly
                        self.vovv[kr, kp, ks, :, :, :, :] = buf_kpt[:, :, nocc:, nocc:].transpose(1, 0, 3, 2) / nkpts
                        self.voov[kr, kp, ks, :, :, :, :] = buf_kpt[:, :, nocc:, :nocc].transpose(1, 0, 3, 2) / nkpts
            cput1 = log.timer_debug1('transforming ovpq', *cput1)

            ## Without k-point symmetry
            # cput1 = time.clock(), time.time()
            # for kp in range(nkpts):
            #    for kq in range(nkpts):
            #        for kr in range(nkpts):
            #            ks = kconserv[kp,kq,kr]
            #            orbv_p = mo_coeff[kp][:,nocc:]
            #            orbv_q = mo_coeff[kq][:,nocc:]
            #            orbv_r = mo_coeff[kr][:,nocc:]
            #            orbv_s = mo_coeff[ks][:,nocc:]
            #            for a in range(nvir):
            #                orbva_p = orbv_p[:,a].reshape(-1,1)
            #                buf_kpt = fao2mo((orbva_p,orbv_q,orbv_r,orbv_s),
            #                                 (kpts[kp],kpts[kq],kpts[kr],kpts[ks]), compact=False)
            #                if mo_coeff[0].dtype == np.float: buf_kpt = buf_kpt.real
            #                buf_kpt = buf_kpt.reshape((1,nvir,nvir,nvir)).transpose(0,2,1,3)
            #                self.vvvv[kp,kr,kq,a,:,:,:] = buf_kpt[:] / nkpts
            # cput1 = log.timer_debug1('transforming vvvv', *cput1)

            cput1 = time.clock(), time.time()
            mem_now = lib.current_memory()[0]
            if (cc.direct and type(cc._scf.with_df) is df.GDF
                and cell.dimension != 2):
                # cc._scf.with_df needs to be df.GDF only (not MDF)
                _init_df_eris(cc, self)

            elif nvir ** 4 * 16 / 1e6 + mem_now < cc.max_memory:
                for (ikp, ikq, ikr) in khelper.symm_map.keys():
                    iks = kconserv[ikp, ikq, ikr]
                    orbv_p = mo_coeff[ikp][:, nocc:]
                    orbv_q = mo_coeff[ikq][:, nocc:]
                    orbv_r = mo_coeff[ikr][:, nocc:]
                    orbv_s = mo_coeff[iks][:, nocc:]
                    # unit cell is small enough to handle vvvv in-core
                    buf_kpt = fao2mo((orbv_p,orbv_q,orbv_r,orbv_s),
                                     kpts[[ikp,ikq,ikr,iks]], compact=False)
                    if dtype == np.float: buf_kpt = buf_kpt.real
                    buf_kpt = buf_kpt.reshape((nvir, nvir, nvir, nvir))
                    for (kp, kq, kr) in khelper.symm_map[(ikp, ikq, ikr)]:
                        buf_kpt_symm = khelper.transform_symm(buf_kpt, kp, kq, kr).transpose(0, 2, 1, 3)
                        self.vvvv[kp, kr, kq] = buf_kpt_symm / nkpts
            else:
                raise MemoryError('Minimal memory requirements %s MB'
                                  % (mem_now + nvir ** 4 / 1e6 * 16 * 2))
                for (ikp, ikq, ikr) in khelper.symm_map.keys():
                    for a in range(nvir):
                        orbva_p = orbv_p[:, a].reshape(-1, 1)
                        buf_kpt = fao2mo((orbva_p, orbv_q, orbv_r, orbv_s),
                                         (kpts[ikp], kpts[ikq], kpts[ikr], kpts[iks]), compact=False)
                        if mo_coeff[0].dtype == np.float: buf_kpt = buf_kpt.real
                        buf_kpt = buf_kpt.reshape((1, nvir, nvir, nvir)).transpose(0, 2, 1, 3)

                        self.vvvv[ikp, ikr, ikq, a, :, :, :] = buf_kpt[0, :, :, :] / nkpts
                        # Store symmetric permutations
                        self.vvvv[ikr, ikp, iks, :, a, :, :] = buf_kpt.transpose(1, 0, 3, 2)[:, 0, :, :] / nkpts
                        self.vvvv[ikq, iks, ikp, :, :, a, :] = buf_kpt.transpose(2, 3, 0, 1).conj()[:, :, 0, :] / nkpts
                        self.vvvv[iks, ikq, ikr, :, :, :, a] = buf_kpt.transpose(3, 2, 1, 0).conj()[:, :, :, 0] / nkpts
            cput1 = log.timer_debug1('transforming vvvv', *cput1)

        log.timer('CCSD integral transformation', *cput0)


def _init_df_eris(cc, eris):
    from pyscf.pbc.df import df
    from pyscf.ao2mo import _ao2mo
    if cc._scf.with_df._cderi is None:
        cc._scf.with_df.build()

    cell = cc._scf.cell
    if cell.dimension == 2:
        raise NotImplementedError

    nocc = cc.nocc
    nmo = cc.nmo
    nvir = nmo - nocc
    nao = cell.nao_nr()

    kpts = cc.kpts
    nkpts = len(kpts)
    naux = cc._scf.with_df.get_naoaux()
    if gamma_point(kpts):
        dtype = np.double
    else:
        dtype = np.complex128
    dtype = np.result_type(dtype, *eris.mo_coeff)
    eris.Lpv = np.empty((nkpts, nkpts, naux, nmo, nvir), dtype=dtype)

    with h5py.File(cc._scf.with_df._cderi, 'r') as f:
        kptij_lst = f['j3c-kptij'].value
        tao = []
        ao_loc = None
        for ki, kpti in enumerate(kpts):
            for kj, kptj in enumerate(kpts):
                kpti_kptj = np.array((kpti, kptj))
                Lpq = np.asarray(df._getitem(f, 'j3c', kpti_kptj, kptij_lst))

                mo = np.hstack((eris.mo_coeff[ki], eris.mo_coeff[kj][:, nocc:]))
                mo = np.asarray(mo, dtype=dtype, order='F')
                if dtype == np.double:
                    _ao2mo.nr_e2(Lpq, mo, (0, nmo, nmo, nmo + nvir), aosym='s2',
                                 out=eris.Lpv[ki, kj])
                else:
                    if Lpq.size != naux * nao ** 2:  # aosym = 's2'
                        Lpq = lib.unpack_tril(Lpq).astype(np.complex128)
                    _ao2mo.r_e2(Lpq, mo, (0, nmo, nmo, nmo + nvir), tao, ao_loc,
                                out=eris.Lpv[ki, kj])
    return eris


imd = imdk


class _IMDS(object):
    # Identical to molecular rccsd_slow
    def __init__(self, cc, with_t3p2=False, copy_amps_t3p2=True, with_t3p2_imds=False):
        self.verbose = cc.verbose
        self.stdout = cc.stdout
        if copy_amps_t3p2 is True:
            self.t1 = cc.t1.copy()
            self.t2 = cc.t2.copy()
        else:
            self.t1 = cc.t1
            self.t2 = cc.t2
        self._cc = cc  # TODO: delete me; causes some issues wiht h5py
        self.eris = cc.eris
        self.kconserv = cc.khelper.kconserv
        self.made_ip_imds = False
        self.made_ea_imds = False
        self._made_shared_2e = False
        # TODO: check whether to hold all stuff in memory
        self._fimd = lib.H5TmpFile() if hasattr(self.eris, "feri1") else None
        self._ip_partition = None
        self._ea_partition = None
        self._made_t3p2_correction = False

        self.with_t3p2 = with_t3p2
        self.with_t3p2_imds = with_t3p2_imds
        self.copy_amps_t3p2 = copy_amps_t3p2

    def _make_shared_1e(self):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris
        kconserv = self.kconserv
        self.Loo = imd.Loo(t1, t2, eris, kconserv)
        self.Lvv = imd.Lvv(t1, t2, eris, kconserv)
        self.Fov = imd.cc_Fov(t1, t2, eris, kconserv)

        log.timer('EOM-CCSD shared one-electron intermediates', *cput0)

    def _make_shared_2e(self):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris
        kconserv = self.kconserv

        if self._fimd is not None:
            nkpts, nocc, nvir = t1.shape
            ovov_dest = self._fimd.create_dataset('ovov', (nkpts, nkpts, nkpts, nocc, nvir, nocc, nvir), t1.dtype.char)
            ovvo_dest = self._fimd.create_dataset('ovvo', (nkpts, nkpts, nkpts, nocc, nvir, nvir, nocc), t1.dtype.char)
        else:
            ovov_dest = ovvo_dest = None

        # 2 virtuals
        self.Wovov = imd.Wovov(t1, t2, eris, kconserv, ovov_dest)
        self.Wovvo = imd.Wovvo(t1, t2, eris, kconserv, ovvo_dest)
        self.Woovv = eris.oovv

        log.timer('EOM-CCSD shared two-electron intermediates', *cput0)

    def make_ip(self, ip_partition=None, **kwargs):
        self.ip_partition = ip_partition
        self.make_t3p2_imds(**kwargs)
        self._make_shared_1e()
        if self._made_shared_2e is False and ip_partition != 'mp':
            self._make_shared_2e()
            self._made_shared_2e = True

        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris
        kconserv = self.kconserv

        if self._fimd is not None:
            nkpts, nocc, nvir = t1.shape
            oooo_dest = self._fimd.create_dataset('oooo', (nkpts, nkpts, nkpts, nocc, nocc, nocc, nocc), t1.dtype.char)
            ooov_dest = self._fimd.create_dataset('ooov', (nkpts, nkpts, nkpts, nocc, nocc, nocc, nvir), t1.dtype.char)
            ovoo_dest = self._fimd.create_dataset('ovoo', (nkpts, nkpts, nkpts, nocc, nvir, nocc, nocc), t1.dtype.char)
        else:
            oooo_dest = ooov_dest = ovoo_dest = None

        # 0 or 1 virtuals
        if ip_partition != 'mp':
            self.Woooo = imd.Woooo(t1, t2, eris, kconserv, oooo_dest)
        self.Wooov = imd.Wooov(t1, t2, eris, kconserv, ooov_dest)
        self.Wovoo = imd.Wovoo(t1, t2, eris, kconserv, ovoo_dest)
        self.made_ip_imds = True
        log.timer('EOM-CCSD IP intermediates', *cput0)

    def make_t3p2_imds(self, **kwargs):
        with_t3p2 = kwargs.get('with_t3p2', False)
        copy_amps_t3p2 = kwargs.get('copy_amps_t3p2', True)
        with_t3p2_imds = kwargs.get('with_t3p2_imds', False)
        #if kwargs:
        #    raise TypeError('Unexpected **kwargs: %r' % kwargs)
        if not (with_t3p2 or with_t3p2_imds):
            return self

        cput0 = (time.clock(), time.time())

        t1, t2, eris = self.t1, self.t2, self.eris
        from pyscf.pbc.cc.t3p2_experimental import get_t3p2_amplitude_contribution as t3p2_drv
        drv = t3p2_drv
        #drv = imd.get_t3p2_amplitude_contribution_slow
        if (not self._made_t3p2_correction):
            delta_ccsd_energy, t1, t2, Wmcik, Wacek = \
                drv(self._cc, t1, t2, eris=eris, copy_amps=copy_amps_t3p2,
                    build_t1_t2=with_t3p2, build_ip_t3p2=with_t3p2_imds,
                    build_ea_t3p2=with_t3p2_imds)
            if with_t3p2_imds:
                self.Wovoo_t3p2 = Wmcik
                self.Wvvvo_t3p2 = Wacek

            self.t1 = t1
            self.t2 = t2
            self._made_t3p2_correction = True

        logger.timer_debug1(self, 'EOM-CCSD T3[2] intermediates', *cput0)
        return self

    def make_ea(self, ea_partition=None, **kwargs):
        self.ea_partition = ea_partition
        self.make_t3p2_imds(**kwargs)
        self._make_shared_1e()
        if self._made_shared_2e is False and ea_partition != 'mp':
            self._make_shared_2e()
            self._made_shared_2e = True

        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris
        kconserv = self.kconserv

        if self._fimd is not None:
            nkpts, nocc, nvir = t1.shape
            vovv_dest = self._fimd.create_dataset('vovv', (nkpts, nkpts, nkpts, nvir, nocc, nvir, nvir), t1.dtype.char)
            vvvo_dest = self._fimd.create_dataset('vvvo', (nkpts, nkpts, nkpts, nvir, nvir, nvir, nocc), t1.dtype.char)
            vvvv_dest = self._fimd.create_dataset('vvvv', (nkpts, nkpts, nkpts, nvir, nvir, nvir, nvir), t1.dtype.char)
        else:
            vovv_dest = vvvo_dest = vvvv_dest = None

        # 3 or 4 virtuals
        self.Wvovv = imd.Wvovv(t1, t2, eris, kconserv, vovv_dest)
        if ea_partition == 'mp' and np.all(t1 == 0):
            self.Wvvvo = imd.Wvvvo(t1, t2, eris, kconserv, vvvo_dest)
        else:
            self.Wvvvv = imd.Wvvvv(t1, t2, eris, kconserv, vvvv_dest)
            self.Wvvvo = imd.Wvvvo(t1, t2, eris, kconserv, self.Wvvvv, vvvo_dest)
        self.made_ea_imds = True
        log.timer('EOM-CCSD EA intermediates', *cput0)

    def make_ee(self):
        raise NotImplementedError

    @property
    def ip_partition(self):
        return self._ip_partition
    @ip_partition.setter
    def ip_partition(self, p):
        if (self._ip_partition == 'mp') and (p != 'mp'):  # If we previously had 'mp', we need to
            self.made_ip_imds = False                     # remake the shared_2e and imds
            self._made_shared_2e = False
        self._ip_partition = p

    @property
    def ea_partition(self):
        return self._ea_partition
    @ea_partition.setter
    def ea_partition(self, p):
        if (self._ea_partition == 'mp') and (p != 'mp'):  # If we previously had 'mp', we need to
            self.made_ea_imds = False                     # remake the shared_2e and imds
            self._made_shared_2e = False
        self._ea_partition = p


def _mem_usage(nkpts, nocc, nvir):
    incore = nkpts ** 3 * (nocc + nvir) ** 4
    # Roughly, factor of two for intermediates and factor of two
    # for safety (temp arrays, copying, etc)
    incore *= 4
    # TODO: Improve incore estimate and add outcore estimate
    outcore = basic = incore
    return incore * 16 / 1e6, outcore * 16 / 1e6, basic * 16 / 1e6


if __name__ == '__main__':
    from pyscf.pbc import gto, scf, cc

    cell = gto.Cell()
    cell.atom = '''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = {'C': [[0, (0.8, 1.0)],
                        [1, (1.0, 1.0)]]}
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 5
    cell.build()

    # Running HF and CCSD with 1x1x2 Monkhorst-Pack k-point mesh
    kmf = scf.KRHF(cell, kpts=cell.make_kpts([1, 1, 2]), exxdiv=None)
    ehf = kmf.kernel()

    mycc = cc.KRCCSD(kmf)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-10
    ecc, t1, t2 = mycc.kernel()
    print(ecc - -0.155298393321855)

    e_ip, _ = mycc.ipccsd(nroots=3, kptlist=(0,))
    e_ea, _ = mycc.eaccsd(nroots=3, kptlist=(0,))
    print(e_ip, e_ea)

    ####
    cell = gto.Cell()
    cell.atom = '''
    He 0.000000000000   0.000000000000   0.000000000000
    He 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.build()

    np.random.seed(2)
    # Running HF and CCSD with 1x1x2 Monkhorst-Pack k-point mesh
    kmf = scf.KRHF(cell, kpts=cell.make_kpts([1, 1, 3]), exxdiv=None)
    nmo = cell.nao_nr()
    kmf.mo_occ = np.zeros((3, nmo))
    kmf.mo_occ[:, :2] = 2
    kmf.mo_energy = np.arange(nmo) + np.random.random((3, nmo)) * .3
    kmf.mo_energy[kmf.mo_occ == 0] += 2
    kmf.mo_coeff = (np.random.random((3, nmo, nmo)) +
                    np.random.random((3, nmo, nmo)) * 1j - .5 - .5j)


    def rand_t1_t2(mycc):
        nkpts = mycc.nkpts
        nocc = mycc.nocc
        nmo = mycc.nmo
        nvir = nmo - nocc
        np.random.seed(1)
        t1 = (np.random.random((nkpts, nocc, nvir)) +
              np.random.random((nkpts, nocc, nvir)) * 1j - .5 - .5j)
        t2 = (np.random.random((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir)) +
              np.random.random((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir)) * 1j - .5 - .5j)
        kconserv = kpts_helper.get_kconserv(kmf.cell, kmf.kpts)
        Ps = kconserve_pmatrix(nkpts, kconserv)
        t2 = t2 + np.einsum('xyzijab,xyzw->yxwjiba', t2, Ps)
        return t1, t2


    mycc = KRCCSD(kmf)
    eris = mycc.ao2mo()
    t1, t2 = rand_t1_t2(mycc)
    Ht1, Ht2 = mycc.update_amps(t1, t2, eris)
    print(lib.finger(Ht1) - (-4.6808039711608824 + 9.4962987225515789j))  # FIXME
    print(lib.finger(Ht2) - (18.613685230812546 + 114.66975731912211j))  # FIXME

    kmf = kmf.density_fit(auxbasis=[[0, (1., 1.)], [0, (.5, 1.)]])
    mycc = KRCCSD(kmf)
    eris = _ERIS(mycc, mycc.mo_coeff, method='outcore')
    t1, t2 = rand_t1_t2(mycc)
    Ht1, Ht2 = mycc.update_amps(t1, t2, eris)
    print(lib.finger(Ht1) - (-3.6611794882508244 + 9.2241044317516554j))  # FIXME
    print(lib.finger(Ht2) - (-196.88536721771101 - 432.29569128644886j))  # FIXME
