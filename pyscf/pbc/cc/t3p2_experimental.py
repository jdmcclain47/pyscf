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
import pyscf.pbc.cc as pbcc
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf
from pyscf.pbc.cc import kintermediates_rhf as imdk
from pyscf.lib.parameters import LOOSE_ZERO_TOL, LARGE_DENOM
from pyscf.lib import linalg_helper
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.lib.kpts_helper import member, gamma_point
from pyscf import __config__
from itertools import product

import ctypes
import sys
from pyscf.cc import _ccsd
from pyscf.cc import eom_rccsd
from pyscf.pbc.tools.pbc import super_cell

einsum = lib.einsum

def _convert_to_int(kpt_indices):
    '''Convert all kpoint indices for 3-particle operator to integers.'''
    out_indices = [0]*6
    for ix, x in enumerate(kpt_indices):
        assert isinstance(x, (int, np.int, np.ndarray, list))
        if isinstance(x, (np.ndarray)) and (x.ndim == 0):
            out_indices[ix] = int(x)
        else:
            out_indices[ix] = x
    return out_indices

def _tile_list(kpt_indices):
    '''Similar to a cartesian product but for a list of kpoint indices for
    a 3-particle operator.'''
    max_length = 0
    out_indices = [0]*6
    for ix, x in enumerate(kpt_indices):
        if hasattr(x, '__len__'):
            max_length = max(max_length, len(x))

    if max_length == 0:
        return kpt_indices
    else:
        for ix, x in enumerate(kpt_indices):
            if isinstance(x, (int, np.int)):
                out_indices[ix] = [x] * max_length
            else:
                out_indices[ix] = x

    return map(list, zip(*out_indices))

def zip_kpoints(kpt_indices):
    '''Similar to a cartesian product but for a list of kpoint indices for
    a 3-particle operator.  Ensures all indices are integers.'''
    out_indices = _convert_to_int(kpt_indices)
    out_indices = _tile_list(out_indices)
    return out_indices

def get_data_slices(kpt_indices, orb_indices, kconserv):
    kpt_indices = zip_kpoints(kpt_indices)
    if isinstance(kpt_indices[0], (int, np.int)):  # Ensure we are working
        kpt_indices = [kpt_indices]                   # with a list of lists

    a0,a1,b0,b1,c0,c1 = orb_indices
    length = len(kpt_indices)*6

    def _vijk_indices(kpt_indices, orb_indices, transpose=(0, 1, 2)):
        '''Get indices needed for t3 construction and a given transpose of (a,b,c).'''
        kpt_indices = ([kpt_indices[x] for x in transpose] +
                       [kpt_indices[x+3] for x in transpose])
        orb_indices = lib.flatten([[orb_indices[2*x], orb_indices[2*x+1]]
                                   for x in transpose])

        ki, kj, kk, ka, kb, kc = kpt_indices
        a0, a1, b0, b1, c0, c1 = orb_indices

        kf = kconserv[ka,ki,kb]
        km = kconserv[kc,kk,kb]
        sl00 = slice(None, None)

        vvop_idx = [ka, kb, ki, slice(a0,a1), slice(b0,b1), sl00, sl00]
        vooo_idx = [ka, ki, kj, slice(a0,a1), sl00, sl00, sl00]
        t2T_vvop_idx = [kc, kf, kj, slice(c0,c1), sl00, sl00, sl00]
        t2T_vooo_idx = [kc, kb, km, slice(c0,c1), sl00, sl00, sl00]
        return vvop_idx, vooo_idx, t2T_vvop_idx, t2T_vooo_idx

    vvop_indices = [0] * length
    vooo_indices = [0] * length
    t2T_vvop_indices = [0] * length
    t2T_vooo_indices = [0] * length

    transpose = [(0, 1, 2), (0, 2, 1), (1, 0, 2),
                 (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    count = 0
    for kpt in kpt_indices:
        for t in transpose:
            vvop_idx, vooo_idx, t2T_vvop_idx, t2T_vooo_idx = _vijk_indices(kpt, orb_indices, t)
            vvop_indices[count] = vvop_idx
            vooo_indices[count] = vooo_idx
            t2T_vvop_indices[count] = t2T_vvop_idx
            t2T_vooo_indices[count] = t2T_vooo_idx
            count += 1

    return vvop_indices, vooo_indices, t2T_vvop_indices, t2T_vooo_indices

def _slice_to_hashable(slice_list):
    '''Creates a hashable tuple and descriptor from iterable of integer
    indices and slices.'''
    out = ()
    is_slice = []
    for s in slice_list:
        if isinstance(s, (int, np.int)):
            out += (s, None)
        elif isinstance(s, slice):
            out += (s.start, s.stop)
        else:
            raise TypeError
    return out

def _hashable_to_slice(*tuple_index):
    '''Generates a list of slices from tuple indices.'''
    length = len(tuple_index)
    assert length % 2 == 0
    out = [slice(tuple_index[i], tuple_index[i+1])
           for i in range(0, length, 2)]
    out = [slice_.start
           if (slice_.stop == None and slice_.start != None) else slice_
           for slice_ in out]
    return tuple(out)

class Condenser(object):
    '''Condenses jobs, removing duplicates.'''
    def __init__(self, verbose=logger.INFO):
        self.job = {}
        self.job_results = {}
        self.verbose = verbose
        self.stdout = sys.stdout
        self.NMAX = 400  # max cache-size
        self.offset = 0
        self.unique_job_args = {}
        pass

    def _job_name_results(self, job_name):
        return job_name + '_res'

    def add_job(self, job_name, func, arg,
                hash_func=_slice_to_hashable, unhash_func=_hashable_to_slice):
        if job_name not in self.job:
            self.job[job_name] = []
            self.unique_job_args[job_name] = set()
            self.job_results[job_name] = {}
        key = (func, unhash_func) + hash_func(arg)
        self.job[job_name].append(key)
        self.unique_job_args[job_name].update([key,])

    def _delete_job(self, job_name):
        del self.job[job_name]
        del self.job_results[job_name]
        del self.unique_job_args[job_name]
        self.offset = 0

    @profile
    def _submit(self, job_name):
        #logger.info(self, 'job %s reduced keys %d -> %d (%f percent)',
        #            job_name, len(self.job[job_name]), len(unique_job_args),
        #            (1. - (1.*len(unique_job_args))/len(self.job[job_name]))*100)

        # Submit only unique arguments that we don't have results for
        dict_ = self.job_results[job_name]
        for k in self.unique_job_args[job_name] ^ set(dict_.keys()):
            func = k[0]
            unhash_func = k[1]
            arg = unhash_func(*k[2:])

            ret = func(arg)
            dict_[k] = ret
        self.job_results[job_name] = dict_
        return

    def submit(self, job_name):
        if job_name not in self.job:
            raise RuntimeError
        self._submit(job_name)
        return

class DataHandler(Condenser):
    '''Wrapper for Condenser for our specific jobs'''
    @profile
    def results(self, job_name, func, args,
                hash_func=_slice_to_hashable, unhash_func=_hashable_to_slice):
        if not isinstance(args, list):
            args = [args,]
        if not isinstance(func, list):
            func = [func,] * len(args)
        out = []
        for keys in zip(func, args):
            k = (keys[0], unhash_func) + hash_func(keys[1])
            out.append(self.job_results[job_name][k])
        return out

    @profile
    def request_data(self, kpt_indices, orb_indices, kconserv, *args):
        idx_args = get_data_slices(kpt_indices, orb_indices, kconserv)
        vvop_indices, vooo_indices, t2T_vvop_indices, t2T_vooo_indices = idx_args
        for task in range(len(vvop_indices)):
            self.add_job('vvop', args[0], vvop_indices[task])
            self.add_job('vooo', args[1], vooo_indices[task])
            self.add_job('t2Tvvop', args[2], t2T_vvop_indices[task])
            self.add_job('t2Tvooo', args[3], t2T_vooo_indices[task])

        self.submit('vvop')
        self.submit('vooo')
        self.submit('t2Tvvop')
        self.submit('t2Tvooo')
        return

    def get_data(self, kpt_indices, orb_indices, kconserv, *args):
        idx_args = get_data_slices(kpt_indices, orb_indices, kconserv)
        vvop_indices, vooo_indices, t2T_vvop_indices, t2T_vooo_indices = idx_args

        vvop = self.results('vvop', args[0], vvop_indices)
        vooo = self.results('vooo', args[1], vooo_indices)
        t2Tvvop = self.results('t2Tvvop', args[2], t2T_vvop_indices)
        t2Tvooo = self.results('t2Tvooo', args[3], t2T_vooo_indices)
        return vvop, vooo, t2Tvvop, t2Tvooo

    def clean(self):
        jobs = ['vvop', 'vooo', 't2Tvvop', 't2Tvooo']
        for j in jobs:
            if len(self.job[j]) > self.NMAX:
                #logger.info(self, 'Clearing cache for job %s', j)
                self._delete_job(j)
        return

def get_full_t3p2(cc, t1, t2, eris):
    '''Build the entire T3[2] array in memory.'''
    nkpts = cc.nkpts
    nocc = cc.nocc
    nmo = cc.nmo
    nvir = nmo - nocc
    kconserv = cc.khelper.kconserv

    standard = True  # standard = True calculates the corresponding quantities
                     # using the same intermediate quantities as the C code.  Useful
                     # for debugging purposes.
    if standard:
        def get_vijkabc(ki, kj, kk, ka, kb, kc):
            '''Build T3[2] for `ijkabc` at a given set of k-points'''
            kd = kconserv[kb, kk, kc]
            ret = einsum('dkbc,ijad->ijkabc', eris.vovv[kd, kk, kb].conj(), t2[ki, kj, ka])
            km = kconserv[ka, ki, kb]
            ret -= einsum('jkmc,imab->ijkabc', eris.ooov[kj, kk, km].conj(), t2[ki, km, ka])
            return ret
    else:
        t2T = np.zeros((nkpts,)*3 + (nvir,)*2 + (nocc,)*2, dtype=np.complex, order='C')
        eris_vooo = np.zeros((nkpts,)*3 + (nvir,) + (nocc,)*3, dtype=np.complex, order='C')
        eris_vooo_C = np.zeros((nkpts,)*3 + (nvir,) + (nocc,)*3, dtype=np.complex, order='C')
        for ki, kj, ka in product(range(nkpts), repeat=3):
            kb = kconserv[ki,ka,kj]
            t2T[ka,kb,kj] = t2[ki,kj,ka].transpose(2,3,1,0)
            eris_vooo[ki,ka,kj] = eris.ooov[kb,kj,ka].conj().transpose(3,2,1,0)
            eris_vooo_C[ki,kj,kb] = eris.ooov[kb,kj,ka].conj().transpose(3,1,2,0).transpose(0,1,3,2)

        def get_vijkabc(ki, kj, kk, ka, kb, kc):
            '''Build T3[2] for `ijkabc` at a given set of k-points'''
            kd = kconserv[kb, kk, kc]
            ret = einsum('dkbc,ijad->ijkabc', eris.vovv[kd, kk, kb].conj(), t2[ki, kj, ka])
            km = kconserv[kc, kk, kb]
            ret -= einsum('aijm,kmcb->ijkabc', eris_vooo_C[ka, ki, kj], t2[kk, km, kc])
            return ret

    fock = eris.fock
    fov = fock[:, :nocc, nocc:]
    foo = np.array([fock[ikpt, :nocc, :nocc].diagonal() for ikpt in range(nkpts)])
    fvv = np.array([fock[ikpt, nocc:, nocc:].diagonal() for ikpt in range(nkpts)])

    t3 = np.empty((nkpts, nkpts, nkpts, nkpts, nkpts, nocc, nocc, nocc, nvir, nvir, nvir),
                   dtype = t2.dtype)
    for ki, kj, kk, ka, kb in product(range(nkpts), repeat=5):
        kc = kpts_helper.get_kconserv3(cc._scf.cell, cc.kpts,
                                       [ki, kj, kk, ka, kb])
        t3[ki, kj, kk, ka, kb] = get_vijkabc(ki, kj, kk, ka, kb, kc)
        t3[ki, kj, kk, ka, kb] += get_vijkabc(ki, kk, kj, ka, kc, kb).transpose(0, 2, 1, 3, 5, 4)
        t3[ki, kj, kk, ka, kb] += get_vijkabc(kj, ki, kk, kb, ka, kc).transpose(1, 0, 2, 4, 3, 5)
        t3[ki, kj, kk, ka, kb] += get_vijkabc(kj, kk, ki, kb, kc, ka).transpose(2, 0, 1, 5, 3, 4)
        t3[ki, kj, kk, ka, kb] += get_vijkabc(kk, ki, kj, kc, ka, kb).transpose(1, 2, 0, 4, 5, 3)
        t3[ki, kj, kk, ka, kb] += get_vijkabc(kk, kj, ki, kc, kb, ka).transpose(2, 1, 0, 5, 4, 3)

        eijk = foo[ki][:, None, None] + foo[kj][None, :, None] + foo[kk][None, None, :]
        eabc = fvv[ka][:, None, None] + fvv[kb][None, :, None] + fvv[kc][None, None, :]
        eijkabc = eijk[:, :, :, None, None, None] - eabc[None, None, None, :, :, :]
        t3[ki, kj, kk, ka, kb] /= eijkabc
    return t3

def add_contribution_pt1(cc, kpt_indices, orb_indices, kconserv, data, out=None):
    '''Calculate T3[2] contribution to T1 array.'''
    nocc = cc.nocc
    nmo = cc.nmo
    nvir = nmo - nocc
    nkpts = cc.nkpts

    ki,kj,kk,ka,kb,kc = kpt_indices
    a0,a1,b0,b1,c0,c1 = orb_indices
    Ptmp_t3Tv = data[0]
    eris_Soovv = data[1]
    eaa = data[2]
    dtype = np.complex
    if out is None:
        out = np.zeros((nocc,(a1-a0)), dtype=data)
    if ki == ka and kc == kconserv[kj, kb, kk]:
        out += 0.5 * einsum('abcijk,jkbc->ia', Ptmp_t3Tv, eris_Soovv) / eaa
    return out

def get_t3p2_amplitude_contribution(cc, t1, t2, eris=None, copy_amps=True,
                                    build_t1_t2=True, build_ip_t3p2=False,
                                    build_ea_t3p2=False):
    """Calculates T1, T2 amplitudes corrected by second-order T3 contribution

    Args:
        eom (:obj:`EOMIP`):
            Object containing coupled-cluster results.
        t1 (:obj:`ndarray`):
            T1 amplitudes.
        t2 (:obj:`ndarray`):
            T2 amplitudes from which the T3[2] amplitudes are formed.
        eris (:obj:`_PhysicistsERIs`):
            Antisymmetrized electron-repulsion integrals in physicist's notation.
        copy_amps (bool):
            Whether to copy the t1, t2 amps.  Currently only the default True is
            allowed.

    Returns:
        delta_ccsd (float):
            Difference of perturbed and unperturbed CCSD ground-state energy,
                energy(T1 + T1[2], T2 + T2[2]) - energy(T1, T2)
        pt1 (:obj:`ndarray`):
            Perturbatively corrected T1 amplitudes.
        pt2 (:obj:`ndarray`):
            Perturbatively corrected T2 amplitudes.

    Reference:
        D. A. Matthews, J. F. Stanton "A new approach to approximate..."
            JCP 145, 124102 (2016), Equation 14
        Shavitt and Bartlett "Many-body Methods in Physics and Chemistry"
            2009, Equation 10.33
    """
    fock = np.asarray(eris.fock)
    nocc = cc.nocc
    nmo = cc.nmo
    nvir = nmo - nocc
    nkpts = cc.nkpts
    kconserv = cc.khelper.kconserv
    #assert(isinstance(eris, gccsd._PhysicistsERIs))

    fov = fock[:, :nocc, nocc:]
    foo = np.array([fock[ikpt, :nocc, :nocc].diagonal() for ikpt in range(nkpts)])
    fvv = np.array([fock[ikpt, nocc:, nocc:].diagonal() for ikpt in range(nkpts)])

    ccsd_energy = cc.energy(t1, t2, eris)

    if copy_amps:
        pt1 = t1.copy()
        pt2 = t2.copy()
    else:
        pt1 = t1
        pt2 = t2

    Wmbkj = None
    if build_ip_t3p2:
        Wmbkj = np.zeros((nkpts,nkpts,nkpts,nocc,nvir,nocc,nocc), dtype=np.complex)

    Wcbej = None
    if build_ea_t3p2:
        Wcbej = np.zeros((nkpts,nkpts,nkpts,nvir,nvir,nvir,nocc), dtype=np.complex)

    @profile
    def get_t3_fast_new():
        print 'creating temp arrays'
        if hasattr(eris, 'vvop'):
            eris_vvop = eris.vvop
            # vooo in chemist notation
            eris_vooo = eris.vooo
            eris_vooo_C = eris.vooo_C
        else:
            eris_vvop = np.zeros((nkpts,)*3 + (nvir,)*2 + (nocc, nmo), dtype=np.complex, order='C')
            # vooo in chemist notation
            eris_vooo = np.zeros((nkpts,)*3 + (nvir,) + (nocc,)*3, dtype=np.complex, order='C')
            eris_vooo_C = np.zeros((nkpts,)*3 + (nvir,) + (nocc,)*3, dtype=np.complex, order='C')
            t2T = np.zeros((nkpts,)*3 + (nvir,)*2 + (nocc,)*2, dtype=np.complex, order='C')
            for ki, kj, ka in product(range(nkpts), repeat=3):
                kb = kconserv[ki, ka, kj]
                eris_vvop[ki,kj,ka,:,:,:,nocc:] = eris.vovv[kb,ka,kj].conj().transpose(3,2,1,0)
                eris_vooo[ki,ka,kj] = eris.ooov[kb,kj,ka].conj().transpose(3,2,1,0)
                #eris_vooo_C[ki,kj,ka] = eris.ooov[kb,kj,ka].conj().transpose(3, 1, 2, 0)
                eris_vooo_C[ki,kj,kb] = eris.ooov[kb,kj,ka].conj().transpose(3, 1, 2, 0).transpose(0,1,3,2)
        t2T = np.zeros((nkpts,)*3 + (nvir,)*2 + (nocc,)*2, dtype=np.complex, order='C')
        for ki, kj, ka in product(range(nkpts), repeat=3):
            kb = kconserv[ki,ka,kj]
            t2T[ka,kb,kj] = t2[ki,kj,ka].transpose(2,3,1,0)
        t1T = np.asarray([t1[kpt].transpose(1,0) for kpt in range(nkpts)], order='C')
        fvo = np.asarray([fov[kpt].transpose(1,0) for kpt in range(nkpts)], order='C')
        fock = np.array([np.diag(x).real for x in eris.fock], dtype=np.float64)
        mo_energy = np.asarray(fock, order='C')
        print 'done...'

        from pyscf.cc import _ccsd
        tasks = []
        kpt_blksize, vir_blksize = 2, 10
        for a0, a1 in lib.prange(0, nvir, vir_blksize):
            for b0, b1 in lib.prange(0, nvir, vir_blksize):
                for c0, c1 in lib.prange(0, nvir, vir_blksize):
                    tasks.append((a0,a1,b0,b1,c0,c1))

        def read_vvop(idx):
            return eris_vvop[idx]
        def read_vooo(idx):
            return eris_vooo_C[idx]
        def get_t2Tvvop(idx):
            return t2T[idx]
        def get_t2Tvooo(idx):
            return t2T[idx]
        args = (read_vvop, read_vooo, get_t2Tvvop, get_t2Tvooo)

        def contract_t3Tv(kpt_indices, orb_indices, data):
            '''Calculate t3T(ransposed) array using C driver.'''
            ki, kj, kk, ka, kb, kc = kpt_indices
            a0, a1, b0, b1, c0, c1 = orb_indices
            slices = np.array([a0, a1, b0, b1, c0, c1], dtype=np.int32)

            mo_offset = np.array([ki*nmo, kj*nmo, kk*nmo,
                                  ka*nmo, kb*nmo, kc*nmo], dtype=np.int32)

            vvop_ab = np.asarray(data[0][0], dtype=np.complex, order='C')
            vvop_ac = np.asarray(data[0][1], dtype=np.complex, order='C')
            vvop_ba = np.asarray(data[0][2], dtype=np.complex, order='C')
            vvop_bc = np.asarray(data[0][3], dtype=np.complex, order='C')
            vvop_ca = np.asarray(data[0][4], dtype=np.complex, order='C')
            vvop_cb = np.asarray(data[0][5], dtype=np.complex, order='C')

            vooo_aj = np.asarray(data[1][0], dtype=np.complex, order='C')
            vooo_ak = np.asarray(data[1][1], dtype=np.complex, order='C')
            vooo_bi = np.asarray(data[1][2], dtype=np.complex, order='C')
            vooo_bk = np.asarray(data[1][3], dtype=np.complex, order='C')
            vooo_ci = np.asarray(data[1][4], dtype=np.complex, order='C')
            vooo_cj = np.asarray(data[1][5], dtype=np.complex, order='C')

            t2T_cj = np.asarray(data[2][0], dtype=np.complex, order='C')
            t2T_bk = np.asarray(data[2][1], dtype=np.complex, order='C')
            t2T_ci = np.asarray(data[2][2], dtype=np.complex, order='C')
            t2T_ak = np.asarray(data[2][3], dtype=np.complex, order='C')
            t2T_bi = np.asarray(data[2][4], dtype=np.complex, order='C')
            t2T_aj = np.asarray(data[2][5], dtype=np.complex, order='C')

            t2T_cb = np.asarray(data[3][0], dtype=np.complex, order='C')
            t2T_bc = np.asarray(data[3][1], dtype=np.complex, order='C')
            t2T_ca = np.asarray(data[3][2], dtype=np.complex, order='C')
            t2T_ac = np.asarray(data[3][3], dtype=np.complex, order='C')
            t2T_ba = np.asarray(data[3][4], dtype=np.complex, order='C')
            t2T_ab = np.asarray(data[3][5], dtype=np.complex, order='C')

            data = [vvop_ab, vvop_ac, vvop_ba, vvop_bc, vvop_ca, vvop_cb,
                    vooo_aj, vooo_ak, vooo_bi, vooo_bk, vooo_ci, vooo_cj,
                    t2T_cj, t2T_cb, t2T_bk, t2T_bc, t2T_ci, t2T_ca, t2T_ak,
                    t2T_ac, t2T_bi, t2T_ba, t2T_aj, t2T_ab]
            data_ptrs = [x.ctypes.data_as(ctypes.c_void_p) for x in data]
            data_ptrs = (ctypes.c_void_p*24)(*data_ptrs)

            a0, a1, b0, b1, c0, c1 = task
            t3T = np.empty((a1-a0,b1-b0,c1-c0) + (nocc,)*3, dtype=np.complex, order='C')

            drv = _ccsd.libcc.zcontract_t3T
            drv(t3T.ctypes.data_as(ctypes.c_void_p),
                mo_energy.ctypes.data_as(ctypes.c_void_p),
                t1T.ctypes.data_as(ctypes.c_void_p),
                fvo.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nocc), ctypes.c_int(nvir),
                mo_offset.ctypes.data_as(ctypes.c_void_p),
                slices.ctypes.data_as(ctypes.c_void_p),
                data_ptrs)
            return t3T

        def add_and_permute(kpt_indices, orb_indices, data):
            '''Performs permutation and addition of t3 temporary arrays.'''
            ki, kj, kk, ka, kb, kc = kpt_indices
            a0, a1, b0, b1, c0, c1 = orb_indices
            slices = np.array([a0, a1, b0, b1, c0, c1], dtype=np.int32)

            mo_offset = np.array([ki*nmo, kj*nmo, kk*nmo,
                                  ka*nmo, kb*nmo, kc*nmo], dtype=np.int32)

            tmp_t3Tv_ijk = np.asarray(data[0], dtype=np.complex, order='C')
            tmp_t3Tv_jik = np.asarray(data[1], dtype=np.complex, order='C')
            tmp_t3Tv_kji = np.asarray(data[2], dtype=np.complex, order='C')

            drv = _ccsd.libcc.MPICCadd_and_permute_t3T
            drv(ctypes.c_int(nocc), ctypes.c_int(nvir),
                tmp_t3Tv_ijk.ctypes.data_as(ctypes.c_void_p),
                tmp_t3Tv_jik.ctypes.data_as(ctypes.c_void_p),
                tmp_t3Tv_kji.ctypes.data_as(ctypes.c_void_p),
                mo_offset.ctypes.data_as(ctypes.c_void_p),
                slices.ctypes.data_as(ctypes.c_void_p))
            # drv has overwritten tmp_t3Tv_ijk
            Ptmp_t3Tv = tmp_t3Tv_ijk  #2.*t3Tv_ijk - t3Tv_jik.transpose(0,1,2,4,3,5)
                                      #            - t3Tv_kji.transpose(0,1,2,5,4,3)
            return Ptmp_t3Tv

        #full_t3v_ijk = get_full_t3p2(cc, t1, t2, eris)

        fetcher = DataHandler()
        for ka, kb in product(range(nkpts), repeat=2):
          eaa = foo[ka][:, None] - fvv[ka][None, :]
          for ki, kj in product(range(nkpts), repeat=2):
            kc_list = kpts_helper.get_kconserv3(cc._scf.cell, cc.kpts,
                                                [ki, kj, range(nkpts), ka, kb])
            for kk in range(nkpts):
                oovv_jXX = eris.oovv[kj, :, :]
                for task_id, task in enumerate(tasks):
                    a0, a1, b0, b1, c0, c1 = task

                    print ki, kj, kk, ka, kb, a0, a1, b0, b1, c0, c1
                    #tmp_t3Tv_ijk = np.empty(((kk1-kk0),(a1-a0),(b1-b0),(c1-c0),nocc,nocc,nocc), dtype=np.complex)
                    #tmp_t3Tv_jik = np.empty(((kk1-kk0),(a1-a0),(b1-b0),(c1-c0),nocc,nocc,nocc), dtype=np.complex)
                    #tmp_t3Tv_kji = np.empty(((kk1-kk0),(a1-a0),(b1-b0),(c1-c0),nocc,nocc,nocc), dtype=np.complex)
                    #for ikk, kk in enumerate(range(kk0, kk1)):
                    #    kc = kc_list[ikk]
                    #    kpt_indices = [[ki,kj,kk,ka,kb,kc],
                    #                   [kj,ki,kk,ka,kb,kc],
                    #                   [kk,kj,ki,ka,kb,kc]]
                    #    fetcher.request_data(kpt_indices[0], task, kconserv, *args)
                    #    fetcher.request_data(kpt_indices[1], task, kconserv, *args)
                    #    fetcher.request_data(kpt_indices[2], task, kconserv, *args)

                    #    data1 = fetcher.get_data(kpt_indices[0], task, kconserv, *args)
                    #    data2 = fetcher.get_data(kpt_indices[1], task, kconserv, *args)
                    #    data3 = fetcher.get_data(kpt_indices[2], task, kconserv, *args)

                    #    tmp_t3Tv_ijk[ikk] = contract_t3Tv(kpt_indices[0], task, data1)
                    #    tmp_t3Tv_jik[ikk] = contract_t3Tv(kpt_indices[1], task, data2)
                    #    tmp_t3Tv_kji[ikk] = contract_t3Tv(kpt_indices[2], task, data3)

                    #Ptmp_t3Tv = (2.*tmp_t3Tv_ijk - tmp_t3Tv_jik.transpose(0,1,2,3,5,4,6)
                    #                             - tmp_t3Tv_kji.transpose(0,1,2,3,6,5,4))

                    #if ki == ka:
                    #    eris_Soovv = (2.*oovv_jXX[kk0:kk1,kb,:,:,b0:b1,c0:c1] -
                    #                     oovv_jXX[kk0:kk1,kc,:,:,c0:c1,b0:b1].transpose(0,1,2,4,3))
                    #    tmp =  0.5 * lib.einsum('xjkbc,xabcijk->ia', eris_Soovv, Ptmp_t3Tv)
                    #    tmp /= eaa[:, a0:a1]

                    #    pt1[ka,:,a0:a1] += tmp

                    tmp_t3Tv_ijk = np.empty(((a1-a0),(b1-b0),(c1-c0),nocc,nocc,nocc), dtype=np.complex)
                    tmp_t3Tv_jik = np.empty(((a1-a0),(b1-b0),(c1-c0),nocc,nocc,nocc), dtype=np.complex)
                    tmp_t3Tv_kji = np.empty(((a1-a0),(b1-b0),(c1-c0),nocc,nocc,nocc), dtype=np.complex)

                    kc = kc_list[kk]
                    kpt_indices = [[ki,kj,kk,ka,kb,kc],
                                   [kj,ki,kk,ka,kb,kc],
                                   [kk,kj,ki,ka,kb,kc]]

                    fetcher.request_data(kpt_indices[0], task, kconserv, *args)
                    fetcher.request_data(kpt_indices[1], task, kconserv, *args)
                    fetcher.request_data(kpt_indices[2], task, kconserv, *args)

                    data1 = fetcher.get_data(kpt_indices[0], task, kconserv, *args)
                    data2 = fetcher.get_data(kpt_indices[1], task, kconserv, *args)
                    data3 = fetcher.get_data(kpt_indices[2], task, kconserv, *args)

                    tmp_t3Tv_ijk = contract_t3Tv(kpt_indices[0],task,data1)
                    tmp_t3Tv_jik = contract_t3Tv(kpt_indices[1],task,data2)
                    tmp_t3Tv_kji = contract_t3Tv(kpt_indices[2],task,data3)

                    #new_tmp_t3Tv_ijk = full_t3v_ijk[ki,kj,kk,ka,kb].transpose(3,4,5,0,1,2)
                    #new_tmp_t3Tv_jik = full_t3v_ijk[kj,ki,kk,ka,kb].transpose(3,4,5,0,1,2)
                    #new_tmp_t3Tv_kji = full_t3v_ijk[kk,kj,ki,ka,kb].transpose(3,4,5,0,1,2)
                    #print 'diff', np.linalg.norm(tmp_t3Tv_ijk - new_tmp_t3Tv_ijk)
                    #tmp_t3Tv_ijk = new_tmp_t3Tv_ijk
                    #tmp_t3Tv_jik = new_tmp_t3Tv_jik
                    #tmp_t3Tv_kji = new_tmp_t3Tv_kji

                    Ptmp_t3Tv = add_and_permute(kpt_indices[0], task, (tmp_t3Tv_ijk,tmp_t3Tv_jik,tmp_t3Tv_kji))

                    # Performing contribution to pt1
                    eris_Soovv = None
                    if ki == ka and kc == kconserv[kj, kb, kk]:
                        eris_Soovv = (2.*oovv_jXX[kk,kb,:,:,b0:b1,c0:c1] -
                                         oovv_jXX[kk,kc,:,:,c0:c1,b0:b1].transpose(0,1,3,2))

                    add_contribution_pt1(cc, kpt_indices[0], task, kconserv,
                                         (Ptmp_t3Tv,eris_Soovv,eaa[:,a0:a1]),
                                         out=pt1[ka,:,a0:a1])

                    # Performing contribution to pt2
                    if ki == ka and kc == kconserv[kj, kb, kk]:
                        ejkbc = (foo[kj][:,None,None,None] + foo[kk][None,:,None,None] -
                                fvv[kb,b0:b1][None,None,:,None] - fvv[kc,c0:c1][None,None,None,:])
                        tmp = einsum('abcijk,ia->jkbc', Ptmp_t3Tv, 0.5*fov[ki,:,a0:a1]) / ejkbc
                        pt2[kj,kk,kb,:,:,b0:b1,c0:c1] += tmp
                        pt2[kk,kj,kc,:,:,c0:c1,b0:b1] += tmp.transpose(1,0,3,2)

                    kd = kconserv[ka,ki,kb]
                    #if kk == kconserv[kd, kj, kc]:
                    eris_vovv = eris.vovv[kd,ki,kb,:,:,b0:b1,a0:a1]
                    ejkdc = (foo[kj][:,None,None,None] + foo[kk][None,:,None,None] -
                             fvv[kd][None,None,:,None] - fvv[kc,c0:c1][None,None,None,:])
                    tmp = einsum('abcijk,diba->jkdc', Ptmp_t3Tv, eris_vovv) / ejkdc
                    pt2[kj,kk,kd,:,:,:,c0:c1] += tmp
                    pt2[kk,kj,kc,:,:,c0:c1,:] += tmp.transpose(1,0,3,2)

                    km = kconserv[kc, kk, kb]
                    eris_ooov = eris.ooov[kj,ki,km,:,:,:,a0:a1]
                    #if ka == kconserv[ki, km, kj]:
                    # TODO: do transpose after in one-shot (rather than at each step?)
                    emkbc = (foo[km][:,None,None,None] + foo[kk][None,:,None,None] -
                            fvv[kb,b0:b1][None,None,:,None] - fvv[kc,c0:c1][None,None,None,:])
                    tmp = einsum('abcijk,jima->mkbc', Ptmp_t3Tv, eris_ooov) / emkbc
                    pt2[km,kk,kb,:,:,b0:b1,c0:c1] -= tmp
                    pt2[kk,km,kc,:,:,c0:c1,b0:b1] -= tmp.transpose(1,0,3,2)

                    # Calculating Wovoo array
                    if build_ip_t3p2:
                        km = kconserv[ka,ki,kc]
                        eris_oovv = eris.oovv[km,ki,kc]
                        tmp = einsum('abcijk,mica->mbkj', Ptmp_t3Tv, eris_oovv)
                        Wmbkj[km,kb,kk,:,b0:b1,:,:] += tmp

                    # Calculating Wvvvo array
                    if build_ea_t3p2:
                        ke = kconserv[ki,ka,kk]
                        eris_oovv = eris.oovv[ki,kk,ka]
                        tmp = einsum('abcijk,ikae->cbej', Ptmp_t3Tv, eris_oovv)
                        Wcbej[kc,kb,ke,:,c0:c1,b0:b1,:] -= tmp

                    fetcher.clean()

    get_t3_fast_new()

    delta_ccsd_energy = cc.energy(pt1, pt2, eris) - ccsd_energy
    logger.info(cc, 'CCSD energy T3[2] correction : %16.12e', delta_ccsd_energy)

    return delta_ccsd_energy, pt1, pt2, Wmbkj, Wcbej

if __name__ == '__main__':
    cell = pbcgto.Cell()
    cell.atom = '''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = {'C': [[0, (0.8, 1.0)],
                        [0, (0.5, 1.0)],
                        [1, (1.0, 1.0)]]}
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.precision = 1e-14
    cell.verbose = 5
    cell.build()

    true_system = False
    if true_system:
        gamma = False
        if gamma:
            mf = pbcscf.RHF(cell)
            mf.conv_tol = 1e-10
            mf.kernel()

            mycc = pbcc.RCCSD(mf)
            mycc.conv_tol = 1e-10
            eris = mycc.ao2mo()
            mycc.kernel(eris=eris)

            class eom(object):
                def __init__(self):
                    self._cc = mycc
                    self.max_memory = 5000
                    self.verbose = 7
                    self.stdout = sys.stdout

            kmf = pbcscf.KRHF(cell, kpts=[0.,0.,0.])
            kmf.conv_tol = 1e-10
            kmf.kernel()

            kmo_coeff = mf.mo_coeff[None,:,:].astype(np.complex)
            mykcc = pbcc.KRCCSD(kmf, mo_coeff=kmo_coeff)
            mykcc.conv_tol = 1e-10
            keris = mykcc.ao2mo(mo_coeff=kmo_coeff)
            mykcc.kernel(eris=keris)
            mykcc.t1 = mycc.t1[None,:].astype(np.complex).copy()
            mykcc.t2 = mycc.t2[None,None,None,:,:,:,:].astype(np.complex).copy()

            myeom = eom()
            delta_ccsd_energy, pt1, pt2, Wmbkj, Wcbej = \
                eom_rccsd.get_t3p2_amplitude_contribution_slow(myeom, mycc.t1, mycc.t2, eris=eris, build_t1_t2=True)
            real_pt1 = pt1.copy()
            real_pt2 = pt2.copy()
            cdelta_ccsd_energy, cpt1, cpt2 = get_t3p2_amplitude_contribution(mykcc, mykcc.t1, mykcc.t2, eris=keris, copy_amps=True)
            print np.linalg.norm(pt1-cpt1)
            print np.linalg.norm(pt2-cpt2)
            print np.linalg.norm(cdelta_ccsd_energy - delta_ccsd_energy)
        else:
            nk = [2,1,1]
            scell = super_cell(cell, nk)
            mf = pbcscf.RHF(scell)
            mf.conv_tol = 1e-12
            mf.kernel()

            mycc = pbcc.RCCSD(mf)
            mycc.conv_tol = 1e-12
            mycc.conv_tol_normt = 1e-12
            eris = mycc.ao2mo()
            mycc.kernel(eris=eris)

            class eom(object):
                def __init__(self):
                    self._cc = mycc
                    self.max_memory = 5000
                    self.verbose = 7
                    self.stdout = sys.stdout

            kmf = pbcscf.KRHF(cell, kpts=cell.make_kpts(nk, with_gamma_point=True))
            kmf.conv_tol = 1e-12
            kmf.kernel()

            mykcc = pbcc.KRCCSD(kmf)
            mykcc.conv_tol = 1e-12
            keris = mykcc.ao2mo()
            mykcc.kernel(eris=keris)
            mykcc.t1 = mykcc.t1.astype(np.complex)
            mykcc.t2 = mykcc.t2.astype(np.complex)

            myeom = eom()
            delta_ccsd_energy, pt1, pt2, Wmbkj, Wcbej = \
                eom_rccsd.get_t3p2_amplitude_contribution_slow(myeom, mycc.t1, mycc.t2, eris=eris, build_t1_t2=True)
            real_pt1 = pt1.copy()
            real_pt2 = pt2.copy()
            cdelta_ccsd_energy, cpt1, cpt2, _, _ = get_t3p2_amplitude_contribution(mykcc, mykcc.t1, mykcc.t2, eris=keris, copy_amps=True)
            print np.linalg.norm(cdelta_ccsd_energy - delta_ccsd_energy/np.prod(nk))
    else:
        def crand(shape):
            return (np.random.rand(np.prod(shape)).reshape(shape) - 0.5 - 0.5*1j -
                    np.random.rand(np.prod(shape)).reshape(shape)*1j)

        nmo = 40
        nocc = 15
        nk = [2,2,1]
        nvir = nmo - nocc
        def make_rand_kmf():
            # Not perfect way to generate a random mf.
            # CSC = 1 is not satisfied and the fock matrix is neither
            # diagonal nor sorted.
            np.random.seed(2)
            kmf = pbcscf.KRHF(cell, kpts=cell.make_kpts(nk))
            kmf.exxdiv = None

            kmf.mo_occ = np.zeros((np.prod(nk), nmo))
            kmf.mo_occ[:, :nocc] = 2
            kmf.mo_energy = np.arange(nmo) + np.random.random((np.prod(nk), nmo)) * .3
            kmf.mo_energy[kmf.mo_occ == 0] += 2
            kmf.mo_coeff = (np.random.random((np.prod(nk), nmo, nmo)) +
                            np.random.random((np.prod(nk), nmo, nmo)) * 1j - .5 - .5j)
            return kmf

        print 'creating rand mf...'
        rand_kmf = make_rand_kmf()

        def rand_t1_t2(kmf, mycc):
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

            def kconserve_pmatrix(nkpts, kconserv):
                Ps = np.zeros((nkpts, nkpts, nkpts, nkpts))
                for ki in range(nkpts):
                    for kj in range(nkpts):
                        for ka in range(nkpts):
                            # Chemist's notation for momentum conserving t2(ki,kj,ka,kb)
                            kb = kconserv[ki, ka, kj]
                            Ps[ki, kj, ka, kb] = 1
                return Ps

            Ps = kconserve_pmatrix(nkpts, kconserv)
            t2 = t2 + np.einsum('xyzijab,xyzw->yxwjiba', t2, Ps)
            return t1, t2

        rand_cc = pbcc.KRCCSD(rand_kmf)
        rand_cc.verbose = 7
        print 'creating rand t1/t2...'
        t1, t2 = rand_t1_t2(rand_kmf, rand_cc)
        rand_cc.t1, rand_cc.t2 = t1, t2
        nkpts = rand_cc.nkpts

        kconserv = kpts_helper.get_kconserv(rand_kmf.cell, rand_kmf.kpts)
        class eris(object):
            def __init__(self):
                import os
                filename = 'myeris.hdf5'
                if not os.path.isfile(filename):
                    f = h5py.File(filename, 'w')

                    eris_vovv = crand((nkpts,nkpts,nkpts,nvir,nocc,nvir,nvir))
                    eris_ooov = crand((nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir))
                    self.oooo = f.create_dataset('oooo', data=crand((nkpts,nkpts,nkpts,nocc,nocc,nocc,nocc)))
                    self.ooov = f.create_dataset('ooov', data=eris_ooov)
                    self.oovv = f.create_dataset('oovv', data=crand((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir)))
                    self.ovov = f.create_dataset('ovov', data=crand((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir)))
                    self.voov = f.create_dataset('voov', data=crand((nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir)))
                    self.vovv = f.create_dataset('vovv', data=eris_vovv)

                    eris_vvop = np.zeros((nkpts,)*3 + (nvir,)*2 + (nocc, nmo), dtype=np.complex, order='C')
                    # vooo in chemist notation
                    eris_vooo = np.zeros((nkpts,)*3 + (nvir,) + (nocc,)*3, dtype=np.complex, order='C')
                    eris_vooo_C = np.zeros((nkpts,)*3 + (nvir,) + (nocc,)*3, dtype=np.complex, order='C')
                    for ki, kj, ka in product(range(nkpts), repeat=3):
                        kb = kconserv[ki, ka, kj]
                        eris_vvop[ki,kj,ka,:,:,:,nocc:] = eris_vovv[kb,ka,kj].conj().transpose(3,2,1,0)
                        eris_vooo[ki,ka,kj] = eris_ooov[kb,kj,ka].conj().transpose(3,2,1,0)
                        eris_vooo_C[ki,kj,ka] = eris_vooo[ki,ka,kj].transpose(0,2,1,3)
                    self.vooo = f.create_dataset('vooo', data=eris_vooo)
                    self.vooo_C = f.create_dataset('voooC', data=eris_vooo_C)
                    self.vvop = f.create_dataset('vvop', data=eris_vvop)

                    f.close()

                f = h5py.File(filename, 'r')
                self.oooo = f['oooo']
                self.ooov = f['ooov']
                self.oovv = f['oovv']
                self.ovov = f['ovov']
                self.voov = f['voov']
                self.vovv = f['vovv']
                self.vvop = f['vvop']
                self.vooo = f['vooo']
                self.vooo_C = f['voooC']

        print 'creating eris...'
        eris = eris()
        eris.fock = np.array([np.diag(x) for x in rand_kmf.mo_energy])
        print 'getting contribution...'
        get_t3p2_amplitude_contribution(rand_cc, t1, t2, eris=eris, copy_amps=True)
