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
from mpi4py import MPI
from pyscf.pbc.mpitools import mpi
from pyscf.pbc.mpitools.mpi_helper import safeAllreduceInPlace

einsum = lib.einsum

rank = MPI.COMM_WORLD.Get_rank()
comm = MPI.COMM_WORLD

def check_write_complete(filename, **kwargs):
    '''Check for `completed` attr in file.'''
    import os
    mode = kwargs.get('mode', 'r')
    if not os.path.isfile(filename):
        return False
    try:
        f = h5py.File(filename, mode=mode, **kwargs)
    except IOError:
        return False
    return f.attrs.get('completed', False)

def check_read_success(filename, **kwargs):
    write_complete = check_write_complete(filename, **kwargs)
    return write_complete

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

def get_data_slices2(kpt_indices, orb_indices, kconserv):
    kpt_indices = zip_kpoints(kpt_indices)
    if isinstance(kpt_indices[0], (int, np.int)):  # Ensure we are working
        kpt_indices = [kpt_indices]                   # with a list of lists

    a0, a1, b0, b1, c0, c1 = orb_indices
    length = len(kpt_indices)

    vvop_indices = [0] * length
    ooov_indices = [0] * length
    oovv_indices1 = [0] * length
    oovv_indices2 = [0] * length

    for ikpt, kpt in enumerate(kpt_indices):
        ki, kj, kk, ka, kb, kc = kpt
        sl00 = slice(None, None)

        kd = kconserv[ka,ki,kb]
        vvop_indices[ikpt] = [ka, kb, ki, slice(a0,a1), slice(b0,b1), sl00, sl00]

        km = kconserv[kc, kk, kb]
        ooov_indices[ikpt] = [kj, ki, km, sl00, sl00, sl00, slice(a0,a1)]

        km = kconserv[ka,ki,kc]
        oovv_indices1[ikpt] = [km, ki, kc, sl00, sl00, slice(c0,c1), slice(a0,a1)]

        ke = kconserv[ki,ka,kk]
        oovv_indices2[ikpt] = [ki, kk, ka, sl00, sl00, slice(a0,a1), sl00]

    return vvop_indices, ooov_indices, oovv_indices1, oovv_indices2

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
        '''Update list of jobs requested, unique jobs, and initialize dict
        for results of these jobs.
        '''
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

    def _submit(self, job_name):
        unq_jobs = self.unique_job_args[job_name]
        jobs = self.job[job_name]
        #logger.debug3(self, 'job %s reduced keys %d -> %d (%f percent)',
        #              job_name, len(jobs), len(unq_jobs),
        #              (1. - (1.*len(unq_jobs))/len(jobs))*100)

        # Submit only unique arguments that we don't have results for
        dict_ = self.job_results[job_name]
        for k in unq_jobs ^ set(dict_.keys()):
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

    def request_data2(self, kpt_indices, orb_indices, kconserv, *args):
        idx_args = get_data_slices2(kpt_indices, orb_indices, kconserv)
        vvop_indices, ooov_indices, oovv_indices1, oovv_indices2 = idx_args
        for task in range(len(vvop_indices)):
            self.add_job('vvop', args[0], vvop_indices[task])
            self.add_job('ooov', args[1], ooov_indices[task])
            self.add_job('oovv', args[2], oovv_indices1[task])
            self.add_job('oovv', args[3], oovv_indices2[task])

        self.submit('vvop')
        self.submit('ooov')
        self.submit('oovv')
        return

    def get_data(self, kpt_indices, orb_indices, kconserv, *args):
        idx_args = get_data_slices(kpt_indices, orb_indices, kconserv)
        vvop_indices, vooo_indices, t2T_vvop_indices, t2T_vooo_indices = idx_args

        vvop = self.results('vvop', args[0], vvop_indices)
        vooo = self.results('vooo', args[1], vooo_indices)
        t2Tvvop = self.results('t2Tvvop', args[2], t2T_vvop_indices)
        t2Tvooo = self.results('t2Tvooo', args[3], t2T_vooo_indices)
        return vvop, vooo, t2Tvvop, t2Tvooo

    def get_data2(self, kpt_indices, orb_indices, kconserv, *args):
        idx_args = get_data_slices2(kpt_indices, orb_indices, kconserv)
        vvop_indices, ooov_indices, oovv_indices1, oovv_indices2 = idx_args

        vvop = self.results('vvop', args[0], vvop_indices)
        ooov = self.results('ooov', args[1], ooov_indices)
        oovv1 = self.results('oovv', args[2], oovv_indices1)
        oovv2 = self.results('oovv', args[3], oovv_indices2)
        return vvop[0], ooov[0], oovv1[0], oovv2[0]

    def clean(self):
        jobs = ['vvop', 'vooo', 't2Tvvop', 't2Tvooo']
        for j in jobs:
            if len(self.job[j]) > self.NMAX:
                #logger.debug3(self, 'Clearing cache for job %s', j)
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

def transpose_t2(t2, nkpts, nocc, nvir, kconserv, out=None):
    '''Creates t2.transpose(2,3,1,0).'''
    if out is None:
        out = np.empty((nkpts,nkpts,nkpts,nvir,nvir,nocc,nocc), dtype=t2.dtype)

    if len(t2.shape) == 7 and t2.shape[:2] == (nkpts, nkpts):
        for ki, kj, ka in product(range(nkpts), repeat=3):
            kb = kconserv[ki,ka,kj]
            out[ka,kb,kj] = t2[ki,kj,ka].transpose(2,3,1,0)
    elif len(t2.shape) == 6 and t2.shape[:2] == (nkpts*(nkpts+1)//2, nkpts):
        #if isinstance(out, h5py.Dataset):  # Can't do multiple indexing vectors
        for ki, kj, ka in product(range(nkpts), repeat=3):
            kb = kconserv[ki,ka,kj]
            # t2[ki,kj,ka] = t2[tril_index(ki,kj),ka]  ki<kj
            # t2[kj,ki,kb] = t2[ki,kj,ka].transpose(1,0,3,2)  ki<kj
            #              = t2[tril_index(ki,kj),ka].transpose(1,0,3,2)
            if ki <= kj:
                tril_idx = (kj*(kj+1))//2 + ki
                out[ka,kb,kj] = t2[tril_idx,ka].transpose(2,3,1,0).copy()
                out[kb,ka,ki] = out[ka,kb,kj].transpose(1,0,3,2)
        #else:
        #    for ka in range(nkpts):
        #        idx0, idx1 = np.tril_indices(nkpts)
        #        kb = kconserv[idx0,ka,idx1]
        #        out[ka,kb,idx0] = t2[:,ka].transpose(0,3,4,2,1)
        #        out[kb,ka,idx1] = t2[:,ka].transpose(0,4,3,1,2)
    else:
        raise ValueError('No known conversion for t2 shape %s' % t2.shape)
    return out

def create_eris_vvop(vovv, nkpts, nocc, nvir, kconserv, out=None):
    '''Creates vvop from vovv array.'''
    nmo = nocc + nvir
    assert(vovv.shape == (nkpts,nkpts,nkpts,nvir,nocc,nvir,nvir))
    if out is None:
        out = np.empty((nkpts,nkpts,nkpts,nvir,nvir,nocc,nmo), dtype=vovv.dtype)
    else:
        assert(out.shape == (nkpts,nkpts,nkpts,nvir,nvir,nocc,nmo))

    for ki, kj, ka in product(range(nkpts), repeat=3):
        kb = kconserv[ki,ka,kj]
        out[ki,kj,ka,:,:,:,nocc:] = vovv[kb,ka,kj].conj().transpose(3,2,1,0)
    return out

def create_eris_vooo(ooov, nkpts, nocc, nvir, kconserv, out=None):
    '''Creates vooo from ooov array.

    This is not exactly chemist's notation, but close.  Here vooo is
    created from ooov, and then the last two indices of vooo are swapped.
    '''
    assert(ooov.shape == (nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir))
    if out is None:
        out = np.empty((nkpts,nkpts,nkpts,nvir,nocc,nocc,nocc), dtype=ooov.dtype)

    for ki, kj, ka in product(range(nkpts), repeat=3):
        kb = kconserv[ki,ka,kj]
        out[ki,kj,kb] = ooov[kb,kj,ka].conj().transpose(3,1,0,2)
    return out

def _add_pt2(pt2, nkpts, kconserv, kpt_indices, orb_indices, val):
    '''Adds term P(ia|jb)[tmp] to pt2.

    P(ia|jb)(tmp[i,j,a,b]) = tmp[i,j,a,b] + tmp[j,i,b,a]

    or equivalently for each i,j,a,b, pt2 is defined as

    pt2[i,j,a,b] += tmp[i,j,a,b]
    pt2[j,i,b,a] += tmp[i,j,a,b].transpose(1,0,3,2)

    If pt2 is lower-triangular, only adds the RHS term that contributes
    to the lower-triangular pt2.
    '''
    ki, kj, ka = kpt_indices
    kb = kconserv[ki,ka,kj]
    idxi, idxj, idxa, idxb = [slice(None, None)
                              if x is None else slice(x[0],x[1])
                              for x in orb_indices]
    if len(pt2.shape) == 7 and pt2.shape[:2] == (nkpts, nkpts):
        pt2[ki,kj,ka,idxi,idxj,idxa,idxb] += val
        pt2[kj,ki,kb,idxi,idxj,idxb,idxa] += val.transpose(1,0,3,2)
    elif len(pt2.shape) == 6 and pt2.shape[:2] == (nkpts*(nkpts+1)//2, nkpts):
        if ki <= kj:  # Add tmp[i,j,a,b] to pt2[i,j,a,b]
            idx = (kj*(kj+1))//2 + ki
            pt2[idx,ka,idxi,idxj,idxa,idxb] += val
            if ki == kj:
                pt2[idx,kb,idxj,idxi,idxb,idxa] += val.transpose(1,0,3,2)
        else:  # pt2[i,a,j,b] += tmp[j,i,a,b].transpose(1,0,3,2)
            idx = (ki*(ki+1))//2 + kj
            pt2[idx,kb,idxj,idxi,idxb,idxa] += val.transpose(1,0,3,2)
    else:
        raise ValueError('No known conversion for t2 shape %s' % t2.shape)

def get_t3p2_amplitude_contribution(cc, t1, t2, eris=None, copy_amps=True,
                                    build_t1_t2=True, build_ip_t3p2=False,
                                    build_ea_t3p2=False, Wmbkj_out=None,
                                    Wcbej_out=None):
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

    Notes:
        If specifying `Wmbkj_out` or `Wcbej_out`, these passing in arrays should
        not be empty; their values should be initialized.

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

    if rank != 0:  # These arrays will be all-reduced at the end
        pt1 *= 0.0
        pt2 *= 0.0

    #lower_pt2 = np.empty((nkpts*(nkpts+1)//2,nkpts,nocc,nocc,nvir,nvir), dtype=t2.dtype)
    #for ki,kj,ka in product(range(nkpts), repeat=3):
    #    if ki <= kj:
    #        lower_pt2[kj*(kj+1)//2+ki,ka] = pt2[ki,kj,ka]
    #pt2 = lower_pt2.copy()
    #ccsd_energy = cc.energy(pt1, pt2, eris)
    #print 'initial energy', ccsd_energy, pt2.shape

    Wmbkj_out = None
    if build_ip_t3p2 and (Wmbkj_out is None):
        Wmbkj_out = np.zeros((nkpts,nkpts,nkpts,nocc,nvir,nocc,nocc), dtype=np.complex)

    Wcbej_out = None
    if build_ea_t3p2 and (Wcbej_out is None):
        Wcbej_out = np.zeros((nkpts,nkpts,nkpts,nvir,nvir,nvir,nocc), dtype=np.complex)

    def get_t3_fast_new():
        print 'creating temp arrays'
        feri_tmp = None
        h5py_kwargs = {'driver':'mpio', 'comm':comm}
        feri_tmp_filename = 'tmp_t3_eris.h5'
        if not check_read_success(feri_tmp_filename):
            print "read failed"
            feri_tmp = h5py.File(feri_tmp_filename, 'w', **h5py_kwargs)
            dtype = np.complex
            t2T_out = feri_tmp.create_dataset('t2T', (nkpts,nkpts,nkpts,nvir,nvir,nocc,nocc), dtype=dtype)
            eris_vvop_out = feri_tmp.create_dataset('vvop', (nkpts,nkpts,nkpts,nvir,nvir,nocc,nmo), dtype=dtype)
            eris_vooo_C_out = feri_tmp.create_dataset('vooo_C', (nkpts,nkpts,nkpts,nvir,nocc,nocc,nocc), dtype=dtype)

            if rank == 0:
                transpose_t2(t2, nkpts, nocc, nvir, kconserv, out=t2T_out)
                create_eris_vvop(eris.vovv, nkpts, nocc, nvir, kconserv, out=eris_vvop_out)
                create_eris_vooo(eris.ooov, nkpts, nocc, nvir, kconserv, out=eris_vooo_C_out)

            feri_tmp.attrs['completed'] = True
            feri_tmp.close()

        feri_tmp = h5py.File(feri_tmp_filename, 'r', **h5py_kwargs)
        t2T = feri_tmp['t2T']
        eris_vvop = feri_tmp['vvop']
        eris_vooo_C = feri_tmp['vooo_C']

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
        def read_oovv(idx):
            return eris.oovv[idx]
        def read_ooov(idx):
            return eris.ooov[idx]
        def read_vovv(idx):
            return eris.vovv[idx]
        args = (read_vvop, read_vooo, get_t2Tvvop, get_t2Tvooo)
        args2 = (read_vvop, read_ooov, read_oovv, read_oovv)

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

        kblocks = []
        for ka, kb, ki, kj in product(range(nkpts), repeat=4):
            kblocks.append((ka,kb,ki,kj))

        #full_t3v_ijk = get_full_t3p2(cc, t1, t2, eris)  # Useful for checking

        comm.Barrier()
        fetcher = DataHandler()
        cput1 = (time.clock(), time.time())
        for ka, kb, ki, kj in mpi.work_share_partition(kblocks, loadmin=2):
            eaa = foo[ka][:, None] - fvv[ka][None, :]
            kc_list = kpts_helper.get_kconserv3(cc._scf.cell, cc.kpts,
                                                [ki, kj, range(nkpts), ka, kb])
            # TODO: one might block over [ki,kj,kk] as well for improved performance
            for kk in range(nkpts):
                oovv_jXX = eris.oovv[kj, :, :]
                cput0 = (time.clock(), time.time())

                for task_id, task in enumerate(tasks):
                    a0, a1, b0, b1, c0, c1 = task

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

                    fetcher.request_data2(kpt_indices[0], task, kconserv, *args2)

                    data1 = fetcher.get_data(kpt_indices[0], task, kconserv, *args)
                    data2 = fetcher.get_data(kpt_indices[1], task, kconserv, *args)
                    data3 = fetcher.get_data(kpt_indices[2], task, kconserv, *args)

                    tmp_t3Tv_ijk = contract_t3Tv(kpt_indices[0],task,data1)
                    tmp_t3Tv_jik = contract_t3Tv(kpt_indices[1],task,data2)
                    tmp_t3Tv_kji = contract_t3Tv(kpt_indices[2],task,data3)
                    Ptmp_t3Tv = add_and_permute(kpt_indices[0], task,
                                    (tmp_t3Tv_ijk,tmp_t3Tv_jik,tmp_t3Tv_kji))

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
                        _add_pt2(pt2, nkpts, kconserv, [kj,kk,kb], [None,None,(b0,b1),(c0,c1)], tmp)

                    data4 = fetcher.get_data2(kpt_indices[0], task, kconserv, *args2)
                    tmp_vvop, tmp_ooov, tmp_oovv1, tmp_oovv2 = data4

                    kd = kconserv[ka,ki,kb]
                    #if kk == kconserv[kd, kj, kc]:
                    tmp_vvov = tmp_vvop[:,:,:,nocc:]
                    ejkdc = (foo[kj][:,None,None,None] + foo[kk][None,:,None,None] -
                             fvv[kd][None,None,:,None] - fvv[kc,c0:c1][None,None,None,:])
                    tmp = einsum('abcijk,abid->jkdc', Ptmp_t3Tv, tmp_vvov.conj()) / ejkdc
                    _add_pt2(pt2, nkpts, kconserv, [kj,kk,kd], [None,None,None,(c0,c1)], tmp)

                    km = kconserv[kc, kk, kb]
                    #eris_ooov = eris.ooov[kj,ki,km,:,:,:,a0:a1]
                    #if ka == kconserv[ki, km, kj]:
                    # TODO: do transpose after in one-shot (rather than at each step?)
                    emkbc = (foo[km][:,None,None,None] + foo[kk][None,:,None,None] -
                            fvv[kb,b0:b1][None,None,:,None] - fvv[kc,c0:c1][None,None,None,:])
                    tmp = einsum('abcijk,jima->mkbc', Ptmp_t3Tv, tmp_ooov) / emkbc
                    _add_pt2(pt2, nkpts, kconserv, [km,kk,kb], [None,None,(b0,b1),(c0,c1)], -1.*tmp)

                    # Calculating Wovoo array
                    if build_ip_t3p2:
                        km = kconserv[ka,ki,kc]
                        #tmp_oovv1 = eris.oovv[km,ki,kc,:,:,c0:c1,a0:a1]
                        tmp = einsum('abcijk,mica->mbkj', Ptmp_t3Tv, tmp_oovv1)
                        Wmbkj_out[km,kb,kk,:,b0:b1,:,:] += tmp

                    # Calculating Wvvvo array
                    if build_ea_t3p2:
                        ke = kconserv[ki,ka,kk]
                        #tmp_oovv2 = eris.oovv[ki,kk,ka,:,:,a0:a1,:]
                        tmp = einsum('abcijk,ikae->cbej', Ptmp_t3Tv, tmp_oovv2)
                        Wcbej_out[kc,kb,ke,c0:c1,b0:b1,:,:] -= tmp

                    fetcher.clean()
                logger.timer_debug1(cc, 'EOM-CCSD T3[2] (%d,%d,%d,%d,%d) [total=%d]'%(ki,kj,kk,ka,kb,nkpts**5), *cput0)
        comm.Barrier()
        logger.timer_debug1(cc, 'EOM-CCSD T3[2]', *cput1)
        feri_tmp.close()

    get_t3_fast_new()
    safeAllreduceInPlace(comm, pt1)
    safeAllreduceInPlace(comm, pt2)
    safeAllreduceInPlace(comm, Wmbkj_out)
    safeAllreduceInPlace(comm, Wcbej_out)

    delta_ccsd_energy = cc.energy(pt1, pt2, eris) - ccsd_energy
    logger.info(cc, 'CCSD energy T3[2] correction : %16.12e', delta_ccsd_energy)

    return delta_ccsd_energy, pt1, pt2, Wmbkj_out, Wcbej_out


def run_gamma():
    '''Script for checking gamma-point code and molecular code.'''
    mf = pbcscf.RHF(cell)
    mf.conv_tol = 1e-10
    mf.kernel()
    mo_coeff = comm.bcast(mf.mo_coeff)
    mf.mo_coeff = mo_coeff

    mycc = pbcc.RCCSD(mf)
    mycc.conv_tol = 1e-10
    eris = mycc.ao2mo()
    mycc.kernel(eris=eris)
    t1 = comm.bcast(mycc.t1)
    t2 = comm.bcast(mycc.t2)
    mycc.t1, mycc.t2 = t1, t2

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
    t1 = comm.bcast(mykcc.t1)
    t2 = comm.bcast(mykcc.t2)
    mykcc.t1, mykcc.t2 = t1, t2

    myeom = eom()
    delta_ccsd_energy, pt1, pt2, Wmbkj, Wcbej = \
        eom_rccsd.get_t3p2_amplitude_contribution_slow(myeom, mycc.t1, mycc.t2, eris=eris, build_t1_t2=True)
    real_pt1 = pt1.copy()
    real_pt2 = pt2.copy()
    cdelta_ccsd_energy, cpt1, cpt2, _, _ = get_t3p2_amplitude_contribution(mykcc, mykcc.t1, mykcc.t2, eris=keris, copy_amps=True)
    print('pt1 difference', np.linalg.norm(pt1-cpt1))
    print('pt2 difference', np.linalg.norm(pt2-cpt2))
    print('CCSD(T)_a delta energy difference', np.linalg.norm(cdelta_ccsd_energy - delta_ccsd_energy))


def run_kpoint():
    nk = [2,1,1]
    scell = super_cell(cell, nk)
    mf = pbcscf.RHF(scell)
    mf.conv_tol = 1e-12
    mf.kernel()
    mo_coeff = comm.bcast(mf.mo_coeff)
    mf.mo_coeff = mo_coeff

    mycc = pbcc.RCCSD(mf)
    mycc.conv_tol = 1e-12
    mycc.conv_tol_normt = 1e-12
    eris = mycc.ao2mo(mo_coeff)
    mycc.kernel(eris=eris)
    t1 = comm.bcast(mycc.t1)
    t2 = comm.bcast(mycc.t2)
    mycc.t1, mycc.t2 = t1, t2

    comm.Barrier()

    class eom(object):
        def __init__(self):
            self._cc = mycc
            self.max_memory = 5000
            self.verbose = 7
            self.stdout = sys.stdout

    kmf = pbcscf.KRHF(cell, kpts=cell.make_kpts(nk, with_gamma_point=True))
    kmf.conv_tol = 1e-12
    kmf.kernel()
    kmo_coeff = comm.bcast(kmf.mo_coeff)
    kmf.mo_coeff = kmo_coeff

    mykcc = pbcc.KRCCSD(kmf)
    mykcc.conv_tol = 1e-12
    keris = mykcc.ao2mo()
    mykcc.kernel(eris=keris)
    mykcc.t1 = mykcc.t1.astype(np.complex)
    mykcc.t2 = mykcc.t2.astype(np.complex)
    t1 = comm.bcast(mykcc.t1)
    t2 = comm.bcast(mykcc.t2)
    mykcc.t1, mykcc.t2 = t1, t2

    myeom = eom()
    delta_ccsd_energy, pt1, pt2, Wmbkj, Wcbej = \
        eom_rccsd.get_t3p2_amplitude_contribution_slow(myeom, mycc.t1, mycc.t2, eris=eris, build_t1_t2=True)
    real_pt1 = pt1.copy()
    real_pt2 = pt2.copy()
    cdelta_ccsd_energy, cpt1, cpt2, _, _ = get_t3p2_amplitude_contribution(mykcc, mykcc.t1, mykcc.t2, eris=keris, copy_amps=True)
    print('CCSD(T)_a delta energy difference ', np.linalg.norm(cdelta_ccsd_energy - delta_ccsd_energy/np.prod(nk)))


def run_benchmark(nocc=15, nvir=25, nk=[2,1,1]):
    def crand(shape):
        return (np.random.rand(np.prod(shape)).reshape(shape) - 0.5 - 0.5*1j -
                np.random.rand(np.prod(shape)).reshape(shape)*1j)

    nmo = nocc + nvir
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
    t1, t2 = rand_t1_t2(rand_kmf, rand_cc)
    rand_cc.t1, rand_cc.t2 = t1, t2
    nkpts = rand_cc.nkpts

    kconserv = kpts_helper.get_kconserv(rand_kmf.cell, rand_kmf.kpts)
    h5py_kwargs = {'driver':'mpio', 'comm':comm}
    class eris(object):
        def __init__(self):
            import os
            filename = 'myeris.hdf5'
            print 'writing'
            if not check_read_success(filename):
                feri = h5py.File(filename, 'w', **h5py_kwargs)

                dtype = np.complex
                eris_vovv = crand((nkpts,nkpts,nkpts,nvir,nocc,nvir,nvir))
                eris_ooov = crand((nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir))
                self.oooo = feri.create_dataset('oooo', (nkpts,nkpts,nkpts,nocc,nocc,nocc,nocc), dtype=dtype)
                self.ooov = feri.create_dataset('ooov', eris_ooov.shape, dtype=dtype)
                self.oovv = feri.create_dataset('oovv', (nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=dtype)
                self.ovov = feri.create_dataset('ovov', (nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=dtype)
                self.voov = feri.create_dataset('voov', (nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), dtype=dtype)
                self.vovv = feri.create_dataset('vovv', eris_vovv.shape, dtype=dtype)

                if rank == 0:
                    self.oooo[:] = crand((nkpts,nkpts,nkpts,nocc,nocc,nocc,nocc))
                    self.ooov[:] = eris_ooov
                    self.oovv[:] = crand((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir))
                    self.ovov[:] = crand((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir))
                    self.voov[:] = crand((nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir))
                    self.vovv[:] = eris_vovv

                feri['completed'] = True
                feri.close()

            print 'reading'
            self.feri = h5py.File(filename, 'r', **h5py_kwargs)
            self.oooo = self.feri['oooo']
            self.ooov = self.feri['ooov']
            self.oovv = self.feri['oovv']
            self.ovov = self.feri['ovov']
            self.voov = self.feri['voov']
            self.vovv = self.feri['vovv']

        def __del__(self):
            if hasattr(self, 'feri'):
                self.feri.close()

    eris = eris()
    eris.fock = np.array([np.diag(x) for x in rand_kmf.mo_energy])
    print 'getting cont'
    get_t3p2_amplitude_contribution(rand_cc, t1, t2, eris=eris, copy_amps=True)

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
    cell.verbose = 10
    cell.build()

    do_benchmark = True
    if do_benchmark:
        run_benchmark(nocc=4, nvir=22, nk=[2,2,2])
    else:
        gamma = False
        if gamma:
            run_gamma()
        else:
            run_kpoint()
