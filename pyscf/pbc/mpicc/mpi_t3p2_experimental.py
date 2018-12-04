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
        assert isinstance(x, (int, np.int, np.ndarray, list, slice))
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

def _transpose_orb(orb_indices, transpose=(0, 1, 2)):
    orb_indices = lib.flatten([[orb_indices[2*x], orb_indices[2*x+1]]
                               for x in transpose])
    return orb_indices

def get_nstep_from_slice(sl):
    length = 1
    if isinstance(sl, slice):
        cur = sl.start
        length = 0
        while cur < sl.stop:
            length += 1
            if sl.step is None:
                cur += 1
            else:
                cur += sl.step
    return length

def get_data_slices(kpt_indices, orb_indices, kconserv):
    kpt_indices = zip_kpoints(kpt_indices)
    if not hasattr(kpt_indices[0], '__len__'):  # Ensure we are working
        kpt_indices = [kpt_indices]             # with a list of lists

    a0,a1,b0,b1,c0,c1 = orb_indices
    length = len(kpt_indices)*6

    def _vijk_indices(kpt_indices, orb_indices, transpose=(0, 1, 2)):
        '''Get indices needed for t3 construction and a given transpose of (a,b,c).'''
        kpt_indices = ([kpt_indices[x] for x in transpose] +
                       [kpt_indices[x+3] for x in transpose])
        orb_indices = _transpose_orb(orb_indices, transpose)

        ki, kj, kk, ka, kb, kc = kpt_indices
        a0, a1, b0, b1, c0, c1 = orb_indices

        km_length = get_nstep_from_slice(kpt_indices[0])

        kf = kconserv[ka,ki,kb]
        km = kconserv[kc,kk,kb]
        sl00 = slice(None, None)

        vvop_idx = [ka, kb, ki, slice(a0,a1), slice(b0,b1), sl00, sl00]
        vooo_idx = [ka, ki, kj, slice(a0,a1), sl00, sl00, sl00]
        t2T_vvop_idx = [kc, kf, kj, slice(c0,c1), sl00, sl00, sl00]
        t2T_vooo_idx = [kc, kb, km, slice(c0,c1), sl00, sl00, sl00]
        return vvop_idx, vooo_idx, t2T_vvop_idx, t2T_vooo_idx

    kf_length = get_nstep_from_slice(kpt_indices[0][0])
    km_length = get_nstep_from_slice(kpt_indices[0][2])

    vvop_indices = [0] * length
    vooo_indices = [0] * length
    t2T_vvop_indices = [0] * length * kf_length
    t2T_vooo_indices = [0] * length * km_length

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
    if not hasattr(kpt_indices[0], '__len__'):  # Ensure we are working
        kpt_indices = [kpt_indices]             # with a list of lists

    a0, a1, b0, b1, c0, c1 = orb_indices
    length = len(kpt_indices)

    vvop_indices = [0] * length
    vooo_indices = [0] * length
    oovv_indices1 = [0] * length
    oovv_indices2 = [0] * length

    for ikpt, kpt in enumerate(kpt_indices):
        ki, kj, kk, ka, kb, kc = kpt
        sl00 = slice(None, None)

        vvop_indices[ikpt] = [ka, kb, ki, slice(a0,a1), slice(b0,b1), sl00, sl00]
        vooo_indices[ikpt] = [ka, ki, kj, slice(a0,a1), sl00, sl00, sl00]

        km = kconserv[ka,ki,kc]
        oovv_indices1[ikpt] = [km, ki, kc, sl00, sl00, slice(c0,c1), slice(a0,a1)]
        oovv_indices2[ikpt] = [ki, kk, ka, sl00, sl00, slice(a0,a1), sl00]

    return vvop_indices, vooo_indices, oovv_indices1, oovv_indices2

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
        self.NMAX = 300  # max cache-size
        self.offset = 0
        self.unique_job_args = {}
        pass

    def _job_name_results(self, job_name):
        return job_name + '_res'

    @profile
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

    @profile
    def request_data2(self, kpt_indices, orb_indices, kconserv, *args):
        idx_args = get_data_slices2(kpt_indices, orb_indices, kconserv)
        vvop_indices, vooo_indices, oovv_indices1, oovv_indices2 = idx_args
        for task in range(len(vvop_indices)):
            self.add_job('vvop', args[0], vvop_indices[task])
        for task in range(len(vooo_indices)):
            self.add_job('vooo', args[1], vooo_indices[task])
        for task in range(len(oovv_indices1)):
            self.add_job('oovv', args[2], oovv_indices1[task])
        for task in range(len(oovv_indices2)):
            self.add_job('oovv', args[3], oovv_indices2[task])

        self.submit('vvop')
        self.submit('vooo')
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
        vvop_indices, vooo_indices, oovv_indices1, oovv_indices2 = idx_args

        vvop = self.results('vvop', args[0], vvop_indices)
        vooo = self.results('vooo', args[1], vooo_indices)
        oovv1 = self.results('oovv', args[2], oovv_indices1)
        oovv2 = self.results('oovv', args[3], oovv_indices2)
        return vvop[0], vooo[0], oovv1[0], oovv2[0]

    def clean(self):
        for j in self.job.keys():
            try:
                if len(self.job_results[j]) > self.NMAX:
                    self._delete_job(j)
            except:
                pass
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
        for a0, a1 in lib.prange(0, nvir, 8):
            out[ki,kj,ka,a0:a1,:,:,nocc:] = vovv[kb,ka,kj,:,:,:,a0:a1].conj().transpose(3,2,1,0)
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

T3_NCONTRACT = 0

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

    @profile
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
                                                    #chunks=(nkpts,nkpts,nkpts,8,8,nocc,nmo))
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
        vir_blksize = (nvir, nvir, nvir)
        #vir_blksize = (11, 11, 11)
        #vir_blksize = (8, 8, 8)
        for a0, a1 in lib.prange(0, nvir, vir_blksize[0]):
            for b0, b1 in lib.prange(0, nvir, vir_blksize[1]):
                for c0, c1 in lib.prange(0, nvir, vir_blksize[2]):
                    tasks.append((a0,a1,b0,b1,c0,c1))
        #assert vir_blksize == (8, 8, 8)  # Assert because chunksize set manually

        def read_vvop(idx):
            return eris_vvop[idx]
        def read_vooo_C(idx):
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
        args = (read_vvop, read_vooo_C, get_t2Tvvop, get_t2Tvooo)
        args2 = (read_vvop, read_vooo_C, read_oovv, read_oovv)

        @profile
        def contract_t3Tv(kpt_indices, orb_indices, data, out=None):
            '''Calculate t3T(ransposed) array using C driver.'''
            global T3_NCONTRACT
            T3_NCONTRACT += 1
            ki, kj, kk, ka, kb, kc = kpt_indices
            a0, a1, b0, b1, c0, c1 = orb_indices
            if out is None:
                t3T = np.empty((a1-a0,b1-b0,c1-c0) + (nocc,)*3, dtype=np.complex, order='C')
            else:
                t3T = out

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

            drv = _ccsd.libcc.zcontract_t3T
            drv(t3T.ctypes.data_as(ctypes.c_void_p),
                mo_energy.ctypes.data_as(ctypes.c_void_p),
                t1T.ctypes.data_as(ctypes.c_void_p),
                fvo.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nocc), ctypes.c_int(nvir),
                mo_offset.ctypes.data_as(ctypes.c_void_p),
                slices.ctypes.data_as(ctypes.c_void_p),
                data_ptrs)
            #print 'creating ', ki, kj, kk, ka, kb, kc, lib.finger(t3T)
            return t3T

        def add_and_permute(kpt_indices, orb_indices, data, perm='ijk'):
            '''Performs permutation and addition of t3 temporary arrays.'''
            ki, kj, kk, ka, kb, kc = kpt_indices
            a0, a1, b0, b1, c0, c1 = orb_indices
            slices = np.array([a0, a1, b0, b1, c0, c1], dtype=np.int32)

            mo_offset = np.array([ki*nmo, kj*nmo, kk*nmo,
                                  ka*nmo, kb*nmo, kc*nmo], dtype=np.int32)

            tmp_t3Tv_ijk = np.asarray(data[0], dtype=np.complex, order='C')
            tmp_t3Tv_jik = np.asarray(data[1], dtype=np.complex, order='C')
            tmp_t3Tv_kji = np.asarray(data[2], dtype=np.complex, order='C')
            out_ijk = np.empty(data[0].shape, dtype=np.complex, order='C')

            if perm == 'ijk':
                swap_idx = 0
            elif perm == 'jik':
                swap_idx = 1
            elif perm == 'kji':
                swap_idx = 2
            elif perm == 'ikj':
                swap_idx = 3
            elif perm == 'jki':
                swap_idx = 4
            elif perm == 'kij':
                swap_idx = 5
            else:
                raise ValueError('No known permutation %s' % perm)

            drv = _ccsd.libcc.MPICCadd_and_permute_t3T
            drv(ctypes.c_int(nocc), ctypes.c_int(nvir),
                ctypes.c_int(swap_idx),
                out_ijk.ctypes.data_as(ctypes.c_void_p),
                tmp_t3Tv_ijk.ctypes.data_as(ctypes.c_void_p),
                tmp_t3Tv_jik.ctypes.data_as(ctypes.c_void_p),
                tmp_t3Tv_kji.ctypes.data_as(ctypes.c_void_p),
                mo_offset.ctypes.data_as(ctypes.c_void_p),
                slices.ctypes.data_as(ctypes.c_void_p))
            # for ki,kj,kk, this operation is:
            #     out = 2.*t3Tv_ijk - t3Tv_jik.transpose(0,1,2,4,3,5)
            #                       - t3Tv_kji.transpose(0,1,2,5,4,3)
            return out_ijk

        kpt_blocks = []
        #kpt_blksize = [nkpts]*3
        kpt_blksize = [6,6,6]
        for ki0, ki1 in lib.prange(0, nkpts, kpt_blksize[0]):
            for kj0, kj1 in lib.prange(0, nkpts, kpt_blksize[1]):
                for kk0, kk1 in lib.prange(0, nkpts, kpt_blksize[2]):
                    kpt_blocks.append((ki0,ki1,kj0,kj1,kk0,kk1))
        #assert tuple(kpt_blksize) == (nkpts,nkpts,nkpts)

        comm.Barrier()
        fetcher = DataHandler()
        cput2 = (time.clock(), time.time())
        count = 0
        ntotal_kpts = nkpts**2*(nkpts*(nkpts+1)*(nkpts+2))//6
        logger.debug(cc, 'Generating integrals for %3d k-points.', ntotal_kpts)
        logger.debug(cc, '    K-point block size (n=%4d): (%3d %3d %3d).', len(kpt_blocks), *kpt_blksize)
        logger.debug(cc, '    AO virt block size (n=%4d): (%3d %3d %3d).', len(tasks), *vir_blksize)
        logger.debug(cc, '    Number of looped blocks: %6d.', len(kpt_blocks)*len(tasks)*(nkpts*(nkpts+1))//2)
        time_spent_t3 = 0
        time_spent_t2 = 0
        for kblock in mpi.work_share_partition(list(np.arange(len(kpt_blocks) * nkpts**2))):
        #for kblock in mpi.work_stealing_partition(list(np.arange(50))):
            ka, kb = divmod(kblock / len(kpt_blocks), nkpts)
            iblock = kblock % len(kpt_blocks)
            ki0, ki1, kj0, kj1, kk0, kk1 = kpt_blocks[iblock]
            eaa = foo[ka][:, None] - fvv[ka][None, :]
            ebb = foo[kb][:, None] - fvv[kb][None, :]
            if ka > kb:
                continue

            logger.debug1(cc, 'Beginning K-block(%d:%d,%d:%d,%d:%d) ka,kb=(%d,%d)' %
                          (ki0,ki1,kj0,kj1,kk0,kk1,ka,kb))
            kpt_seen = 0  # To help with indexing on which iteration we are on
            for task_id, task in enumerate(tasks):
                a0, a1, b0, b1, c0, c1 = task

                kc_list = []
                for ki in range(ki0,ki1):
                  for kj in range(kj0,kj1):
                    kc_list.extend(kpts_helper.get_kconserv3(cc._scf.cell, cc.kpts,
                                                             [ki, kj, range(kk0,kk1), ka, kb]))

                tmp_t3Tv_ijk = np.empty(((ki1-ki0),(kj1-kj0),(kk1-kk0),(a1-a0),(b1-b0),(c1-c0),nocc,nocc,nocc), dtype=np.complex)
                tmp_t3Tv_jik = np.empty(((kj1-kj0),(ki1-ki0),(kk1-kk0),(a1-a0),(b1-b0),(c1-c0),nocc,nocc,nocc), dtype=np.complex)
                tmp_t3Tv_kji = np.empty(((kk1-kk0),(kj1-kj0),(ki1-ki0),(a1-a0),(b1-b0),(c1-c0),nocc,nocc,nocc), dtype=np.complex)
                tmp_t3Tv_ikj = np.empty(((ki1-ki0),(kk1-kk0),(kj1-kj0),(a1-a0),(b1-b0),(c1-c0),nocc,nocc,nocc), dtype=np.complex)
                tmp_t3Tv_jki = np.empty(((kj1-kj0),(kk1-kk0),(ki1-ki0),(a1-a0),(b1-b0),(c1-c0),nocc,nocc,nocc), dtype=np.complex)
                tmp_t3Tv_kij = np.empty(((kk1-kk0),(ki1-ki0),(kj1-kj0),(a1-a0),(b1-b0),(c1-c0),nocc,nocc,nocc), dtype=np.complex)

                cput0 = (time.clock(), time.time())
                cput1 = (time.clock(), time.time())
                for zki, ki in enumerate(range(ki0, ki1)):
                  for zkj, kj in enumerate(range(kj0, kj1)):  # Fill in `jik` from `ijk`
                    for zkk, kk in enumerate(range(kk0, kk1)):
                      kc = kc_list[zki*(kk1-kk0)*(kj1-kj0) + zkj*(kk1-kk0) + zkk]
                      if kb > kc:
                          continue
                      kpt_seen = 1

                      kpt_indices = [ki,kj,kk,ka,kb,kc]
                      fetcher.request_data(kpt_indices, task, kconserv, *args)
                      data1 = fetcher.get_data(kpt_indices, task, kconserv, *args)
                      contract_t3Tv(kpt_indices, task, data1, out=tmp_t3Tv_ijk[zki, zkj, zkk])
                      if ki in range(kj0,kj1) and kj in range(ki0,ki1):
                          tmp_t3Tv_jik[ki-kj0, kj-ki0, zkk] = tmp_t3Tv_ijk[zki, zkj, zkk]
                      else:
                          kpt_indices = [kj,ki,kk,ka,kb,kc]
                          fetcher.request_data(kpt_indices, task, kconserv, *args)
                          data1 = fetcher.get_data(kpt_indices, task, kconserv, *args)
                          contract_t3Tv(kpt_indices, task, data1, out=tmp_t3Tv_jik[zkj, zki, zkk])

                      if ki in range(kk0,kk1) and kk in range(ki0,ki1):
                          tmp_t3Tv_kji[ki-kk0, zkj, kk-ki0] = tmp_t3Tv_ijk[zki, zkj, zkk]
                      else:
                          kpt_indices = [kk,kj,ki,ka,kb,kc]
                          fetcher.request_data(kpt_indices, task, kconserv, *args)
                          data1 = fetcher.get_data(kpt_indices, task, kconserv, *args)
                          contract_t3Tv(kpt_indices, task, data1, out=tmp_t3Tv_kji[zkk, zkj, zki])

                      if kk in range(kj0,kj1) and kj in range(kk0,kk1):
                          tmp_t3Tv_ikj[zki, kj-kk0, kk-kj0] = tmp_t3Tv_ijk[zki, zkj, zkk]
                      else:
                          kpt_indices = [ki,kk,kj,ka,kb,kc]
                          fetcher.request_data(kpt_indices, task, kconserv, *args)
                          data1 = fetcher.get_data(kpt_indices, task, kconserv, *args)
                          contract_t3Tv(kpt_indices, task, data1, out=tmp_t3Tv_ikj[zki, zkk, zkj])
                      fetcher.clean()

                for zki, ki in enumerate(range(ki0, ki1)):
                  for zkj, kj in enumerate(range(kj0, kj1)):  # Fill in `jki` from `jik`
                    for zkk, kk in enumerate(range(kk0, kk1)):
                      kc = kc_list[zki*(kk1-kk0)*(kj1-kj0) + zkj*(kk1-kk0) + zkk]
                      if kb > kc:
                          continue
                      kpt_seen = 1

                      if ki in range(kk0,kk1) and kk in range(ki0,ki1):
                          tmp_t3Tv_jki[zkj, ki-kk0, kk-ki0] = tmp_t3Tv_jik[zkj, zki, zkk]
                      else:
                          kpt_indices = [kj,kk,ki,ka,kb,kc]
                          fetcher.request_data(kpt_indices, task, kconserv, *args)
                          data1 = fetcher.get_data(kpt_indices, task, kconserv, *args)
                          contract_t3Tv(kpt_indices, task, data1, out=tmp_t3Tv_jki[zkj, zkk, zki])
                      fetcher.clean()

                for zki, ki in enumerate(range(ki0, ki1)):
                  for zkj, kj in enumerate(range(kj0, kj1)):  # Fill in `kij` from `kji`
                    for zkk, kk in enumerate(range(kk0, kk1)):
                      kc = kc_list[zki*(kk1-kk0)*(kj1-kj0) + zkj*(kk1-kk0) + zkk]
                      if kb > kc:
                          continue
                      kpt_seen = 1

                      if ki in range(kj0,kj1) and ki in range(kj0,kj1):
                          tmp_t3Tv_kij[zkk, kj-ki0, ki-kj0] = tmp_t3Tv_kji[zkk, zkj, zki]
                      else:
                          kpt_indices = [kk,ki,kj,ka,kb,kc]
                          fetcher.request_data(kpt_indices, task, kconserv, *args)
                          data1 = fetcher.get_data(kpt_indices, task, kconserv, *args)
                          contract_t3Tv(kpt_indices, task, data1, out=tmp_t3Tv_kij[zkk, zki, zkj])
                      fetcher.clean()


                # For debugging...
                #
                #for zki, ki in enumerate(range(ki0, ki1)):
                #  for zkj, kj in enumerate(range(kj0, kj1)):
                #    new_kc_list = kpts_helper.get_kconserv3(cc._scf.cell, cc.kpts,
                #                                        [ki, kj, range(kk0,kk1), ka, kb])
                #    for zkk, kk in enumerate(range(kk0, kk1)):
                #        kc = new_kc_list[zkk]
                #        print 'requesting ', kj, ki, kk, ka, kb, kc
                #        kpt_indices = [kj,ki,kk,ka,kb,kc]
                #        fetcher.request_data(kpt_indices, task, kconserv, *args)
                #        data1 = fetcher.get_data(kpt_indices, task, kconserv, *args)
                #        print np.linalg.norm(tmp_t3Tv_jik[zkj,zki,zkk] - contract_t3Tv(kpt_indices, task, data1))
                #        kpt_indices = [kk,kj,ki,ka,kb,kc]
                #        fetcher.request_data(kpt_indices, task, kconserv, *args)
                #        data1 = fetcher.get_data(kpt_indices, task, kconserv, *args)
                #        print np.linalg.norm(tmp_t3Tv_kji[zkk,zkj,zki] - contract_t3Tv(kpt_indices, task, data1))
                #        kpt_indices = [kj,ki,kk,ka,kb,kc]
                #        fetcher.request_data(kpt_indices, task, kconserv, *args)
                #        data1 = fetcher.get_data(kpt_indices, task, kconserv, *args)
                #        print np.linalg.norm(tmp_t3Tv_jik[zkj,zki,zkk] - contract_t3Tv(kpt_indices, task, data1))
                #        kpt_indices = [ki,kk,kj,ka,kb,kc]
                #        fetcher.request_data(kpt_indices, task, kconserv, *args)
                #        data1 = fetcher.get_data(kpt_indices, task, kconserv, *args)
                #        print np.linalg.norm(tmp_t3Tv_ikj[zki,zkk,zkj] - contract_t3Tv(kpt_indices, task, data1))
                #        kpt_indices = [kk,ki,kj,ka,kb,kc]
                #        fetcher.request_data(kpt_indices, task, kconserv, *args)
                #        data1 = fetcher.get_data(kpt_indices, task, kconserv, *args)
                #        print np.linalg.norm(tmp_t3Tv_kij[zkk,zki,zkj] - contract_t3Tv(kpt_indices, task, data1))
                #        kpt_indices = [kj,kk,ki,ka,kb,kc]
                #        fetcher.request_data(kpt_indices, task, kconserv, *args)
                #        data1 = fetcher.get_data(kpt_indices, task, kconserv, *args)
                #        print np.linalg.norm(tmp_t3Tv_jki[zkj,zkk,zki] - contract_t3Tv(kpt_indices, task, data1))

                if kpt_seen:
                    time_spent_t3 += time.time() - cput0[1]
                    logger.timer_debug1(cc,
                        '      t3[2] gen (%d:%d,%d:%d,%d:%d) ka,kb=(%d,%d)' %
                        (ki0,ki1,kj0,kj1,kk0,kk1,ka,kb), *cput0)
                else:
                    continue  # Skip to next ieration; don't grab data

                count = 0
                cput0 = (time.clock(), time.time())
                for zkj, kj in enumerate(range(kj0,kj1)):
                  oovv_jkX = eris.oovv[zkj,kk0:kk1,:]
                  for zkk, kk in enumerate(range(kk0,kk1)):
                    for zki, ki in enumerate(range(ki0,ki1)):
                      kc = kc_list[zki*(kk1-kk0)*(kj1-kj0) + zkj*(kk1-kk0) + zkk]
                      count += 1
                      if kb > kc:
                          continue
                      kpt_indices = [[ki,kj,kk,ka,kb,kc],
                                     [ki,kj,kk,kb,ka,kc],
                                     [ki,kj,kk,kc,kb,ka],
                                     [ki,kj,kk,ka,kc,kb],
                                     [ki,kj,kk,kc,ka,kb],
                                     [ki,kj,kk,kb,kc,ka]]

                      ecc = foo[kc][:, None] - fvv[kc][None, :]

                      Ptmp_t3Tv_ijk = add_and_permute(kpt_indices[0], task,
                                      (tmp_t3Tv_ijk[zki,zkj,zkk], tmp_t3Tv_jik[zkj,zki,zkk], tmp_t3Tv_kji[zkk,zkj,zki]), 'ijk')
                      Ptmp_t3Tv_jik = add_and_permute(kpt_indices[1], task,
                                      (tmp_t3Tv_jik[zkj,zki,zkk], tmp_t3Tv_ijk[zki,zkj,zkk], tmp_t3Tv_jki[zkj,zkk,zki]), 'jik')
                      Ptmp_t3Tv_kji = add_and_permute(kpt_indices[2], task,
                                      (tmp_t3Tv_kji[zkk,zkj,zki], tmp_t3Tv_kij[zkk,zki,zkj], tmp_t3Tv_ijk[zki,zkj,zkk]), 'kji')
                      Ptmp_t3Tv_ikj = add_and_permute(kpt_indices[3], task,
                                      (tmp_t3Tv_ikj[zki,zkk,zkj], tmp_t3Tv_jki[zkj,zkk,zki], tmp_t3Tv_kij[zkk,zki,zkj]), 'ikj')
                      Ptmp_t3Tv_jki = add_and_permute(kpt_indices[4], task,
                                      (tmp_t3Tv_jki[zkj,zkk,zki], tmp_t3Tv_ikj[zki,zkk,zkj], tmp_t3Tv_jik[zkj,zki,zkk]), 'jki')
                      Ptmp_t3Tv_kij = add_and_permute(kpt_indices[5], task,
                                      (tmp_t3Tv_kij[zkk,zki,zkj], tmp_t3Tv_kji[zkk,zkj,zki], tmp_t3Tv_ikj[zki,zkk,zkj]), 'kij')

                      eris_Soovv1 = (2.*oovv_jkX[zkk,kb,:,:,b0:b1,c0:c1] -
                                        oovv_jkX[zkk,kc,:,:,c0:c1,b0:b1].transpose(0,1,3,2))
                      eris_Soovv2 = (2.*oovv_jkX[zkk,kc,:,:,c0:c1,b0:b1] -
                                        oovv_jkX[zkk,kb,:,:,b0:b1,c0:c1].transpose(0,1,3,2))
                      eris_Soovv3 = (2.*oovv_jkX[zkk,ka,:,:,a0:a1,c0:c1] -
                                        oovv_jkX[zkk,kc,:,:,c0:c1,a0:a1].transpose(0,1,3,2))
                      eris_Soovv4 = (2.*oovv_jkX[zkk,kc,:,:,c0:c1,a0:a1] -
                                        oovv_jkX[zkk,ka,:,:,a0:a1,c0:c1].transpose(0,1,3,2))
                      eris_Soovv5 = (2.*oovv_jkX[zkk,ka,:,:,a0:a1,b0:b1] -
                                        oovv_jkX[zkk,kb,:,:,b0:b1,a0:a1].transpose(0,1,3,2))
                      eris_Soovv6 = (2.*oovv_jkX[zkk,kb,:,:,b0:b1,a0:a1] -
                                        oovv_jkX[zkk,ka,:,:,a0:a1,b0:b1].transpose(0,1,3,2))

                      if ki == ka and kc == kconserv[kj, kb, kk]:
                          pt1[ka,:,a0:a1] += 0.5*einsum('abcijk,jkbc->ia', Ptmp_t3Tv_ijk, eris_Soovv1) / eaa[:,a0:a1]
                          if kb < kc:
                              pt1[ka,:,a0:a1] += 0.5*einsum('abcikj,jkcb->ia', Ptmp_t3Tv_ikj, eris_Soovv2) / eaa[:,a0:a1]

                      if ki == kb and kc == kconserv[kk, ka, kj]:
                          if ka < kb:
                              pt1[kb,:,b0:b1] += 0.5*einsum('abcjik,jkac->ib', Ptmp_t3Tv_jik, eris_Soovv3) / ebb[:,b0:b1]
                          if ka < kb and kb < kc:
                              pt1[kb,:,b0:b1] += 0.5*einsum('abckij,jkca->ib', Ptmp_t3Tv_kij, eris_Soovv4) / ebb[:,b0:b1]

                      if ki == kc and kb == kconserv[kk, ka, kj]:
                          if kb < kc:
                              pt1[kc,:,c0:c1] += 0.5*einsum('abcjki,jkab->ic', Ptmp_t3Tv_jki, eris_Soovv5) / ecc[:,c0:c1]
                          if ka < kb:
                              pt1[kc,:,c0:c1] += 0.5*einsum('abckji,jkba->ic', Ptmp_t3Tv_kji, eris_Soovv6) / ecc[:,c0:c1]

                      ## Performing contribution to pt2
                      #if ki == ka and kc == kconserv[kj, kb, kk]:
                      #    ejkbc = (foo[kj][:,None,None,None] + foo[kk][None,:,None,None] -
                      #            fvv[kb,b0:b1][None,None,:,None] - fvv[kc,c0:c1][None,None,None,:])
                      #    tmp = einsum('abcijk,ia->jkbc', Ptmp_t3Tv, 0.5*fov[ki,:,a0:a1]) / ejkbc
                      #    _add_pt2(pt2, nkpts, kconserv, [kj,kk,kb], [None,None,(b0,b1),(c0,c1)], tmp)

                      # TODO: can clean this up a bit/reduce number of lines
                      task1 = task
                      task2 = _transpose_orb(task, transpose=(1,0,2))
                      task3 = _transpose_orb(task, transpose=(2,1,0))
                      task4 = _transpose_orb(task, transpose=(0,2,1))
                      task5 = _transpose_orb(task, transpose=(2,0,1))
                      task6 = _transpose_orb(task, transpose=(1,2,0))

                      fetcher.request_data2(kpt_indices[0], task1, kconserv, *args2)
                      fetcher.request_data2(kpt_indices[1], task2, kconserv, *args2)
                      fetcher.request_data2(kpt_indices[2], task3, kconserv, *args2)
                      fetcher.request_data2(kpt_indices[3], task4, kconserv, *args2)
                      fetcher.request_data2(kpt_indices[4], task5, kconserv, *args2)
                      fetcher.request_data2(kpt_indices[5], task6, kconserv, *args2)

                      data1 = fetcher.get_data2(kpt_indices[0], task1, kconserv, *args2)
                      data2 = fetcher.get_data2(kpt_indices[1], task2, kconserv, *args2)
                      data3 = fetcher.get_data2(kpt_indices[2], task3, kconserv, *args2)
                      data4 = fetcher.get_data2(kpt_indices[3], task4, kconserv, *args2)
                      data5 = fetcher.get_data2(kpt_indices[4], task5, kconserv, *args2)
                      data6 = fetcher.get_data2(kpt_indices[5], task6, kconserv, *args2)

                      tmp_vvop1, tmp_vooo1, tmp_oovv11, tmp_oovv21 = data1
                      tmp_vvop2, tmp_vooo2, tmp_oovv12, tmp_oovv22 = data2
                      tmp_vvop3, tmp_vooo3, tmp_oovv13, tmp_oovv23 = data3
                      tmp_vvop4, tmp_vooo4, tmp_oovv14, tmp_oovv24 = data4
                      tmp_vvop5, tmp_vooo5, tmp_oovv15, tmp_oovv25 = data5
                      tmp_vvop6, tmp_vooo6, tmp_oovv16, tmp_oovv26 = data6

                      tmp_vvov1 = tmp_vvop1[:,:,:,nocc:]
                      tmp_vvov2 = tmp_vvop2[:,:,:,nocc:]
                      tmp_vvov3 = tmp_vvop3[:,:,:,nocc:]
                      tmp_vvov4 = tmp_vvop4[:,:,:,nocc:]
                      tmp_vvov5 = tmp_vvop5[:,:,:,nocc:]
                      tmp_vvov6 = tmp_vvop6[:,:,:,nocc:]

                      kd = kconserv[ka,ki,kb]
                      ejkdc = (foo[kj][:,None,None,None] + foo[kk][None,:,None,None] -
                               fvv[kd][None,None,:,None] - fvv[kc,c0:c1][None,None,None,:])
                      kd = kconserv[ka,ki,kc]
                      ejkdb = (foo[kj][:,None,None,None] + foo[kk][None,:,None,None] -
                               fvv[kd][None,None,:,None] - fvv[kb,b0:b1][None,None,None,:])
                      kd = kconserv[kc,ki,kb]
                      ejkda = (foo[kj][:,None,None,None] + foo[kk][None,:,None,None] -
                               fvv[kd][None,None,:,None] - fvv[ka,a0:a1][None,None,None,:])

                      kd = kconserv[ka,ki,kb]
                      tmp = einsum('abcijk,abid->jkdc', Ptmp_t3Tv_ijk, tmp_vvov1.conj()) / ejkdc
                      _add_pt2(pt2, nkpts, kconserv, [kj,kk,kd], [None,None,None,(c0,c1)], tmp)

                      if ka < kb:
                          kd = kconserv[ka,ki,kb]
                          tmp = einsum('abcjik,baid->jkdc', Ptmp_t3Tv_jik, tmp_vvov2.conj()) / ejkdc
                          _add_pt2(pt2, nkpts, kconserv, [kj,kk,kd], [None,None,None,(c0,c1)], tmp)

                      if kb < kc:
                          kd = kconserv[ka,ki,kc]
                          tmp = einsum('abcikj,acid->jkdb', Ptmp_t3Tv_ikj, tmp_vvov4.conj()) / ejkdb
                          _add_pt2(pt2, nkpts, kconserv, [kj,kk,kd], [None,None,None,(b0,b1)], tmp)

                          kd = kconserv[ka,ki,kc]
                          tmp = einsum('abcjki,caid->jkdb', Ptmp_t3Tv_jki, tmp_vvov5.conj()) / ejkdb
                          _add_pt2(pt2, nkpts, kconserv, [kj,kk,kd], [None,None,None,(b0,b1)], tmp)

                      if ka < kb:
                          kd = kconserv[kc,ki,kb]
                          tmp = einsum('abckji,cbid->jkda', Ptmp_t3Tv_kji, tmp_vvov3.conj()) / ejkda
                          _add_pt2(pt2, nkpts, kconserv, [kj,kk,kd], [None,None,None,(a0,a1)], tmp)

                      if ka < kb and kb < kc:
                          kd = kconserv[kc,ki,kb]
                          tmp = einsum('abckij,bcid->jkda', Ptmp_t3Tv_kij, tmp_vvov6.conj()) / ejkda
                          _add_pt2(pt2, nkpts, kconserv, [kj,kk,kd], [None,None,None,(a0,a1)], tmp)

                      km = kconserv[kc, kk, kb]
                      emkbc = (foo[km][:,None,None,None] + foo[kk][None,:,None,None] -
                              fvv[kb,b0:b1][None,None,:,None] - fvv[kc,c0:c1][None,None,None,:])
                      emkcb = (foo[km][:,None,None,None] + foo[kk][None,:,None,None] -
                              fvv[kc,c0:c1][None,None,:,None] - fvv[kb,b0:b1][None,None,None,:])
                      km = kconserv[ka, kk, kb]
                      emkab = (foo[km][:,None,None,None] + foo[kk][None,:,None,None] -
                              fvv[ka,a0:a1][None,None,:,None] - fvv[kb,b0:b1][None,None,None,:])
                      emkba = (foo[km][:,None,None,None] + foo[kk][None,:,None,None] -
                              fvv[kb,b0:b1][None,None,:,None] - fvv[ka,a0:a1][None,None,None,:])
                      km = kconserv[ka, kk, kc]
                      emkac = (foo[km][:,None,None,None] + foo[kk][None,:,None,None] -
                              fvv[ka,a0:a1][None,None,:,None] - fvv[kc,c0:c1][None,None,None,:])
                      emkca = (foo[km][:,None,None,None] + foo[kk][None,:,None,None] -
                              fvv[kc,c0:c1][None,None,:,None] - fvv[ka,a0:a1][None,None,None,:])

                      km = kconserv[kc, kk, kb]
                      tmp = einsum('abcijk,aijm->mkbc', Ptmp_t3Tv_ijk, tmp_vooo1.conj()) / emkbc
                      _add_pt2(pt2, nkpts, kconserv, [km,kk,kb], [None,None,(b0,b1),(c0,c1)], -1.*tmp)

                      if ka < kb:
                          km = kconserv[ka, kk, kc]
                          tmp = einsum('abcjik,bijm->mkac', Ptmp_t3Tv_jik, tmp_vooo2.conj()) / emkac
                          _add_pt2(pt2, nkpts, kconserv, [km,kk,ka], [None,None,(a0,a1),(c0,c1)], -1.*tmp)

                          km = kconserv[ka, kk, kb]
                          tmp = einsum('abckji,cijm->mkba', Ptmp_t3Tv_kji, tmp_vooo3.conj()) / emkba
                          _add_pt2(pt2, nkpts, kconserv, [km,kk,kb], [None,None,(b0,b1),(a0,a1)], -1.*tmp)

                      if kb < kc:
                          km = kconserv[kc, kk, kb]
                          tmp = einsum('abcikj,aijm->mkcb', Ptmp_t3Tv_ikj, tmp_vooo4.conj()) / emkcb
                          _add_pt2(pt2, nkpts, kconserv, [km,kk,kc], [None,None,(c0,c1),(b0,b1)], -1.*tmp)

                          km = kconserv[ka, kk, kb]
                          tmp = einsum('abcjki,cijm->mkab', Ptmp_t3Tv_jki, tmp_vooo5.conj()) / emkab
                          _add_pt2(pt2, nkpts, kconserv, [km,kk,ka], [None,None,(a0,a1),(b0,b1)], -1.*tmp)

                      if ka < kb and kb < kc:
                          km = kconserv[ka, kk, kc]
                          tmp = einsum('abckij,bijm->mkca', Ptmp_t3Tv_kij, tmp_vooo6.conj()) / emkca
                          _add_pt2(pt2, nkpts, kconserv, [km,kk,kc], [None,None,(c0,c1),(a0,a1)], -1.*tmp)

                      # Calculating Wovoo array
                      if build_ip_t3p2:
                          km = kconserv[ka,ki,kc]
                          tmp = einsum('abcijk,mica->mbkj', Ptmp_t3Tv_ijk, tmp_oovv11)
                          Wmbkj_out[km,kb,kk,:,b0:b1,:,:] += tmp

                          if ka < kb:
                              km = kconserv[kb,ki,kc]
                              tmp = einsum('abcjik,micb->makj', Ptmp_t3Tv_jik, tmp_oovv12)
                              Wmbkj_out[km,ka,kk,:,a0:a1,:,:] += tmp

                              km = kconserv[ka,ki,kc]
                              tmp = einsum('abckji,miac->mbkj', Ptmp_t3Tv_kji, tmp_oovv13)
                              Wmbkj_out[km,kb,kk,:,b0:b1,:,:] += tmp

                          if kb < kc:
                              km = kconserv[ka,ki,kb]
                              tmp = einsum('abcikj,miba->mckj', Ptmp_t3Tv_ikj, tmp_oovv14)
                              Wmbkj_out[km,kc,kk,:,c0:c1,:,:] += tmp

                              km = kconserv[kb,ki,kc]
                              tmp = einsum('abcjki,mibc->makj', Ptmp_t3Tv_jki, tmp_oovv15)
                              Wmbkj_out[km,ka,kk,:,a0:a1,:,:] += tmp

                          if ka < kb and kb < kc:
                              km = kconserv[ka,ki,kb]
                              tmp = einsum('abckij,miab->mckj', Ptmp_t3Tv_kij, tmp_oovv16)
                              Wmbkj_out[km,kc,kk,:,c0:c1,:,:] += tmp

                      # Calculating Wvvvo array
                      if build_ea_t3p2:
                          ke = kconserv[ki,ka,kk]
                          tmp = einsum('abcijk,ikae->cbej', Ptmp_t3Tv_ijk, tmp_oovv21)
                          Wcbej_out[kc,kb,ke,c0:c1,b0:b1,:,:] -= tmp

                          if ka < kb:
                              ke = kconserv[ki,kb,kk]
                              tmp = einsum('abcjik,ikbe->caej', Ptmp_t3Tv_jik, tmp_oovv22)
                              Wcbej_out[kc,ka,ke,c0:c1,a0:a1,:,:] -= tmp

                              ke = kconserv[ki,kc,kk]
                              tmp = einsum('abckji,ikce->abej', Ptmp_t3Tv_kji, tmp_oovv23)
                              Wcbej_out[ka,kb,ke,a0:a1,b0:b1,:,:] -= tmp

                          if kb < kc:
                              ke = kconserv[ki,ka,kk]
                              tmp = einsum('abcikj,ikae->bcej', Ptmp_t3Tv_ikj, tmp_oovv24)
                              Wcbej_out[kb,kc,ke,b0:b1,c0:c1,:,:] -= tmp

                              ke = kconserv[ki,kc,kk]
                              tmp = einsum('abcjki,ikce->baej', Ptmp_t3Tv_jki, tmp_oovv25)
                              Wcbej_out[kb,ka,ke,b0:b1,a0:a1,:,:] -= tmp

                          if ka < kb and kb < kc:
                              ke = kconserv[ki,kb,kk]
                              tmp = einsum('abckij,ikbe->acej', Ptmp_t3Tv_kij, tmp_oovv26)
                              Wcbej_out[ka,kc,ke,a0:a1,c0:c1,:,:] -= tmp
                      fetcher.clean()

                logger.timer_debug1(cc, 't1[2],t2[2] gen (%d:%d,%d:%d,%d:%d) ka,kb=(%d,%d)' %
                                    (ki0,ki1,kj0,kj1,kk0,kk1,ka,kb), *cput0)
                logger.timer_debug1(cc, 'EOM t3[2] total (%d:%d,%d:%d,%d:%d) ka,kb=(%d,%d)' %
                                    (ki0,ki1,kj0,kj1,kk0,kk1,ka,kb), *cput1)


        global T3_NCONTRACT
        T3_NCONTRACT = comm.allreduce(T3_NCONTRACT)
        comm.Barrier()
        logger.debug1(cc, 'Generated %s integrals (=%s without symmetry).', T3_NCONTRACT, ntotal_kpts*6*len(tasks))
        logger.timer_debug1(cc, 'EOM-CCSD T3[2]', *cput2)
        logger.debug1(cc, 'Total time spent in t3 %s', time_spent_t3)
        logger.debug1(cc, 'Total time spent in t2/t1 %s', (time.time() - cput2[1]) - time_spent_t3)
        feri_tmp.close()

    get_t3_fast_new()

    cput0 = (time.clock(), time.time())
    safeAllreduceInPlace(comm, pt1)
    safeAllreduceInPlace(comm, pt2)
    if build_ip_t3p2:
        safeAllreduceInPlace(comm, Wmbkj_out)
    if build_ea_t3p2:
        safeAllreduceInPlace(comm, Wcbej_out)
    logger.timer_debug1(cc, 'reducing integrals', *cput0)

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
    nk = [1,1,2]
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
        np.random.seed(0)
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
                np.random.seed(0)
                feri = h5py.File(filename, 'w', **h5py_kwargs)

                my_rand_vovv = crand((nvir,nocc,nvir,nvir))
                my_rand_ooov = crand((nocc,nocc,nocc,nvir))
                dtype = np.complex
                self.oooo = feri.create_dataset('oooo', (nkpts,nkpts,nkpts,nocc,nocc,nocc,nocc), dtype=dtype)
                self.ooov = feri.create_dataset('ooov', (nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir), dtype=dtype)
                self.oovv = feri.create_dataset('oovv', (nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=dtype)
                self.ovov = feri.create_dataset('ovov', (nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=dtype)
                self.voov = feri.create_dataset('voov', (nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), dtype=dtype)
                self.vovv = feri.create_dataset('vovv', (nkpts,nkpts,nkpts,nvir,nocc,nvir,nvir), dtype=dtype)

                crand_oooo = crand((nocc,nocc,nocc,nocc))
                crand_ooov = my_rand_ooov
                crand_vovv = my_rand_vovv
                crand_oovv = crand((nocc,nocc,nvir,nvir))
                crand_ovov = crand((nocc,nvir,nocc,nvir))
                crand_voov = crand((nvir,nocc,nocc,nvir))
                kblocks = []
                for ki, kj, kk in product(range(nkpts), repeat=3):
                    kblocks.append((ki,kj,kk))
                for kblock in mpi.work_share_partition(kblocks, loadmin=2):
                    ki, kj, kk = kblock
                    self.oooo[ki,kj,kk] = crand_oooo
                    self.ooov[ki,kj,kk] = crand_ooov
                    self.oovv[ki,kj,kk] = crand_oovv
                    self.ovov[ki,kj,kk] = crand_ovov
                    self.voov[ki,kj,kk] = crand_voov
                    for a0, a1 in lib.prange(0, nvir, 8):
                        self.vovv[ki,kj,kk,a0:a1] = crand_vovv[a0:a1,:,:,:]

                feri.attrs['completed'] = True
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
    get_t3p2_amplitude_contribution(rand_cc, t1, t2, eris=eris, copy_amps=True,
                                    build_ip_t3p2=True, build_ea_t3p2=True)

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
        run_benchmark(nocc=8, nvir=22, nk=[1,1,3])
    else:
        gamma = False
        if gamma:
            run_gamma()
        else:
            run_kpoint()
