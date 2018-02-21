#!/usr/bin/env python

'''
Mean field with k-points sampling

The 2-electron integrals are computed using Poisson solver with FFT by default.
In most scenario, it should be used with pseudo potential.
'''

from pyscf.pbc import gto, scf, dft
import numpy

cell = gto.M(
    a = numpy.eye(3)*3.5668,
    atom = '''C     0.      0.      0.
              C     0.8917  0.8917  0.8917
              C     1.7834  1.7834  0.
              C     2.6751  2.6751  0.8917
              C     1.7834  0.      1.7834
              C     2.6751  0.8917  2.6751
              C     0.      1.7834  1.7834
              C     0.8917  2.6751  2.6751''',
    basis = 'gth-szv',
    pseudo = 'gth-pade',
    verbose = 0,
)

nk = [3,3,3]  # 4 k-poins for each axis, 4^3=64 kpts in total
kpts = cell.make_kpts(nk, with_gamma_point=False)
kmf = dft.KRKS(cell, kpts)
# Turn to the atomic grids if you like
kmf.grids = dft.gen_grid.BeckeGrids(cell)
kmf.xc = 'lda'
kmf.kernel()

kpts_object = zip(['GHOST']*numpy.prod(nk), kpts)

from geom import *
rawsys = SymmSys(kpts_object, recenter_coords=False)
#print rawsys.search_possible_rotations()
mirror_vec = rawsys.search_mirrorx(n=1)
mirror_op = householder(mirror_vec)
print mirror_vec

TOLERANCE = 1e-14

def pairs_invariant_to_op(op, order=2):
    tol = TOLERANCE
    coords = rawsys.atoms[:, 1:]
    new_coords = numpy.dot(coords, op)
    dist = numpy.linalg.norm(coords[:,None] - new_coords[None,:], axis=2)
    try:
        # Find where the new coordinate and old coordinate differ by less than tolerance.
        # Remove all duplicate pairs of equivalent indices.
        pairs = [(ix, numpy.where(x < tol)[0][0]) for ix, x in enumerate(dist)]
        pairs = numpy.sort(pairs)
        unique_pairs = numpy.unique(pairs, axis=0)
    except IndexError:  # a zero-value was not found for some index
        return False

    if (len(numpy.unique(unique_pairs[:,0])) < len(unique_pairs[:,0]) or
        len(numpy.unique(unique_pairs[:,1])) < len(unique_pairs[:,1])):
        # We have one index equal to more than one other index due to
        # two input values being the same... throw error?
        raise AssertionError('No one-to-one mapping between old coordinates and new '
                             'coordinates under operation; operation, old coordinates =',
                             op, coords)

    return unique_pairs
    #yield all((_vec_in_vecs(x, coords) for x in new_coords))

#def create_matching_pairs(pair1, pair2):

kpt_inv = pairs_invariant_to_op(-1)
print kpt_inv
zaxis, n = rawsys.search_highest_cn()
print zaxis, n
#print [(rawsys.atoms[x[0]], rawsys.atoms[x[1]]) for x in kpt_inv]
