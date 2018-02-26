import numpy
from geom import rotation_mat
from geom import SymmSys

#ROTATION_OPERATOR = {
#    '1': numpy.array([1, 0, 0],
#                     [0, 1, 0],
#                     [0, 0, 1]),
#    '2a': numpy.array([1,  0,  0],
#                      [0, -1,  0],
#                      [0,  0, -1]),
#    '2b': numpy.array([-1,  0,  0],
#                      [ 0,  1,  0],
#                      [ 0,  0, -1]),
#    '2c': numpy.array([-1,  0,  0],
#                      [ 0, -1,  0],
#                      [ 0,  0,  1]),
#    '2d': numpy.array([ 0,  1,  0],
#                      [ 1,  0,  0],
#                      [ 0,  0, -1]),
#    '2e': numpy.array([ 0, -1,  0],
#                      [-1,  0,  0],
#                      [ 0,  0, -1]),
#    '2f': numpy.array([ 1, -1,  0],
#                      [ 0, -1,  0],
#                      [ 0,  0, -1]),
#    '2g': numpy.array([ 1,  0,  0],
#                      [ 1, -1,  0],
#                      [ 0,  0, -1]),
#    '3q': numpy.array([ 0,  0,  1],
#                      [ 1,  0,  0],
#                      [ 0,  1,  0]),
#    '3c': numpy.array([ 0, -1,  0],
#                      [ 1, -1,  0],
#                      [ 0,  0,  1]),
#    '4c': numpy.array([ 0, -1,  0],
#                      [ 1,  0,  0],
#                      [ 0,  0,  1]),
#    '6c': numpy.array([ 1, -1,  0],
#                      [ 1,  0,  0],
#                      [ 0,  0,  1]),
#
#}

TOLERANCE = 1e-14

#def pairs_invariant_to_op(op, order=2):
#    tol = TOLERANCE
#    coords = rawsys.atoms[:, 1:]
#    new_coords = numpy.dot(coords, op)
#    dist = numpy.linalg.norm(coords[:,None] - new_coords[None,:], axis=2)
#    try:
#        # Find where the new coordinate and old coordinate differ by less than tolerance.
#        # Remove all duplicate pairs of equivalent indices.
#        pairs = [(ix, numpy.where(x < tol)[0][0]) for ix, x in enumerate(dist)]
#        pairs = numpy.sort(pairs)
#        unique_pairs = numpy.unique(pairs, axis=0)
#    except IndexError:  # a zero-value was not found for some index
#        return False
#
#    if (len(numpy.unique(unique_pairs[:,0])) < len(unique_pairs[:,0]) or
#        len(numpy.unique(unique_pairs[:,1])) < len(unique_pairs[:,1])):
#        # We have one index equal to more than one other index due to
#        # two input values being the same... throw error?
#        raise AssertionError('No one-to-one mapping between old coordinates and new '
#                             'coordinates under operation; operation, old coordinates =',
#                             op, coords)
#
#    return unique_pairs
#    #yield all((_vec_in_vecs(x, coords) for x in new_coords))

def pairs_invariant_to_op(symsys, op):
    tol = TOLERANCE
    coords = symsys.atoms[:, 1:]
    new_coords = numpy.dot(coords, op)
    dist = numpy.linalg.norm(coords[:,None] - new_coords[None,:], axis=2)

    try:
        # Find where the new coordinate and old coordinate differ by less than tolerance.
        # Remove all duplicate pairs of equivalent indices.
        pairs = [(ix, numpy.where(x < tol)[0][0]) for ix, x in enumerate(dist)]
        unique_pairs = []
        seen = numpy.zeros(len(coords), dtype=bool)
        for kpt_idx, mapped_kpt_idx in pairs:

            # Find how each k-point changed when the operation was applied, finding
            # the subgroup for the given operation.
            if not seen[kpt_idx]:
                mapping_history = [kpt_idx,]
                seen[kpt_idx] = 1
                next_mapped_kpt_idx = pairs[kpt_idx][1]

                # Keep following the mapping until we reach the original kpoint.
                while next_mapped_kpt_idx != kpt_idx:
                    next_kpt_idx, next_mapped_kpt_idx = pairs[next_mapped_kpt_idx]
                    mapping_history.append(next_kpt_idx)
                    seen[next_kpt_idx] = 1
                unique_pairs.append([mapping_history, -1])

    except IndexError:  # np.where failed; one of the original atoms was not mapped onto another
        return False

    return unique_pairs

def get_stars(kpts, atoms, only_inversion=False):
    kpts_object = zip(['GHOST']*len(kpts), kpts)
    kptsys = SymmSys(kpts_object, recenter_coords=False)
    atomsys = SymmSys(atoms)
    if only_inversion:
        op = -1
    else:
        kpt_rot = kptsys.search_possible_rotations()
        kpt_rot = sorted(kpt_rot, reverse=True, key=lambda x: x[1])

        for zaxis, n in kpt_rot:
            if atomsys.has_rotation(zaxis, n):
                op = rotation_mat(zaxis, numpy.pi*2/n)
                break
    stars = pairs_invariant_to_op(kptsys, op)
    return stars
