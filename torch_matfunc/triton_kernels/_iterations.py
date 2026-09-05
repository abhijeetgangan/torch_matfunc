"""Newton-Schulz inverses and Denman-Beavers iterations for the fused kernels."""

import triton
import triton.language as tl

from torch_matfunc.triton_kernels._core import (
    eye,
    matrix_1_norm,
    matrix_1_norm_c,
    mm,
    mm_c,
)

# Newton-Schulz budget: prefix steps, then SCHULZ_REST more if the residual check fails.
SCHULZ_PREFIX = 12
SCHULZ_REST = 24
# ||A X - I||_1 early-exit tolerance; must sit above the dtype floor of about n*eps*kappa.
INV_TOL_FP64 = 1e-11
INV_TOL_FP32 = 4e-6


@triton.jit
def matrix_inf_norm(A, N: tl.constexpr):
    """Induced inf-norm: max row sum."""
    row_sums = tl.sum(tl.abs(A), axis=1)
    return tl.max(row_sums)


@triton.jit
def matrix_inf_norm_c(Ar, Ai, N: tl.constexpr):
    return tl.max(tl.sum(tl.sqrt(Ar * Ar + Ai * Ai), axis=1))


@triton.jit
def entry(M, row, col, N: tl.constexpr):
    rows = tl.arange(0, N)
    cols = tl.arange(0, N)
    return tl.sum(tl.where((rows[:, None] == row) & (cols[None, :] == col), M, 0.0))


@triton.jit
def mat2(a00, a01, a10, a11, dtype):
    i = tl.arange(0, 2)
    j = tl.arange(0, 2)
    z = tl.zeros((2, 2), dtype=dtype)
    z = tl.where((i[:, None] == 0) & (j[None, :] == 0), a00, z)
    z = tl.where((i[:, None] == 0) & (j[None, :] == 1), a01, z)
    z = tl.where((i[:, None] == 1) & (j[None, :] == 0), a10, z)
    z = tl.where((i[:, None] == 1) & (j[None, :] == 1), a11, z)
    return z


@triton.jit
def floor_det(det):
    """Floor ``|det|`` at 1e-30, keeping the sign of det."""
    return tl.where(tl.abs(det) < 1e-30, tl.where(det < 0, -1e-30, 1e-30), det)


@triton.jit
def invert2(A):
    a = entry(A, 0, 0, 2)
    b = entry(A, 0, 1, 2)
    c = entry(A, 1, 0, 2)
    d = entry(A, 1, 1, 2)
    det = floor_det(a * d - b * c)
    invdet = 1.0 / det
    return mat2(d * invdet, -b * invdet, -c * invdet, a * invdet, A.dtype)


@triton.jit
def invert2_c(Ar, Ai):
    """2x2 complex inverse; pivot floor on ``|det|``."""
    ar = entry(Ar, 0, 0, 2)
    ai = entry(Ai, 0, 0, 2)
    br = entry(Ar, 0, 1, 2)
    bi = entry(Ai, 0, 1, 2)
    cr = entry(Ar, 1, 0, 2)
    ci = entry(Ai, 1, 0, 2)
    dr = entry(Ar, 1, 1, 2)
    di = entry(Ai, 1, 1, 2)
    det_r = ar * dr - ai * di - (br * cr - bi * ci)
    det_i = ar * di + ai * dr - (br * ci + bi * cr)
    det_abs = tl.sqrt(det_r * det_r + det_i * det_i)
    det_abs = tl.where(det_abs < 1e-30, 1e-30, det_abs)
    inv_r = det_r / det_abs / det_abs
    inv_i = -det_i / det_abs / det_abs
    # adj(A) / det
    out_r = mat2(
        dr * inv_r - di * inv_i,
        -(br * inv_r - bi * inv_i),
        -(cr * inv_r - ci * inv_i),
        ar * inv_r - ai * inv_i,
        Ar.dtype,
    )
    out_i = mat2(
        dr * inv_i + di * inv_r,
        -(br * inv_i + bi * inv_r),
        -(cr * inv_i + ci * inv_r),
        ar * inv_i + ai * inv_r,
        Ai.dtype,
    )
    return out_r, out_i


@triton.jit
def mat4_row(M, e0, e1, e2, e3, r):
    rows = tl.arange(0, 4)
    cols = tl.arange(0, 4)
    M = tl.where((rows[:, None] == r) & (cols[None, :] == 0), e0, M)
    M = tl.where((rows[:, None] == r) & (cols[None, :] == 1), e1, M)
    M = tl.where((rows[:, None] == r) & (cols[None, :] == 2), e2, M)
    M = tl.where((rows[:, None] == r) & (cols[None, :] == 3), e3, M)
    return M


@triton.jit
def invert4(A):
    """Closed-form 4x4 inverse via the adjugate built from 2x2-minor cofactors."""
    a00 = entry(A, 0, 0, 4)
    a01 = entry(A, 0, 1, 4)
    a02 = entry(A, 0, 2, 4)
    a03 = entry(A, 0, 3, 4)
    a10 = entry(A, 1, 0, 4)
    a11 = entry(A, 1, 1, 4)
    a12 = entry(A, 1, 2, 4)
    a13 = entry(A, 1, 3, 4)
    a20 = entry(A, 2, 0, 4)
    a21 = entry(A, 2, 1, 4)
    a22 = entry(A, 2, 2, 4)
    a23 = entry(A, 2, 3, 4)
    a30 = entry(A, 3, 0, 4)
    a31 = entry(A, 3, 1, 4)
    a32 = entry(A, 3, 2, 4)
    a33 = entry(A, 3, 3, 4)

    s0 = a00 * a11 - a10 * a01
    s1 = a00 * a12 - a10 * a02
    s2 = a00 * a13 - a10 * a03
    s3 = a01 * a12 - a11 * a02
    s4 = a01 * a13 - a11 * a03
    s5 = a02 * a13 - a12 * a03
    c5 = a22 * a33 - a32 * a23
    c4 = a21 * a33 - a31 * a23
    c3 = a21 * a32 - a31 * a22
    c2 = a20 * a33 - a30 * a23
    c1 = a20 * a32 - a30 * a22
    c0 = a20 * a31 - a30 * a21

    det = floor_det(s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0)
    inv = 1.0 / det

    out = tl.zeros((4, 4), dtype=A.dtype)
    out = mat4_row(
        out,
        (a11 * c5 - a12 * c4 + a13 * c3) * inv,
        (-a01 * c5 + a02 * c4 - a03 * c3) * inv,
        (a31 * s5 - a32 * s4 + a33 * s3) * inv,
        (-a21 * s5 + a22 * s4 - a23 * s3) * inv,
        0,
    )
    out = mat4_row(
        out,
        (-a10 * c5 + a12 * c2 - a13 * c1) * inv,
        (a00 * c5 - a02 * c2 + a03 * c1) * inv,
        (-a30 * s5 + a32 * s2 - a33 * s1) * inv,
        (a20 * s5 - a22 * s2 + a23 * s1) * inv,
        1,
    )
    out = mat4_row(
        out,
        (a10 * c4 - a11 * c2 + a13 * c0) * inv,
        (-a00 * c4 + a01 * c2 - a03 * c0) * inv,
        (a30 * s4 - a31 * s2 + a33 * s0) * inv,
        (-a20 * s4 + a21 * s2 - a23 * s0) * inv,
        2,
    )
    out = mat4_row(
        out,
        (-a10 * c3 + a11 * c1 - a12 * c0) * inv,
        (a00 * c3 - a01 * c1 + a02 * c0) * inv,
        (-a30 * s3 + a31 * s1 - a32 * s0) * inv,
        (a20 * s3 - a21 * s1 + a22 * s0) * inv,
        3,
    )
    return out


@triton.jit
def schulz_cubic_iters(A, X, I, N: tl.constexpr, niter):
    """Hyperpower inverse of order 3: X <- X (I + R + R^2), R = I - AX."""
    for _ in range(niter):
        R = I - mm(A, X, N)
        X = mm(X, I + R + mm(R, R, N), N)
    return X


@triton.jit
def invert_schulz(A, N: tl.constexpr, nprefix, nrest, inv_tol):
    """Scaled cubic Newton-Schulz inverse per Higham, X0 = A^T / (||A||_1 ||A||_inf). Runtime
    trip counts nprefix/nrest keep Triton from unrolling the iteration loops."""
    I = eye(N, A.dtype)
    scale = matrix_1_norm(A, N) * matrix_inf_norm(A, N)
    scale = tl.where(scale < 1e-30, 1e-30, scale)
    X = tl.trans(A) / scale
    X = schulz_cubic_iters(A, X, I, N, nprefix)
    residual = matrix_1_norm(mm(A, X, N) - I, N)
    # Zero-trip rest phase, not a scalar if: mid-kernel ifs around these loops miscompile.
    nrest_eff = tl.where(residual >= inv_tol, nrest, 0)
    X = schulz_cubic_iters(A, X, I, N, nrest_eff)
    return X


@triton.jit
def invert(A, N: tl.constexpr, nprefix, nrest, inv_tol):
    """nxn inverse: closed-form adjugate 2x2 / 4x4, Newton-Schulz for n>=8."""
    if N == 2:
        return invert2(A)
    if N == 4:
        return invert4(A)
    return invert_schulz(A, N, nprefix, nrest, inv_tol)


@triton.jit
def invert_warm(A, X0, N: tl.constexpr, nprefix, nrest, inv_tol):
    """invert() with warm-start candidate X0; cold init and full prefix when X0 stalls."""
    if N == 2:
        return invert2(A)
    if N == 4:
        return invert4(A)
    I = eye(N, A.dtype)
    r0 = matrix_1_norm(mm(A, X0, N) - I, N)
    scale = matrix_1_norm(A, N) * matrix_inf_norm(A, N)
    scale = tl.where(scale < 1e-30, 1e-30, scale)
    use_warm = r0 < 0.5
    X = tl.where(use_warm, X0, tl.trans(A) / scale)
    # A contracting warm start needs 4 cubic steps in place of the cold prefix.
    X = schulz_cubic_iters(A, X, I, N, tl.where(use_warm, 4, nprefix))
    residual = matrix_1_norm(mm(A, X, N) - I, N)
    nrest_eff = tl.where(residual >= inv_tol, nrest, 0)
    return schulz_cubic_iters(A, X, I, N, nrest_eff)


@triton.jit
def denman_beavers_step(Y, Z, N: tl.constexpr, nprefix, nrest, inv_tol):
    # DB converges to Z = Y^-1, so each co-iterate warm-starts the other's inverse.
    Y_inv = invert_warm(Y, Z, N, nprefix, nrest, inv_tol)
    Z_inv = invert_warm(Z, Y, N, nprefix, nrest, inv_tol)
    return 0.5 * (Y + Z_inv), 0.5 * (Z + Y_inv)


@triton.jit
def denman_beavers(
    A, N: tl.constexpr, db_prefix, db_rest, nprefix, nrest, inv_tol, TOL: tl.constexpr
):
    """Principal sqrt via Denman-Beavers. Callers prescale to near unit norm, so TOL is relative.

    No scalar ``if`` here: mid-kernel ifs miscompile, so skip logic uses zero-trip loop counts.
    """
    I = eye(N, A.dtype)
    Y = A
    Z = I
    for _ in range(db_prefix):
        Y, Z = denman_beavers_step(Y, Z, N, nprefix, nrest, inv_tol)
    residual = matrix_1_norm(mm(Y, Y, N) - A, N)
    # With db_prefix = db_rest = 0 this returns A unchanged, which keeps logm's ISS if-free.
    nrest_eff = tl.where(residual >= TOL, db_rest, 0)
    for _ in range(nrest_eff):
        Y, Z = denman_beavers_step(Y, Z, N, nprefix, nrest, inv_tol)
    return Y


@triton.jit
def cmul(ar, ai, br, bi):
    """Complex scalar product ``(ar + i*ai)(br + i*bi)``."""
    return ar * br - ai * bi, ar * bi + ai * br


@triton.jit
def minor2_c(pr, pi, qr, qi, rr, ri, sr, si):
    """Complex 2x2 minor ``p*s - r*q`` from scalar entries."""
    ur, ui = cmul(pr, pi, sr, si)
    vr, vi = cmul(rr, ri, qr, qi)
    return ur - vr, ui - vi


@triton.jit
def cof3_c(xr, xi, pr, pi, yr, yi, qr, qi, zr, zi, rr, ri):
    """Alternating 3-term cofactor ``x*p - y*q + z*r`` over complex scalars."""
    ur, ui = cmul(xr, xi, pr, pi)
    vr, vi = cmul(yr, yi, qr, qi)
    wr, wi = cmul(zr, zi, rr, ri)
    return ur - vr + wr, ui - vi + wi


@triton.jit
def cof3n_c(xr, xi, pr, pi, yr, yi, qr, qi, zr, zi, rr, ri):
    """Negative-parity cofactor ``-x*p + y*q - z*r`` over complex scalars."""
    ur, ui = cmul(xr, xi, pr, pi)
    vr, vi = cmul(yr, yi, qr, qi)
    wr, wi = cmul(zr, zi, rr, ri)
    return -ur + vr - wr, -ui + vi - wi


@triton.jit
def invert4_c(Ar, Ai):
    """Complex 4x4 adjugate inverse: :func:`invert4`'s formulas over complex scalars."""
    a00r = entry(Ar, 0, 0, 4)
    a00i = entry(Ai, 0, 0, 4)
    a01r = entry(Ar, 0, 1, 4)
    a01i = entry(Ai, 0, 1, 4)
    a02r = entry(Ar, 0, 2, 4)
    a02i = entry(Ai, 0, 2, 4)
    a03r = entry(Ar, 0, 3, 4)
    a03i = entry(Ai, 0, 3, 4)
    a10r = entry(Ar, 1, 0, 4)
    a10i = entry(Ai, 1, 0, 4)
    a11r = entry(Ar, 1, 1, 4)
    a11i = entry(Ai, 1, 1, 4)
    a12r = entry(Ar, 1, 2, 4)
    a12i = entry(Ai, 1, 2, 4)
    a13r = entry(Ar, 1, 3, 4)
    a13i = entry(Ai, 1, 3, 4)
    a20r = entry(Ar, 2, 0, 4)
    a20i = entry(Ai, 2, 0, 4)
    a21r = entry(Ar, 2, 1, 4)
    a21i = entry(Ai, 2, 1, 4)
    a22r = entry(Ar, 2, 2, 4)
    a22i = entry(Ai, 2, 2, 4)
    a23r = entry(Ar, 2, 3, 4)
    a23i = entry(Ai, 2, 3, 4)
    a30r = entry(Ar, 3, 0, 4)
    a30i = entry(Ai, 3, 0, 4)
    a31r = entry(Ar, 3, 1, 4)
    a31i = entry(Ai, 3, 1, 4)
    a32r = entry(Ar, 3, 2, 4)
    a32i = entry(Ai, 3, 2, 4)
    a33r = entry(Ar, 3, 3, 4)
    a33i = entry(Ai, 3, 3, 4)

    s0r, s0i = minor2_c(a00r, a00i, a01r, a01i, a10r, a10i, a11r, a11i)
    s1r, s1i = minor2_c(a00r, a00i, a02r, a02i, a10r, a10i, a12r, a12i)
    s2r, s2i = minor2_c(a00r, a00i, a03r, a03i, a10r, a10i, a13r, a13i)
    s3r, s3i = minor2_c(a01r, a01i, a02r, a02i, a11r, a11i, a12r, a12i)
    s4r, s4i = minor2_c(a01r, a01i, a03r, a03i, a11r, a11i, a13r, a13i)
    s5r, s5i = minor2_c(a02r, a02i, a03r, a03i, a12r, a12i, a13r, a13i)
    c5r, c5i = minor2_c(a22r, a22i, a23r, a23i, a32r, a32i, a33r, a33i)
    c4r, c4i = minor2_c(a21r, a21i, a23r, a23i, a31r, a31i, a33r, a33i)
    c3r, c3i = minor2_c(a21r, a21i, a22r, a22i, a31r, a31i, a32r, a32i)
    c2r, c2i = minor2_c(a20r, a20i, a23r, a23i, a30r, a30i, a33r, a33i)
    c1r, c1i = minor2_c(a20r, a20i, a22r, a22i, a30r, a30i, a32r, a32i)
    c0r, c0i = minor2_c(a20r, a20i, a21r, a21i, a30r, a30i, a31r, a31i)

    # det = s0*c5 - s1*c4 + s2*c3 + s3*c2 - s4*c1 + s5*c0
    d1r, d1i = cof3_c(s0r, s0i, c5r, c5i, s1r, s1i, c4r, c4i, s2r, s2i, c3r, c3i)
    d2r, d2i = cof3_c(s3r, s3i, c2r, c2i, s4r, s4i, c1r, c1i, s5r, s5i, c0r, c0i)
    det_r = d1r + d2r
    det_i = d1i + d2i
    det_abs = tl.sqrt(det_r * det_r + det_i * det_i)
    det_abs = tl.where(det_abs < 1e-30, 1e-30, det_abs)
    inv_r = det_r / det_abs / det_abs
    inv_i = -det_i / det_abs / det_abs

    b00r, b00i = cof3_c(a11r, a11i, c5r, c5i, a12r, a12i, c4r, c4i, a13r, a13i, c3r, c3i)
    b01r, b01i = cof3n_c(a01r, a01i, c5r, c5i, a02r, a02i, c4r, c4i, a03r, a03i, c3r, c3i)
    b02r, b02i = cof3_c(a31r, a31i, s5r, s5i, a32r, a32i, s4r, s4i, a33r, a33i, s3r, s3i)
    b03r, b03i = cof3n_c(a21r, a21i, s5r, s5i, a22r, a22i, s4r, s4i, a23r, a23i, s3r, s3i)

    b10r, b10i = cof3n_c(a10r, a10i, c5r, c5i, a12r, a12i, c2r, c2i, a13r, a13i, c1r, c1i)
    b11r, b11i = cof3_c(a00r, a00i, c5r, c5i, a02r, a02i, c2r, c2i, a03r, a03i, c1r, c1i)
    b12r, b12i = cof3n_c(a30r, a30i, s5r, s5i, a32r, a32i, s2r, s2i, a33r, a33i, s1r, s1i)
    b13r, b13i = cof3_c(a20r, a20i, s5r, s5i, a22r, a22i, s2r, s2i, a23r, a23i, s1r, s1i)

    b20r, b20i = cof3_c(a10r, a10i, c4r, c4i, a11r, a11i, c2r, c2i, a13r, a13i, c0r, c0i)
    b21r, b21i = cof3n_c(a00r, a00i, c4r, c4i, a01r, a01i, c2r, c2i, a03r, a03i, c0r, c0i)
    b22r, b22i = cof3_c(a30r, a30i, s4r, s4i, a31r, a31i, s2r, s2i, a33r, a33i, s0r, s0i)
    b23r, b23i = cof3n_c(a20r, a20i, s4r, s4i, a21r, a21i, s2r, s2i, a23r, a23i, s0r, s0i)

    b30r, b30i = cof3n_c(a10r, a10i, c3r, c3i, a11r, a11i, c1r, c1i, a12r, a12i, c0r, c0i)
    b31r, b31i = cof3_c(a00r, a00i, c3r, c3i, a01r, a01i, c1r, c1i, a02r, a02i, c0r, c0i)
    b32r, b32i = cof3n_c(a30r, a30i, s3r, s3i, a31r, a31i, s1r, s1i, a32r, a32i, s0r, s0i)
    b33r, b33i = cof3_c(a20r, a20i, s3r, s3i, a21r, a21i, s1r, s1i, a22r, a22i, s0r, s0i)

    out_r = tl.zeros((4, 4), dtype=Ar.dtype)
    out_i = tl.zeros((4, 4), dtype=Ai.dtype)
    e00r, e00i = cmul(b00r, b00i, inv_r, inv_i)
    e01r, e01i = cmul(b01r, b01i, inv_r, inv_i)
    e02r, e02i = cmul(b02r, b02i, inv_r, inv_i)
    e03r, e03i = cmul(b03r, b03i, inv_r, inv_i)
    out_r = mat4_row(out_r, e00r, e01r, e02r, e03r, 0)
    out_i = mat4_row(out_i, e00i, e01i, e02i, e03i, 0)
    e10r, e10i = cmul(b10r, b10i, inv_r, inv_i)
    e11r, e11i = cmul(b11r, b11i, inv_r, inv_i)
    e12r, e12i = cmul(b12r, b12i, inv_r, inv_i)
    e13r, e13i = cmul(b13r, b13i, inv_r, inv_i)
    out_r = mat4_row(out_r, e10r, e11r, e12r, e13r, 1)
    out_i = mat4_row(out_i, e10i, e11i, e12i, e13i, 1)
    e20r, e20i = cmul(b20r, b20i, inv_r, inv_i)
    e21r, e21i = cmul(b21r, b21i, inv_r, inv_i)
    e22r, e22i = cmul(b22r, b22i, inv_r, inv_i)
    e23r, e23i = cmul(b23r, b23i, inv_r, inv_i)
    out_r = mat4_row(out_r, e20r, e21r, e22r, e23r, 2)
    out_i = mat4_row(out_i, e20i, e21i, e22i, e23i, 2)
    e30r, e30i = cmul(b30r, b30i, inv_r, inv_i)
    e31r, e31i = cmul(b31r, b31i, inv_r, inv_i)
    e32r, e32i = cmul(b32r, b32i, inv_r, inv_i)
    e33r, e33i = cmul(b33r, b33i, inv_r, inv_i)
    out_r = mat4_row(out_r, e30r, e31r, e32r, e33r, 3)
    out_i = mat4_row(out_i, e30i, e31i, e32i, e33i, 3)
    return out_r, out_i


@triton.jit
def schulz_cubic_iters_c(Ar, Ai, Xr, Xi, I, N: tl.constexpr, niter):
    Z = tl.zeros((N, N), dtype=Ar.dtype)
    for _ in range(niter):
        AXr, AXi = mm_c(Ar, Ai, Xr, Xi, N)
        Rr = I - AXr
        Ri = Z - AXi
        R2r, R2i = mm_c(Rr, Ri, Rr, Ri, N)
        Mr = I + Rr + R2r
        Mi = Ri + R2i
        Xr, Xi = mm_c(Xr, Xi, Mr, Mi, N)
    return Xr, Xi


@triton.jit
def invert_schulz_c(Ar, Ai, N: tl.constexpr, nprefix, nrest, inv_tol):
    """Scaled cubic Newton-Schulz: X0 = A^H / (||A||_1 ||A||_inf)."""
    I = eye(N, Ar.dtype)
    scale = matrix_1_norm_c(Ar, Ai, N) * matrix_inf_norm_c(Ar, Ai, N)
    scale = tl.where(scale < 1e-30, 1e-30, scale)
    Xr = tl.trans(Ar) / scale
    Xi = -tl.trans(Ai) / scale
    Xr, Xi = schulz_cubic_iters_c(Ar, Ai, Xr, Xi, I, N, nprefix)
    AXr, AXi = mm_c(Ar, Ai, Xr, Xi, N)
    residual = matrix_1_norm(AXr - I, N) + matrix_1_norm(AXi, N)
    # Zero-trip rest phase; see invert_schulz.
    nrest_eff = tl.where(residual >= inv_tol, nrest, 0)
    Xr, Xi = schulz_cubic_iters_c(Ar, Ai, Xr, Xi, I, N, nrest_eff)
    return Xr, Xi


@triton.jit
def invert_c(Ar, Ai, N: tl.constexpr, nprefix, nrest, inv_tol):
    if N == 2:
        return invert2_c(Ar, Ai)
    if N == 4:
        return invert4_c(Ar, Ai)
    return invert_schulz_c(Ar, Ai, N, nprefix, nrest, inv_tol)


@triton.jit
def invert_warm_c(Ar, Ai, X0r, X0i, N: tl.constexpr, nprefix, nrest, inv_tol):
    """Complex invert_c() with warm-start candidate X0; cold init when X0 stalls."""
    if N == 2:
        return invert2_c(Ar, Ai)
    if N == 4:
        return invert4_c(Ar, Ai)
    I = eye(N, Ar.dtype)
    AXr, AXi = mm_c(Ar, Ai, X0r, X0i, N)
    r0 = matrix_1_norm(AXr - I, N) + matrix_1_norm(AXi, N)
    scale = matrix_1_norm_c(Ar, Ai, N) * matrix_inf_norm_c(Ar, Ai, N)
    scale = tl.where(scale < 1e-30, 1e-30, scale)
    use_warm = r0 < 0.5
    Xr = tl.where(use_warm, X0r, tl.trans(Ar) / scale)
    Xi = tl.where(use_warm, X0i, -tl.trans(Ai) / scale)
    Xr, Xi = schulz_cubic_iters_c(Ar, Ai, Xr, Xi, I, N, tl.where(use_warm, 4, nprefix))
    AXr, AXi = mm_c(Ar, Ai, Xr, Xi, N)
    residual = matrix_1_norm(AXr - I, N) + matrix_1_norm(AXi, N)
    nrest_eff = tl.where(residual >= inv_tol, nrest, 0)
    return schulz_cubic_iters_c(Ar, Ai, Xr, Xi, I, N, nrest_eff)


@triton.jit
def denman_beavers_step_c(Yr, Yi, Zr, Zi, N: tl.constexpr, nprefix, nrest, inv_tol):
    # DB converges to Z = Y^-1, so each co-iterate warm-starts the other's inverse.
    Yinv_r, Yinv_i = invert_warm_c(Yr, Yi, Zr, Zi, N, nprefix, nrest, inv_tol)
    Zinv_r, Zinv_i = invert_warm_c(Zr, Zi, Yr, Yi, N, nprefix, nrest, inv_tol)
    return 0.5 * (Yr + Zinv_r), 0.5 * (Yi + Zinv_i), 0.5 * (Zr + Yinv_r), 0.5 * (Zi + Yinv_i)


@triton.jit
def denman_beavers_c(
    Ar, Ai, N: tl.constexpr, db_prefix, db_rest, nprefix, nrest, inv_tol, TOL: tl.constexpr
):
    """Complex :func:`denman_beavers`; same prescale and control-flow contract."""
    I = eye(N, Ar.dtype)
    Z0 = tl.zeros((N, N), dtype=Ar.dtype)
    Yr, Yi = Ar, Ai
    Zr, Zi = I, Z0
    for _ in range(db_prefix):
        Yr, Yi, Zr, Zi = denman_beavers_step_c(Yr, Yi, Zr, Zi, N, nprefix, nrest, inv_tol)
    YYr, YYi = mm_c(Yr, Yi, Yr, Yi, N)
    residual = matrix_1_norm_c(YYr - Ar, YYi - Ai, N)
    nrest_eff = tl.where(residual >= TOL, db_rest, 0)
    for _ in range(nrest_eff):
        Yr, Yi, Zr, Zi = denman_beavers_step_c(Yr, Yi, Zr, Zi, N, nprefix, nrest, inv_tol)
    return Yr, Yi
