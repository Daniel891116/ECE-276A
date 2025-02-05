# import numpy as np
import math
import jax.numpy as jnp
from jax import jit

@jit
def compute_exp_quat(q):
    """
    Compute the exponential of a quaternion.
    For a quaternion q = [q0, q1, q2, q3], define
        exp(q) = exp(q0) * [cos(|v|), sin(|v|)/|v| * v]
    where v = [q1, q2, q3]. When |v|==0, we define sin(|v|)/|v| = 1.
    """
    s = q[0]
    v = q[1:]
    v_norm = jnp.linalg.norm(v)
    scalar_factor = jnp.exp(s)
    # Avoid division by zero:
    safe_factor = jnp.where(v_norm > 0, jnp.sin(v_norm) / v_norm, 1.0)
    vec_factor = v * safe_factor
    # Build the quaternion as [cos(v_norm), vec_factor] and multiply by scalar_factor.
    return scalar_factor * jnp.insert(vec_factor, 0, jnp.cos(v_norm))


@jit
def compute_exp_quat_vec(q):
    """
    Compute the exponential of a quaternion vector.
    
    For each quaternion q = [q0, q1, q2, q3], we define:
    
        exp(q) = exp(q0) * [cos(|v|), sin(|v|)/|v| * v]
    
    where v = [q1, q2, q3] and |v| is its Euclidean norm.
    When |v| is zero, the vector part is defined to be zero.
    
    Parameters:
      q: jax.numpy array of shape (N, 4) representing quaternions.
      
    Returns:
      q_exp: jax.numpy array of shape (N, 4) representing the exponential.
    """
    # Compute the magnitude of the vector part.
    vmags = jnp.linalg.norm(q[:, 1:], axis=1)  # shape (N,)
    
    # Compute the exponential of the scalar part.
    scalar_exp = jnp.exp(q[:, 0])               # shape (N,)
    
    # Compute the scalar component: exp(q0)*cos(|v|)
    scalar_component = scalar_exp * jnp.cos(vmags)
    
    # Compute the safe factor: sin(|v|)/|v| when |v| > 0, otherwise 1.
    factor = jnp.where(vmags > 0, jnp.sin(vmags) / vmags, jnp.ones_like(vmags))
    
    # Compute the vector component: exp(q0)*factor*v.
    vector_component = scalar_exp[:, None] * factor[:, None] * q[:, 1:]
    
    # Combine the scalar and vector components into the resulting quaternion.
    q_exp = jnp.concatenate([scalar_component[:, None], vector_component], axis=1)
    
    return q_exp

@jit
def compute_quat_inv_vec(q):
    """
    Compute the inverse of a quaternion vector.
    q: (N, 4) array of quaternions.
    """

    num = jnp.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], axis=1)
    
    # Compute the squared norm of each quaternion.
    norm_sq = jnp.sum(q ** 2, axis=1, keepdims=True) + 1e-10
    
    return num / norm_sq

@jit
def compute_quat_prod(q, p):
    q_s, q_v = q[0], q[1:]
    p_s, p_v = p[0], p[1:]

    scalar = q_s * p_s - jnp.dot(q_v, p_v)

    vector = q_s * jnp.array(p_v) + p_s * jnp.array(q_v) + jnp.cross(q_v, p_v)

    return jnp.concatenate([jnp.array([scalar]), vector])

@jit
def compute_quat_prod_vec(q, p):
    q_s, q_v = q[:, 0], q[:, 1:]
    p_s, p_v = p[:, 0], p[:, 1:]

    scalar = q_s * p_s - jnp.sum(q_v * p_v, axis=1)
    scalar = jnp.expand_dims(scalar, axis=1)

    vector = jnp.expand_dims(q_s, axis=1) * jnp.array(p_v) + \
        jnp.expand_dims(p_s, axis=1) * jnp.array(q_v) + jnp.cross(q_v, p_v)

    return jnp.hstack((scalar, vector))

@jit
def compute_log_quat(q):
    """
    Compute the logarithm of a quaternion.
    
    For a unit quaternion q = [q0, q1, q2, q3] (with |q|=1),
      log(q) = [0, theta * (v/||v||)],
    where theta = arccos(q0) and v = [q1, q2, q3]. For ||v||==0, we return zero.
    """
    s = q[0]
    v = q[1:]
    v_norm = jnp.linalg.norm(v)
    factor = jnp.where(v_norm > 0, jnp.arccos(s) / v_norm, 0.0)
    v_factor = v * factor
    return jnp.concatenate([jnp.array([0.0]), v_factor])

@jit
def compute_log_quat_vec(q):
    """
    Compute the logarithm of a batch of quaternions.
    
    Parameters:
      q: jax.numpy array of shape (N, 4)
      
    Returns:
      A jax.numpy array of shape (N, 4) containing the logarithms.
    """
    qv_mag = jnp.linalg.norm(q[:, 1:], axis=1) + 1e-10
    q_mag = jnp.linalg.norm(q, axis=1) + 1e-10
    real_col = jnp.log(q_mag)
    im_angles = jnp.arccos(q[:, 0] / q_mag)
    im_col = jnp.expand_dims(im_angles, axis=1) * q[:, 1:] / jnp.expand_dims(qv_mag, axis=1)
    return jnp.concatenate([real_col[:, None], im_col], axis=1)

@jit
def compute_f_vec(q, tau, omega):
    """
    Vectorizing compute the motion model.
    q: (N, 4) array of quaternions.
    tau: (N,) array of time steps.
    omega: (N, 3) array of angular velocities.
    """

    quat_omega = jnp.zeros_like(q)
    quat_omega = quat_omega.at[:, 1:].set(tau * omega / 2)
    exp_quat_omega = compute_exp_quat_vec(quat_omega)
    f_out = compute_quat_prod_vec(q, exp_quat_omega)

    return f_out

@jit
def compute_h_vec(q, GRAVITY):
    tmp = jnp.zeros_like(q)
    tmp = tmp.at[:, 3].set(-GRAVITY)

    return compute_quat_prod_vec(compute_quat_prod_vec(compute_quat_inv_vec(q), tmp), q)

def compute_f(q, tau, omega):
    quat_omega = jnp.insert(tau * omega / 2, 0, 0)
    exp_quat_omega = compute_exp_quat(quat_omega)
    f_out = compute_quat_prod(q, exp_quat_omega)

    return f_out