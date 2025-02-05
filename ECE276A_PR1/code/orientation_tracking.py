import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import transforms3d
from jax import grad, jit
from load_data import read_data
from panoroma import construct_panorama_v
from plot_code import plt_cost
from quaternion_operation import compute_f, compute_f_vec, compute_h_vec, compute_log_quat_vec, compute_quat_inv_vec, compute_quat_prod_vec
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)


@jit
def IMU_calibration(imu_data):
    """
    Calibrate the IMU data.

    The raw data is assumed to have shape (N, 7) with:
      - Row 0: timestamps (often in microseconds; we convert them to seconds)
      - Rows 1-3: accelerometer data (Ax, Ay, Az)
      - Rows 4-6: gyroscope data (Wx, Wy, Wz)
    """
    # Separate timestamps and sensor measurements.
    # Convert timestamps to seconds (adjust the divisor if your timestamps are in a different unit)
    imu_ts = imu_data[:, 0]
    imu_vals = imu_data[:, 1:]  # Now shape (N, 6)
    imu_calib = jnp.zeros(imu_data.shape, dtype="float64")
    imu_calib = imu_calib.at[:, 0].set(imu_ts)
    # Define scale factors.
    # For example, use the provided scale factor for the accelerometer.
    scale_factor_acc = 3300.0 / 1023.0 / 300
    # For the gyroscope, you may need a different factor. For now, we assume it is already in rad/s.
    scale_factor_gyro = 3300.0 / 1023.0 / 3.33 * jnp.pi / 180
    # Compute the biases using the first 750 time samples (columns), not rows!
    acc_bias = jnp.mean(imu_vals[:750, 0:3], axis=0, keepdims=True)  # shape (3,1)
    gyro_bias = jnp.mean(imu_vals[:750, 3:6], axis=0, keepdims=True)  # shape (3,1)
    # Remove biases and apply scale factors for each sensor type.
    imu_calib = imu_calib.at[:, 1:4].set((imu_vals[:, 0:3] - acc_bias) * scale_factor_acc)
    imu_calib = imu_calib.at[:, 4:7].set((imu_vals[:, 3:6] - gyro_bias) * scale_factor_gyro)
    # Add 1 to z-acceleration to account for gravity.
    imu_calib = imu_calib.at[:, 3].set(imu_calib[:, 3] + 1.0)
    return imu_calib


@jit
def compute_cost_function(q, IMU_data):
    """
    Compute the cost function for the orientation tracking.
    The cost function is based on the difference between the predicted and observed gravity vectors.

    Parameters:
      q: (T-1, 4) array of quaternions.
      IMU_data: (T, 7) array of IMU data.

    Returns:
      cost: Scalar cost value.
    """
    q_0_T = jnp.vstack((jnp.array([1, 0, 0, 0], dtype=jnp.float64), q))
    q_inv = compute_quat_inv_vec(q)
    # Predicted gravity vectors using the observation model.
    h_vec = compute_h_vec(q, GRAVITY=-1)
    # Compute the tau matrix.
    tau = IMU_data[1:, 0:1] - IMU_data[0:-1, 0:1]
    # Compute the omega matrix.
    omega = IMU_data[:-1, 4:7]
    acc = jnp.hstack((jnp.zeros((IMU_data.shape[0] - 1, 1)), IMU_data[1:, 1:4]))
    # Compute the predicted quaternion matrix.
    f = compute_f_vec(q_0_T[:-1, :], tau, omega)
    term1 = 2 * compute_log_quat_vec(compute_quat_prod_vec(q_inv, f))
    cost1 = jnp.sum(jnp.square(term1))
    cost2 = jnp.sum(jnp.square(acc - h_vec))
    return 0.5 * (cost1 + cost2)


# @partial(jit, static_argnames=['itr', 'lr'])
def PGD_optim(IMU_data, itr=100, lr=1e-2):
    q0 = jnp.array([1, 0, 0, 0], dtype=jnp.float64)
    num_imu_ts = IMU_data.shape[0]
    # this stores q1 to qT
    opt_quat_arr = jnp.zeros((num_imu_ts - 1, 4), dtype=jnp.float64)
    opt_quat_arr = opt_quat_arr.at[0].set(compute_f(q0, IMU_data[1, 0] - IMU_data[0, 0], IMU_data[0, 4:7]))
    # Compute quaternion at time t
    for t in jnp.arange(1, num_imu_ts - 1):
        tau_t = IMU_data[t + 1, 0] - IMU_data[t, 0]
        omega_t = IMU_data[t, 4:7]
        q_t = compute_f(opt_quat_arr[t - 1], tau_t, omega_t)
        q_t = q_t / jnp.linalg.norm(q_t)  # Normalize quaternion
        opt_quat_arr = opt_quat_arr.at[t].set(q_t)

    costs = [compute_cost_function(opt_quat_arr, IMU_data)]
    grad_cost_fn = grad(compute_cost_function, argnums=0)

    for _ in tqdm(jnp.arange(itr)):
        grads = grad_cost_fn(opt_quat_arr, IMU_data)
        proj_grad = grads - jnp.sum(grads * opt_quat_arr, axis=1, keepdims=True) * opt_quat_arr
        # opt_quat_arr -= lr * grad
        opt_quat_arr -= lr * proj_grad
        # Project onto the unit quaternion constraint
        opt_quat_arr /= jnp.linalg.norm(opt_quat_arr, axis=1, keepdims=True)
        costs.append(compute_cost_function(opt_quat_arr, IMU_data))

    return opt_quat_arr, costs


def rotation_verification(imu_data):
    """
    Compute the orientation (quaternions) from the IMU data using the motion model:
      q_{t+1} = q_t  ⊗ exp([0, dt * ω / 2])
    Also, convert the Vicon rotation matrices to quaternions.
    """
    imu_ts = imu_data[:, 0]
    imu_vals = imu_data[:, 1:]
    # Initialize quaternion (using [w, x, y, z] ordering)
    q = jnp.array([1, 0, 0, 0])
    quaternions = [q]

    # Integrate the angular velocity.
    for i in range(1, len(imu_ts)):
        dt = imu_ts[i] - imu_ts[i - 1]
        # Extract the gyroscope measurement (rows 3-5) at time step i-1.
        w = imu_vals[i - 1, 3:6]  # (Wx, Wy, Wz)
        q = compute_f(q, dt, w)
        quaternions.append(q.copy())

    quaternions = jnp.array(quaternions)

    return quaternions


def plot_comparison(imu_quaternions, vicon_data, dataset):
    """
    Convert quaternions to Euler angles (roll, pitch, yaw) and plot each axis' rotation.

    Parameters:
      imu_quaternions: jnp.ndarray of shape (N, 4) containing [w, x, y, z] quaternions.
      vicon_quaternions: jnp.ndarray of shape (N, 3, 3) containing rotation matrices.
    """
    # Convert each quaternion to Euler angles using the 'sxyz' convention.
    # This returns (roll, pitch, yaw) for each quaternion.

    imu_euler = jnp.array([transforms3d.euler.quat2euler(q, axes="sxyz") for q in imu_quaternions], copy=False)
    vicon_euler = jnp.array([transforms3d.euler.mat2euler(r, axes="sxyz") for r in vicon_data])

    # imu_euler and vicon_euler now have shape (N, 3), where:
    #   column 0: roll  (rotation about x-axis)
    #   column 1: pitch (rotation about y-axis)
    #   column 2: yaw   (rotation about z-axis)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    axis_labels = ["Roll (x-axis)", "Pitch (y-axis)", "Yaw (z-axis)"]
    rots_title = ["row", "pitch", "yaw"]
    for i in range(3):
        axs[i].plot(imu_euler[:, i], label=f"IMU {axis_labels[i]}")
        axs[i].plot(vicon_euler[:, i], label=f"Vicon {axis_labels[i]}")
        axs[i].set_title(axis_labels[i])
        axs[i].set_xlabel("Sample Index")
        axs[i].set_ylabel("Angle (radians)")
        axs[i].legend()
        axs[i].grid(linestyle="--")
        axs[i].set_title(f"True {rots_title[i]} vs estimates {rots_title[i]} in dataset {dataset}", fontsize=16)

    plt.tight_layout()
    plt.savefig(f"comp_{dataset}.png")
    plt.show()


def plot_acceleration(imu_data_calib):
    """
    Plot the acceleration data from the calibrated IMU data.

    Parameters:
      imu_data_calib: Calibrated IMU data of shape (N, 7).
    """
    timestamps = imu_data_calib[:, 0]
    acc_data = imu_data_calib[:, 1:4]

    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    axis_labels = ["Ax (m/s^2)", "Ay (m/s^2)", "Az (m/s^2)"]

    for i in range(3):
        axs[i].plot(timestamps, acc_data[:, i], label=axis_labels[i])
        axs[i].set_title(axis_labels[i])
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Acceleration (m/s^2)")
        axs[i].legend()

    plt.tight_layout()
    plt.show()


def main():
    for dataset in [10, 11]:
        # train_prefix = "trainset"
        train_prefix = "testset"
        prefix = train_prefix
        cfile = f"data/{prefix}/cam/cam{dataset}.p"
        ifile = f"data/{prefix}/imu/imuRaw{dataset}.p"
        vfile = f"data/{train_prefix}/vicon/viconRot{dataset}.p"

        # Load data.
        imu_data = read_data(ifile).T  # Transpose the IMU data.
        vicon_data = read_data(vfile)
        cam_data = read_data(cfile)  # Load camera data
        vicon_rots = np.rollaxis(vicon_data["rots"], 2)  # Swap axes to get shape (N, 3, 3).

        # Calibrate IMU data.
        imu_data_calib = IMU_calibration(imu_data)
        # Get the quaternion estimates (and convert Vicon rotation matrices).
        # imu_quaternions = rotation_verification(imu_data_calib)

        # Plot the comparison.
        # plot_comparison(imu_quaternions, vicon_rots)

        # Plot the acceleration data.
        # plot_acceleration(imu_data_calib)

        # Projected Gradient Descent Training
        opt_quaternion, costs = PGD_optim(imu_data_calib, itr=500, lr=5e-3)
        plt_cost(costs, dataset)

        # Plot the comparison.
        # plot_comparison(opt_quaternion, vicon_rots, dataset)

        orientations = []
        # Convert each VICON rotation matrix to a quaternion.
        for i in tqdm(range(vicon_rots.shape[0])):
            R = vicon_rots[i, :, :]
            q = transforms3d.quaternions.mat2quat(R)  # returns [w, x, y, z]
            orientations.append(q)
        orientations = np.array(orientations)

        # Visualize all camera images projected onto a 3D sphere.
        # fig, _ = visualize_cams_on_sphere(cam_data, orientations,
        #                         hor_fov=np.radians(60),
        #                         ver_fov=np.radians(45),
        #                         sample_rate=5)

        # input("Press Enter to close the visualization...")
        # plt.close(fig)  # Close the figure after displaying.

        # Construct a panorama using the optimized quaternion.
        panorama = construct_panorama_v(cam_data, imu_data_calib, opt_quaternion, pano_width=2000, pano_height=500, use_vicon=False, blending=False)
        # # Display the panorama.
        plt.figure(figsize=(12, 4))
        plt.imshow(panorama)
        plt.title(f"Panorama of dataset {dataset} using optimized quaternion.")
        plt.axis("off")
        plt.savefig(f"pano_{dataset}_estimates.png")
        plt.show()

        panorama = construct_panorama_v(cam_data, imu_data_calib, opt_quaternion, pano_width=2000, pano_height=500, use_vicon=True, vicon_orientations=orientations, vicon_ts=vicon_data["ts"])
        # Display the panorama.
        plt.figure(figsize=(12, 4))
        plt.imshow(panorama)
        plt.title(f"Panorama of dataset {dataset} using VICON data.")
        plt.axis("off")
        plt.savefig(f"pano_{dataset}_vicon.png")
        plt.show()


if __name__ == "__main__":
    main()
