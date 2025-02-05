import numpy as np
import transforms3d
from tqdm import tqdm
import matplotlib.pyplot as plt

def spherical_projection(image: np.ndarray,
                         R: np.ndarray,
                         hor_fov: float = np.radians(60),
                         ver_fov: float = np.radians(45),
                         sample_rate: int = 5):
    
    H, W, _ = image.shape
    cx, cy = W / 2, H / 2

    # Compute focal lengths from FOV.
    f_x = (W / 2) / np.tan(hor_fov / 2)
    f_y = (H / 2) / np.tan(ver_fov / 2)

    # Create a grid of pixel coordinates with subsampling.
    u = np.arange(0, W, sample_rate)
    v = np.arange(0, H, sample_rate)
    uu, vv = np.meshgrid(u, v)  # Each of shape (H_sample, W_sample)
    
    # Convert pixel coordinates to normalized camera coordinates.
    # (x, y, 1) with x = (u - cx)/f_x, y = (v - cy)/f_y.
    x = (uu - cx) / f_x
    y = (vv - cy) / f_y
    z = np.ones_like(x)
    dirs = np.stack((x, y, z), axis=-1)  # Shape (H_sample, W_sample, 3)
    norms = np.linalg.norm(dirs, axis=2, keepdims=True)
    dirs = dirs / norms  # Normalize to unit vectors.

    # Rotate the direction vectors into world coordinates.
    dirs_flat = dirs.reshape(-1, 3)  # (N, 3)
    world_dirs = (R @ dirs_flat.T).T  # (N, 3)
    
    # Get the corresponding colors from the image.
    # Note: use the same subsampled indices.
    colors = image[vv, uu, :]  # Shape (H_sample, W_sample, 3)
    colors_flat = colors.reshape(-1, 3)
    # Normalize colors if image is uint8.
    if image.dtype == np.uint8:
        colors_flat = colors_flat / 255.0

    return world_dirs, colors_flat

def visualize_image_on_sphere(image: np.ndarray,
                              R: np.ndarray,
                              hor_fov: float = np.radians(60),
                              ver_fov: float = np.radians(45),
                              sample_rate: int = 5) -> (plt.Figure, plt.Axes):
    """
    Projects a single RGB image onto a unit sphere and visualizes the result in 3D.
    
    The function computes the normalized ray directions for the pixels (using the computed
    focal lengths from hor_fov and ver_fov), rotates them using the given rotation matrix R,
    and then displays the points on a 3D scatter plot colored by the pixel colors.
    
    Parameters:
      image: NumPy array of shape (H, W, 3) representing the input RGB image.
      R: 3x3 rotation matrix (from a quaternion) that rotates the camera frame into world coordinates.
      hor_fov: Horizontal field-of-view (in radians); default is 60°.
      ver_fov: Vertical field-of-view (in radians); default is 45°.
      sample_rate: Subsampling rate along each dimension (default=5) to speed up plotting.
    """
    H, W, _ = image.shape
    cx, cy = W / 2, H / 2

    # Compute focal lengths from FOV.
    f_x = (W / 2) / np.tan(hor_fov / 2)
    f_y = (H / 2) / np.tan(ver_fov / 2)

    # Create a grid of pixel coordinates with subsampling.
    u = np.arange(0, W, sample_rate)
    v = np.arange(0, H, sample_rate)
    uu, vv = np.meshgrid(u, v)  # Each of shape (H_sample, W_sample)
    
    # Convert pixel coordinates to normalized camera coordinates.
    # (x, y, 1) with x = (u - cx)/f_x, y = (v - cy)/f_y.
    x = (uu - cx) / f_x
    y = (vv - cy) / f_y
    z = np.ones_like(x)
    dirs = np.stack((x, y, z), axis=-1)  # Shape (H_sample, W_sample, 3)
    norms = np.linalg.norm(dirs, axis=2, keepdims=True)
    dirs = dirs / norms  # Normalize to unit vectors.

    # Rotate the direction vectors into world coordinates.
    dirs_flat = dirs.reshape(-1, 3)  # (N, 3)
    world_dirs = (R @ dirs_flat.T).T  # (N, 3)
    
    # Get the corresponding colors from the image.
    # Note: use the same subsampled indices.
    colors = image[vv, uu, :]  # Shape (H_sample, W_sample, 3)
    colors_flat = colors.reshape(-1, 3)
    # Normalize colors if image is uint8.
    if image.dtype == np.uint8:
        colors_flat = colors_flat / 255.0

    # Plot the 3D scatter.
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(world_dirs[:, 0], world_dirs[:, 1], world_dirs[:, 2],
                    c=colors_flat, s=1, marker='.')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.axes.set_xlim3d(left=-1, right=1)
    ax.axes.set_ylim3d(bottom=-1, top=1) 
    ax.axes.set_zlim3d(bottom=-1, top=1) 
    ax.set_title("3D Projection of Image on Unit Sphere")
    # Set equal aspect ratio for a proper sphere view.
    ax.set_box_aspect([1, 1, 1])
    fig.canvas.draw_idle()
    plt.show(block=False)
    plt.pause(0.1)
    
    return fig, ax

def visualize_cam_on_sphere(cam_data: dict,
                            orientation: np.ndarray,
                            hor_fov: float = np.radians(60),
                            ver_fov: float = np.radians(45),
                            sample_rate: int = 5) -> None:
    """
    Visualize one camera image (from cam_data["cam"]) on a 3D sphere.
    
    This function selects a camera image (for example, the first one) and uses the corresponding
    orientation (given as a quaternion in [w,x,y,z] order) to compute a rotation matrix.
    Then, it calls visualize_image_on_sphere to project the image onto a unit sphere and display it.
    
    Parameters:
      cam_data: Dictionary with key 'cam' that is a NumPy array of shape (H, W, 3, T)
                containing the RGB images.
      orientation: A quaternion (4-element array) corresponding to the chosen time stamp.
      hor_fov: Horizontal field-of-view (in radians).
      ver_fov: Vertical field-of-view (in radians).
      sample_rate: Subsampling rate for visualization.
    """
    # Extract the first image (or choose another index as desired).
    images = cam_data["cam"]  # Expected shape: (H, W, 3, T)
    # For example, choose the first image.
    image = images[..., 0]  # Shape (H, W, 3)
    
    # Convert the provided quaternion to a rotation matrix.
    R = transforms3d.quaternions.quat2mat(np.array(orientation))
    
    # Visualize the projection.
    fig, _ = visualize_image_on_sphere(image, R, hor_fov, ver_fov, sample_rate)
    input("Press Enter to close the visualization...")

    plt.close(fig)  # Close the figure after displaying.

def visualize_cams_on_sphere(cam_data: dict,
                            orientation: np.ndarray,
                            hor_fov: float = np.radians(60),
                            ver_fov: float = np.radians(45),
                            sample_rate: int = 5) -> (plt.Figure, plt.Axes):
    """
    Visualize one camera image (from cam_data["cam"]) on a 3D sphere.
    
    This function selects a camera image (for example, the first one) and uses the corresponding
    orientation (given as a quaternion in [w,x,y,z] order) to compute a rotation matrix.
    Then, it calls visualize_image_on_sphere to project the image onto a unit sphere and display it.
    
    Parameters:
      cam_data: Dictionary with key 'cam' that is a NumPy array of shape (H, W, 3, T)
                containing the RGB images.
      orientation: A quaternion (4-element array) corresponding to the chosen time stamp.
      hor_fov: Horizontal field-of-view (in radians).
      ver_fov: Vertical field-of-view (in radians).
      sample_rate: Subsampling rate for visualization.
    """
    # Extract the first image (or choose another index as desired).
    images = np.rollaxis(cam_data["cam"], 3, 0)  # Expected shape: (T, H, W, 3)
    projected_pixels = []
    colors = []
    for image, quat in tqdm(zip(images[::80], orientation[::80])):
        # Convert the provided quaternion to a rotation matrix.
        R = transforms3d.quaternions.quat2mat(np.array(quat))
        projected_pixel, color = spherical_projection(image, R, hor_fov, ver_fov, sample_rate)
        projected_pixels.append(projected_pixel)
        colors.append(color)
    
    world_dirs = np.concatenate(projected_pixels, axis=0)
    colors_flat = np.concatenate(colors, axis=0)

    # Plot the 3D scatter.
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(world_dirs[:, 0], world_dirs[:, 1], world_dirs[:, 2],
                    c=colors_flat, s=1, marker='.')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.axes.set_xlim3d(left=-1, right=1)
    ax.axes.set_ylim3d(bottom=-1, top=1) 
    ax.axes.set_zlim3d(bottom=-1, top=1) 
    ax.set_title("3D Projection of Image on Unit Sphere")
    # Set equal aspect ratio for a proper sphere view.
    ax.set_box_aspect([1, 1, 1])
    fig.canvas.draw_idle()
    plt.show(block=False)
    plt.pause(0.1)
    
    return fig, ax

def visualize_all_pixels_on_sphere(cam_data: dict,
                                   orientations: np.ndarray,
                                   hor_fov: float = np.radians(60),
                                   ver_fov: float = np.radians(45)) -> (plt.Figure, plt.Axes):
    """
    Project every pixel of all camera images onto the 3D unit sphere using the pinhole camera model and the given orientations.
    
    Parameters:
      cam_data: Dictionary with key "cam" that is a NumPy array of shape (H, W, 3, T) containing T RGB images.
      orientations: NumPy array of shape (T, 4) containing orientation quaternions (in [w, x, y, z] order) for each image.
      hor_fov: Horizontal field-of-view in radians (default: 60°).
      ver_fov: Vertical field-of-view in radians (default: 45°).
      
    Returns:
      fig, ax: The matplotlib figure and 3D axes objects for interactive visualization.
    """
    # Get the image dimensions.
    images = cam_data["cam"]  # shape: (H, W, 3, T)
    H, W, C, T = images.shape
    
    # Compute the intrinsic parameters (focal lengths) from the FOVs.
    cx, cy = W / 2.0, H / 2.0
    f_x = (W / 2.0) / np.tan(hor_fov / 2.0)
    f_y = (H / 2.0) / np.tan(ver_fov / 2.0)
    
    all_points = []
    all_colors = []
    
    # Process each image.
    for t in tqdm(range(T)):
        # Get the t-th image.
        image = images[..., t]  # shape: (H, W, 3)
        # Get the corresponding orientation (a quaternion in [w, x, y, z] order) and compute its rotation matrix.
        q = orientations[t]
        R = transforms3d.quaternions.quat2mat(np.array(q))
        
        # Create a grid of pixel coordinates.
        u = np.arange(W)
        v = np.arange(H)
        uu, vv = np.meshgrid(u, v)  # each of shape (H, W)
        
        # Convert pixel coordinates to normalized camera coordinates.
        # Using the pinhole model: x = (u - cx)/f_x, y = (v - cy)/f_y, z = 1.
        x = (uu - cx) / f_x
        y = (vv - cy) / f_y
        z = np.ones_like(x)
        dirs = np.stack((x, y, z), axis=-1)  # shape: (H, W, 3)
        
        # Normalize the direction vectors so they lie on the unit sphere.
        norms = np.linalg.norm(dirs, axis=2, keepdims=True)
        dirs = dirs / norms  # Now each pixel has a unit direction in the camera frame.
        
        # Rotate these directions into the world frame using the rotation matrix.
        world_dirs = (R @ dirs.reshape(-1, 3).T).T  # shape: (H*W, 3)
        
        # Append the computed world directions.
        all_points.append(world_dirs)
        
        # Extract and reshape the colors for all pixels.
        colors = image.reshape(-1, 3)
        if image.dtype == np.uint8:
            colors = colors.astype(np.float64) / 255.0
        all_colors.append(colors)
    
    # Concatenate points and colors from all images.
    all_points = np.concatenate(all_points, axis=0)  # shape: (T * H * W, 3)
    all_colors = np.concatenate(all_colors, axis=0)  # shape: (T * H * W, 3)
    
    # Create an interactive 3D scatter plot.
    plt.ion()  # enable interactive mode
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all points with their colors.
    sc = ax.scatter(all_points[::5000, 0], all_points[::5000, 1], all_points[::5000, 2],
                    c=all_colors[::5000], s=0.5, marker='.')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Projection of All Pixels onto 3D Unit Sphere")
    ax.axes.set_xlim3d(left=-1, right=1)
    ax.axes.set_ylim3d(bottom=-1, top=1) 
    ax.axes.set_zlim3d(bottom=-1, top=1) 
    ax.set_box_aspect([1, 1, 1])
    
    # Draw and allow interaction.
    fig.canvas.draw_idle()
    plt.show(block=False)
    plt.pause(0.1)
    
    return fig, ax

def warp_image_to_panorama_v(image: np.ndarray,
                             R: np.ndarray,
                             pano_width: int,
                             pano_height: int,
                             hor_fov: float = np.radians(60),
                             ver_fov: float = np.radians(45)) -> np.ndarray:
    """
    Vectorized warping of a single RGB image to a cylindrical panorama.
    
    Parameters:
      image: NumPy array of shape (H, W, 3) representing the input RGB image.
      R: 3x3 rotation matrix (from a quaternion) that rotates the camera frame into world coordinates.
      pano_width: Width of the output panorama (pixels).
      pano_height: Height of the output panorama (pixels).
      hor_fov: Horizontal field-of-view (radians). (Default: 60°)
      ver_fov: Vertical field-of-view (radians). (Default: 45°)
      
    Returns:
      warped: NumPy array of shape (pano_height, pano_width, 3) containing the warped image.
    """
    H, W, C = image.shape
    cx, cy = W / 2, H / 2

    # Compute focal lengths from fov:
    f_x = (W / 2) / np.tan(hor_fov / 2)
    f_y = (H / 2) / np.tan(ver_fov / 2)

    # Create pixel coordinate grid.
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)  # each of shape (H, W)

    # Convert pixels to normalized camera coordinates.
    x = (uu - cx) / f_x
    y = (vv - cy) / f_y
    z = np.ones_like(x)
    dirs = np.stack((x, y, z), axis=-1)  # (H, W, 3)
    dirs = dirs / np.linalg.norm(dirs, axis=2, keepdims=True)

    # Rotate all directions by R.
    world_dirs = (R @ dirs.reshape(-1, 3).T).T.reshape(H, W, 3)

    # Convert world directions to cylindrical coordinates.
    theta = np.arctan2(world_dirs[..., 0], world_dirs[..., 1])
    h_val = world_dirs[..., 2]

    # Map theta ([-pi, pi]) to panorama horizontal coordinate and
    # h_val (assumed to be roughly in [-1, 1]) to vertical coordinate.
    u_pano = ((theta + np.pi) / (2 * np.pi)) * pano_width
    v_pano = (1 - (h_val + 1) / 2) * pano_height

    u_pano = np.clip(np.round(u_pano).astype(np.int32), 0, pano_width - 1)
    v_pano = np.clip(np.round(v_pano).astype(np.int32), 0, pano_height - 1)

    # Now, scatter the image pixels into the warped panorama using vectorized indexing.
    warped = np.zeros((pano_height, pano_width, C), dtype=image.dtype)
    # Flatten indices and image:
    flat_u = u_pano.flatten()
    flat_v = v_pano.flatten()
    flat_image = image.reshape(-1, C)
    # Use advanced indexing to assign all at once.
    warped[flat_v, flat_u] = flat_image  # if multiple pixels map to the same location, the last wins.
    
    return warped

# def warp_image_to_panorama_v(image: np.ndarray,
#                              R: np.ndarray,
#                              pano_width: int,
#                              pano_height: int,
#                              hor_fov: float = np.radians(60),
#                              ver_fov: float = np.radians(45)) -> np.ndarray:
#     """
#     Vectorized warping of a single RGB image to a cylindrical panorama.
    
#     This version uses the unnormalized pinhole rays computed with a focal length (here taken
#     as the average of f_x and f_y computed from hor_fov and ver_fov) and rotates them into the world
#     frame using R. Then the horizontal coordinate is computed from a swapped arctan2 so that the
#     azimuth is given by arctan2(Z, X) rather than arctan2(X, Z), which helps distribute the images
#     horizontally.
    
#     Parameters:
#       image: NumPy array of shape (H, W, 3) representing the input RGB image.
#       R: 3x3 rotation matrix (from a quaternion) that rotates the camera frame into world coordinates.
#          (Here R is assumed to satisfy: world_dir = R * camera_dir.)
#       pano_width: Width of the output panorama (pixels).
#       pano_height: Height of the output panorama (pixels).
#       hor_fov: Horizontal field-of-view (radians). (Default: 60°)
#       ver_fov: Vertical field-of-view (radians). (Default: 45°)
      
#     Returns:
#       warped: NumPy array of shape (pano_height, pano_width, 3) containing the warped image.
#     """
#     H, W, C = image.shape
#     cx, cy = W / 2.0, H / 2.0

#     # Compute focal lengths from fov.
#     f_x = (W / 2.0) / np.tan(hor_fov / 2.0)
#     f_y = (H / 2.0) / np.tan(ver_fov / 2.0)
#     f_avg = (f_x + f_y) / 2.0  # use as the cylinder's radius

#     # Create pixel coordinate grid.
#     u = np.arange(W)
#     v = np.arange(H)
#     uu, vv = np.meshgrid(u, v)  # each of shape (H, W)

#     # Compute unnormalized rays in the camera frame.
#     # Use d = [u - cx, v - cy, f_avg].
#     X = uu - cx
#     Y = vv - cy
#     Z = f_avg * np.ones_like(X)
#     d = np.stack((X, Y, Z), axis=-1)  # shape (H, W, 3)

#     # Rotate rays into the world frame.
#     d_world = (R @ d.reshape(-1, 3).T ).T.reshape(H, W, 3)
#     # d_world components: [X_world, Y_world, Z_world].

#     # Compute cylindrical coordinates.
#     # Use the swapped arctan2: horizontal angle = arctan2(Z_world, X_world)
#     theta = np.arctan2(d_world[..., 2], d_world[..., 0])
#     # Vertical coordinate: use Y_world directly.
#     v_cyl = d_world[..., 1]

#     # Map theta from [-pi, pi] to horizontal panorama coordinate.
#     u_pano = ((theta + np.pi) / (2 * np.pi)) * pano_width

#     # For vertical coordinate, assume the cylinder's vertical span is given by:
#     # v_min = -f_avg * tan(ver_fov/2) and v_max = f_avg * tan(ver_fov/2).
#     v_min = -f_avg * np.tan(ver_fov / 2.0)
#     v_max =  f_avg * np.tan(ver_fov / 2.0)
#     v_pano = (1 - (v_cyl - v_min) / (v_max - v_min)) * pano_height

#     # Clip and round indices.
#     u_pano = np.clip(np.round(u_pano).astype(np.int32), 0, pano_width - 1)
#     v_pano = np.clip(np.round(v_pano).astype(np.int32), 0, pano_height - 1)

#     # Scatter the image pixels into the warped panorama.
#     warped = np.zeros((pano_height, pano_width, C), dtype=image.dtype)
#     flat_u = u_pano.flatten()
#     flat_v = v_pano.flatten()
#     flat_image = image.reshape(-1, C)
#     warped[flat_v, flat_u] = flat_image  # if multiple pixels map to the same location, the last wins.
    
#     return warped


def construct_panorama_v(cam_data: dict,
                         imu_data: np.ndarray,
                         orientation_estimates: np.ndarray,
                         pano_width: int = 2000,
                         pano_height: int = 500,
                         hor_fov: float = np.radians(60),
                         ver_fov: float = np.radians(45),
                         use_vicon: bool = False,
                         blending: bool = False,
                         vicon_orientations: np.ndarray = None,
                         vicon_ts: np.ndarray = None) -> np.ndarray:
    """
    Construct a panoramic image by stitching together RGB camera images,
    warped using the corresponding body orientation.
    
    For each camera image (stored in cam_data['images'] with shape (H, W, 3, T)),
    the function finds the closest-in-the-past orientation (from IMU_data[1:] timestamps)
    and uses it to compute a rotation matrix to warp the image. The warped images
    are then blended (using a simple overwrite scheme) into a single panorama.
    
    Parameters:
      cam_data: Dictionary with keys:
                - 'ts': 1D NumPy array of camera image timestamps (length T).
                - 'images': NumPy array of shape (H, W, 3, T) containing the RGB images.
      imu_data: Calibrated IMU data as a NumPy array of shape (N, 7); the first column contains timestamps.
                (Orientation estimates correspond to imu_data[1:] timestamps.)
      orientation_estimates: NumPy array of shape (N-1, 4) containing orientation quaternions (q₁ to q_T)
                             from your estimation method.
      pano_width: Desired width of the output panorama (pixels).
      pano_height: Desired height of the output panorama (pixels).
      hor_fov: Horizontal field-of-view (radians) of the camera.
      ver_fov: Vertical field-of-view (radians) of the camera.
      use_vicon: If True, the function will use vicon_orientations as the orientation source.
      vicon_orientations: NumPy array of VICON orientation quaternions (shape (N, 4)); used if use_vicon is True.
      
    Returns:
      panorama: NumPy array of shape (pano_height, pano_width, 3) representing the stitched panoramic image.
    """
    # Ensure orientation timestamps are 1D (from imu_data[1:]).
    
    # Choose orientation source.
    if use_vicon:
        orientation_source = vicon_orientations
        orient_ts = np.squeeze(vicon_ts)
    else:
        orientation_source = orientation_estimates
        imu_np = np.array(imu_data)
        orient_ts = np.squeeze(imu_np[1:, 0])

    # Camera timestamps and images.
    cam_ts = np.squeeze(np.array(cam_data['ts']))  # shape (T,)
    images = cam_data['cam']                     # shape (H, W, 3, T)
    H, W, C, T = images.shape
    # imu-to-camera rotation matrix (from IMU frame to camera frame).
    cam_R_imu = np.array([
        [ 0,  0,  1],
        [-1,  0,  0],
        [ 0, -1,  0]],
        dtype=np.float32
    ).T

    if blending:
        # Preallocate panorama.
        panorama = np.zeros((T, pano_height, pano_width, 3), dtype=np.uint8)

        # For each camera image, find the closest previous orientation and warp the image.
        for t in tqdm(range(0, T, 1)):
            # Initialize a blank canvas for the panorama.
            canvas = np.zeros((pano_height, pano_width, 3), dtype=np.uint8)
            # Get the scalar timestamp.
            cam_time = float(cam_ts[t])
            # Find the index of the last orientation timestamp that is <= cam_time.
            idx = np.searchsorted(orient_ts, cam_time, side='right') - 1
            if idx < 0:
                idx = 0
            elif idx >= len(orient_ts):
                idx = len(orient_ts) - 1
            q = orientation_source[idx]  # orientation quaternion [w, x, y, z]
            # Convert quaternion to a rotation matrix.
            world_R_imu = transforms3d.quaternions.quat2mat(np.array(q))
            world_R_cam = world_R_imu @ cam_R_imu.T
            # Extract the t-th camera image.
            image = images[..., t]  # shape (H, W, 3)
            # Warp the image using the vectorized warp function.
            warped = warp_image_to_panorama_v(image, world_R_cam, pano_width, pano_height, hor_fov, ver_fov)
            # Compute a mask of valid (nonzero) pixels.
            mask = (warped.sum(axis=2) > 0)
            # Vectorized assignment: overwrite panorama pixels where warped is nonzero.
            canvas[mask] = warped[mask]
            panorama[t, :, :] = canvas
        
        # Blending: average the overlapping pixels.
        non_zero_count = np.count_nonzero(np.sum(panorama, axis = 3), axis=0, keepdims=True)
        pano_out = panorama.sum(axis=0) / (non_zero_count[..., np.newaxis] + 1e-10)

        return np.squeeze(pano_out).astype(np.uint8)
    else:
        # Preallocate panorama.
        panorama = np.zeros((pano_height, pano_width, 3), dtype=np.uint8)

        # For each camera image, find the closest previous orientation and warp the image.
        for t in tqdm(range(0, T, 1)):
            # Get the scalar timestamp.
            cam_time = float(cam_ts[t])
            # Find the index of the last orientation timestamp that is <= cam_time.
            idx = np.searchsorted(orient_ts, cam_time, side='right') - 1
            if idx < 0:
                idx = 0
            elif idx >= len(orient_ts):
                idx = len(orient_ts) - 1
            q = orientation_source[idx]  # orientation quaternion [w, x, y, z]
            # Convert quaternion to a rotation matrix.
            world_R_imu = transforms3d.quaternions.quat2mat(np.array(q))
            world_R_cam = world_R_imu @ cam_R_imu.T
            # Extract the t-th camera image.
            image = images[..., t]  # shape (H, W, 3)
            # Warp the image using the vectorized warp function.
            warped = warp_image_to_panorama_v(image, world_R_cam, pano_width, pano_height, hor_fov, ver_fov)
            # Compute a mask of valid (nonzero) pixels.
            mask = (warped.sum(axis=2) > 0)
            # Vectorized assignment: overwrite panorama pixels where warped is nonzero.
            panorama[mask] = warped[mask]

        return panorama
