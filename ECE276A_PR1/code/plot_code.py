import matplotlib.pyplot as plt


def plt_rpy(vic_arr, vic_ts, imu_arr, imu_ts, dataset_idx, optim = False):
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('Dataset ' + str(dataset_idx), fontsize=16)

    plt.subplot(3, 1, 1)
    plt.plot(vic_ts, vic_arr[:, 0], label="True Roll")
    plt.plot(imu_ts, imu_arr[:, 0], label="Estimated Roll")
    plt.grid(linestyle='--')
    plt.title("True Roll vs Estimated Roll in degrees")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(vic_ts, vic_arr[:, 1], label="True Pitch")
    plt.plot(imu_ts, imu_arr[:, 1], label="Estimated Pitch")
    plt.grid(linestyle='--')
    plt.title("True Pitch vs Estimated Pitch in degrees")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(vic_ts, vic_arr[:, 2], label="True Yaw")
    plt.plot(imu_ts, imu_arr[:, 2], label="Estimated Yaw")
    plt.grid(linestyle='--')
    plt.title("True Yaw vs Estimated Yaw in degrees")
    plt.legend()
    if optim:
        plt.savefig("../picture/dataset_" + str(dataset_idx) + "/dataset" + str(dataset_idx) + "_optim.png")
    else:
        plt.savefig("../picture/dataset_" + str(dataset_idx) + "/dataset" + str(dataset_idx) + ".png")
    # plt.show()
    
    # plt.close()


def plot_rpy_no_vicon(opt_rpy_arr, imu_ts, dataset_idx, optim = False):
    fig = plt.figure(figsize=(10, 10))

    fig.suptitle('Dataset ' + str(dataset_idx), fontsize=16)
    plt.subplot(3, 1, 1)
    plt.plot(imu_ts, opt_rpy_arr[:, 0])
    plt.grid(linestyle='--')
    plt.title("Estimated Optimized Roll Angle (in degrees)")

    plt.subplot(3, 1, 2)
    plt.plot(imu_ts, opt_rpy_arr[:, 1])
    plt.grid(linestyle='--')
    plt.title("Estimated Optimized Pitch Angle (in degrees)")

    plt.subplot(3, 1, 3)
    plt.plot(imu_ts, opt_rpy_arr[:, 2])
    plt.grid(linestyle='--')
    plt.title("Estimated Optimized Yaw Angle (in degrees)")
    if optim:
        plt.savefig("../picture/dataset_" + str(dataset_idx) + "/dataset" + str(dataset_idx) + "_optim_no_vicon.png")
    else:
        plt.savefig("../picture/dataset_" + str(dataset_idx) + "/dataset" + str(dataset_idx) + "_no_vicon.png")
    # plt.show()
    # plt.close()


def plt_cost(cost_lst, dataset_idx):
    fig = plt.figure(figsize=(10, 7))
    plt.plot(cost_lst)
    plt.grid(linestyle='--')
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Cost c(q1:T)", fontsize=14)
    plt.title(f"Cost Function for Dataset {dataset_idx}.", fontsize=16)
    plt.savefig(f"cost_{dataset_idx}.png")
    plt.show()
    # plt.close()