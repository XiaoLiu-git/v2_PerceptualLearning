import matplotlib.pyplot as plt
import numpy as np
import pdb

plt.rcParams['font.sans-serif'] = [
    'Times New Roman']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
plt.rcParams['svg.fonttype'] = 'none'
plt.style.available
plt.style.use('seaborn')


def plot_acc(path, plot_error=False):
    acc = np.load(path)
    mean_acc = np.mean(acc, axis=0) * 100
    var_acc = np.var(acc, axis=0) * 100
    label_index = ["train", "ori1_loc1", "ori2_loc2", "ori1_loc2", "ori2_loc1"]
    plt.figure()
    if plot_error:
        for i in range(acc.shape[-1]):
            plt.plot(np.arange(0, 201, 2), mean_acc[:, i], label=label_index[i],
                     linewidth=4)
            plt.fill_between(np.arange(0, 201, 2), mean_acc[:, i]-var_acc[:, i],
                             mean_acc[:, i]+var_acc[:, i], alpha=0.2)

    else:
        plt.plot(np.arange(0, 201, 2), mean_acc, label=label_index, linewidth=4)
    plt.ylabel("Accuracy %", fontsize=18)
    plt.xlabel("No. epoch", fontsize=18)
    plt.legend(prop={'size': 18})
    plt.show()
    return


def plot_heatmap(path, plot_error=False):
    plt.style.use('default')
    acc = np.load(path)
    mean_acc = np.mean(acc, axis=0)*100
    var_acc = np.var(acc, axis=0)*100
    if plot_error:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14,6))
        # pdb.set_trace()
        im = ax.imshow(mean_acc)
        ax.set(xlabel="orientation $\Delta$ $\\theta$",
               ylabel="location $\Delta$ x")
        ax.set_title("Accuracy %")
        cbar = ax.figure.colorbar(im, ax=ax)

        for i in range(acc.shape[1]):
            for j in range(acc.shape[2]):
                text = ax.text(j,i,int(mean_acc[j,i]), ha="center",
                               va="center")

        im2 =ax2.imshow(var_acc)
        ax2.set(xlabel="orientation $\Delta$ $\\theta$",
               ylabel="location $\Delta$ x")
        ax2.set_title("Accuracy %")
        cbar = ax2.figure.colorbar(im2, ax=ax2)

        for i in range(acc.shape[1]):
            for j in range(acc.shape[2]):
                text = ax2.text(j,i,int(var_acc[j,i]), ha="center",
                               va="center")

    else:
        fig, ax = plt.subplots()
        im = ax.imshow(mean_acc)
        ax.set(xlabel="orientation $\Delta$ $\\theta$",
               ylabel="location $\Delta$ x")
        ax.set_title("Accuracy %")

        for i in range(acc.shape[1]):
            for j in range(acc.shape[2]):
                text = ax.text(j, i, int(mean_acc[i, j]), ha="center",
                               va="center")

        cbar = ax.figure.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.show()
    return


plot_acc('./2022-08-24/New/merg_ori_diss_cc3AccALL.npy', plot_error=True)
plot_heatmap('./2022-08-24/New/merg_ori_diss_cc3Acc_test.npy')
plot_heatmap('./2022-08-24/New/merg_ori_diss_cc3Acc_test_best.npy')


