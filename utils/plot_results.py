import numpy as np
import json
from os import listdir
from os.path import isfile, join
import collections
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 22})
from matplotlib.ticker import FormatStrFormatter


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def create_fed_test(folders, metrics=['iou', 'ece', 'nll'], class_=str(2), out_path='../imgs/', dataset='peso'):
    # Get File
    folders = [join(folder, dataset) for folder in folders]
    experiment_files_all = [[(f, folder) for f in listdir(folder) if isfile(join(folder, f))] for folder in folders]
    exp_names = ['Ours', 'FedAvgM']
    out_path = out_path + str(dataset) + '.pdf'
    m_names = ['IOU (↑)', 'ECE (↓)', 'NLL (↓)']

    bottle, fedavgm = [], []
    for f in experiment_files_all[0]:
        bottle.append(f)

    for f in experiment_files_all[1]:
        fedavgm.append(f)

    experiment_files_all = [bottle, fedavgm]
    fig, axs = plt.subplots(1, 3, figsize=(16, 7))
    plt.subplots_adjust(bottom=0.28)
    plt.legend()
    plt.subplots_adjust(wspace=0.4, hspace=0.2)
    for m_id, m in enumerate(metrics):
        for exp_id, experiment_files in enumerate(experiment_files_all):
            experiment_dicts = []
            for dict_file in experiment_files:
                dict_name = dict_file[0].split('.')[0]
                with open(join(dict_file[1], dict_file[0])) as f:
                    experiment_dicts.append((dict_name, json.load(f)))

            result_list_dict = {}
            for exp_dict in experiment_dicts:
                exp_name = exp_dict[0].split('_')
                if exp_name[0] not in ['0', '1']:
                    continue
                round = int(exp_name[-1])
                key = round

                if key in result_list_dict:
                    result_list_dict[key].append(float(exp_dict[1][m][class_]))
                else:
                    result_list_dict[key] = [float(exp_dict[1][m][class_])]

            result_dict_mean = {}
            result_dict_std = {}
            for key in result_list_dict.keys():
                new_key = key
                result_dict_mean[new_key] = np.mean(np.array(result_list_dict[key]))
                result_dict_std[new_key] = np.std(np.array(result_list_dict[key]))

            result_dict_mean = collections.OrderedDict(sorted(result_dict_mean.items()))
            colors = [[(0 / 255, 131 / 255, 204 / 255), (233 / 255, 80 / 255, 62 / 255)],
                      [(252 / 255, 202 / 255, 0 / 255), (0 / 255, 156 / 255, 218 / 255)],
                      [(230 / 255, 0 / 255, 26 / 255), (128 / 255, 69 / 255, 151 / 255)],
                      [(153 / 255, 192 / 255, 0 / 255), (221 / 255, 223 / 255, 72 / 255)],
                      [(166 / 255, 0 / 255, 132 / 255), (80 / 255, 182 / 255, 149 / 255)],
                      [(80 / 255, 182 / 255, 149 / 255)]]

            def moving_average(x, w):
                return np.convolve(np.pad(x, pad_width=w // 2, mode='edge'), np.ones(w), 'valid') / w

            axs[m_id].plot(list(result_dict_mean.keys()), moving_average(np.array(list(result_dict_mean.values())), 7),
                           color=colors[exp_id][0], linewidth=2, markersize=0, label=exp_names[exp_id], linestyle='-',
                           marker='o')

            print(list(result_dict_mean.values())[-1], m, exp_names[exp_id])

            if m == 'iou':
                lims = (0.4, 0.65)
            elif m == 'ece':
                lims = (0.005, 0.025)
            else:
                lims = (0.18, 0.35)

            axs[m_id].set_ylim(lims)
            axs[m_id].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            axs[m_id].yaxis.set_major_locator(plt.MaxNLocator(7))

            if exp_id == 0:
                axs[m_id].set_xlabel(m_names[m_id], fontweight='bold')
            else:
                pass
                # axs[m_id, 1].set_yticks([])
                # axs[m_id, 2].set_yticks([])
    handles, labels = axs[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2)

    plt.show()
    plt.savefig(out_path, bbox_inches='tight', dpi=1200)


if __name__ == '__main__':
    create_fed_test(["../experiments/federated/results/fed_bottle", "../experiments/federated/results/fed_avgm"])