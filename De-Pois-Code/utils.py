import numpy as np
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt

attack_box_map = {
    "bb": 'Black-Box',
    "wb": 'White-Box',
}

def preprocess_data(x_train, x_test):
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_test = (x_test.astype(np.float32) - 127.5) / 127.5
    return x_train, x_test

def graph_line(x, y, title, xaxis_title, yaxis_title, traces_names):
    os.makedirs('graphs', exist_ok=True)
    fig = go.Figure()
    for i, trace_name in enumerate(traces_names):
        fig.add_trace(go.Scatter(x=x[0], y=y[i],
                                 name=trace_name))
    fig.update_layout(
        title_text=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        title_x=0.5)
    fig.write_image(f"graphs/{title}.png")

def extract_attack_group(attack_group):
    critic_acc = []
    depois_acc = []
    epsilons = []

    for eps, stats in attack_group.items():
        critic_acc.append(stats['critic_acc'])
        depois_acc.append(stats['depois_acc'])
        epsilons.append(str(eps))

    return critic_acc, depois_acc, epsilons

def graph_stats(overall_stats, attack_box):
    attack_box_str = attack_box_map[attack_box]
    critic_acc_all = []
    depois_acc_all = []
    epsilons_all = []
    attack_modes = ['CR_only', 'CL_only', 'CR_then_CL', 'CL_then_CR']
    for attack_mode in attack_modes:
        critic_acc, depois_acc, epsilons = extract_attack_group(overall_stats[attack_mode])

        critic_acc_all.append(critic_acc)
        depois_acc_all.append(depois_acc)
        epsilons_all.append(epsilons)

    graph_line(epsilons_all, critic_acc_all, f'Critic Accuracy vs. Perturbation Budget ({attack_box_str})',
                 'Perturbation Budgets',
                  'Accuracy',
                   traces_names = ['FGSM(Critic_only)', 'FGSM(Classifier_only) (baseline)', 'FGSM(Critic->Classifier)', 'FGSM(Classifier->Critic)']
                   )
    graph_line(epsilons_all, depois_acc_all, f'De-Pois Accuracy vs. Perturbation Budget ({attack_box_str})',
                 'Perturbation Budgets',
                  'Accuracy',
                   traces_names = ['FGSM(Critic_only)', 'FGSM(Classifier_only) (baseline)', 'FGSM(Critic->Classifier)', 'FGSM(Classifier->Critic)']
                   )


def get_best_attack_mode(attack_stats):
    auc_min = 1e10
    best_attack_mode = None
    for attack_mode in attack_stats:
        auc = 0.0
        for eps, stats in attack_stats[attack_mode].items():
            auc += stats['depois_acc']
        if auc < auc_min:
            auc_min = auc
            best_attack_mode = attack_mode
    
    return best_attack_mode


def graph_wb_vs_bb(overall_stats):
    stats_bb = overall_stats["bb"]
    stats_wb = overall_stats["wb"]

    best_mode_wb = get_best_attack_mode(stats_wb)


    critic_accs_bb, depois_accs_bb, epsilons = extract_attack_group(stats_bb[best_mode_wb])
    critic_accs_wb, depois_accs_wb, epsilons = extract_attack_group(stats_wb[best_mode_wb])

    epsilons_all = [epsilons, epsilons]
    critic_acc_all = [critic_accs_wb, critic_accs_bb]
    depois_acc_all = [depois_accs_wb, depois_accs_bb]


    graph_line(epsilons_all, critic_acc_all, f'Critic Accuracy vs. Perturbation Budget (attacking {best_mode_wb})',
                'Perturbation Budgets',
                'Accuracy',
                traces_names = ['White-Box', 'Black-Box']
                )

    graph_line(epsilons_all, depois_acc_all, f'De-Pois Accuracy vs. Perturbation Budget (attacking {best_mode_wb})',
                'Perturbation Budgets',
                'Accuracy',
                traces_names = ['White-Box', 'Black-Box']
                )


def vis_predictions(x_eval, y_pred, n_val):
    rows, cols = 4, 4

    fig,ax = plt.subplots(nrows = rows, ncols = cols)

    ids = np.random.randint(0,n_val,rows*cols)
    for i in range(cols):   
        for j in range(rows):
            ax[j][i].set_title('predicted label: {0}'. format(y_pred[ids[(i*rows)+j]]))
            two_d = (np.reshape(x_eval[ids[(i*rows)+j]], (28, 28)))
            ax[j][i].imshow(two_d, cmap='gray')
            ax[j][i].axes.get_xaxis().set_visible(False)
            ax[j][i].axes.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.show()
    #plt.savefig()