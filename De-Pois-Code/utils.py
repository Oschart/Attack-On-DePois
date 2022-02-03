import numpy as np
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt

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

def graph_stats(overall_stats):
    critic_accs = []
    depois_accs = []
    epsilons = []
    for attack_mode in overall_stats:
        ca = []
        depois_acc = []
        epsl = []

        critic_first_str = attack_mode
        for eps, stats in overall_stats[attack_mode].items():
            ca.append(stats['critic_acc'])
            depois_acc.append(stats['depois_acc'])
            epsl.append(str(eps))
        critic_accs.append(ca)
        depois_accs.append(depois_acc)
        epsilons.append(epsl)

    graph_line(epsilons, critic_accs, 'Critic Accuracy vs. Perturbation Budget',
                 'Perturbation Budgets',
                  'Accuracy',
                   traces_names = ['FGSM(Critic_only)', 'FGSM(Cls_only) (baseline)', 'FGSM(Critic->Cls)', 'FGSM(Cls->Critic)']
                   )
    graph_line(epsilons, depois_accs, 'De-Pois Accuracy vs. Perturbation Budget',
                 'Perturbation Budgets',
                  'Accuracy',
                   traces_names = ['FGSM(Critic_only)', 'FGSM(Cls_only) (baseline)', 'FGSM(Critic->Cls)', 'FGSM(Cls->Critic)']
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