import numpy as np
import os
import plotly.graph_objects as go
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
    cls_accs = []
    epsilons = []
    for critic_first in overall_stats:
        ca = []
        clsa = []
        epsl = []
        critic_first_str = 'FGSM(Critic->Cls)' if critic_first else 'FGSM(Cls->Critic)'
        for eps, stats in overall_stats[critic_first].items():
            ca.append(stats['critic_acc'])
            clsa.append(stats['cls_acc'])
            epsl.append(str(eps))
        critic_accs.append(ca)
        cls_accs.append(clsa)
        epsilons.append(epsl)

    graph_line(epsilons, critic_accs, 'Critic Accuracy vs. Perturbation Budget',
                 'Perturbation Budgets',
                  'Accuracy',
                   traces_names = ['FGSM(Critic->Cls]', 'FGSM(Cls->Critic)']
                   )
    graph_line(epsilons, cls_accs, 'De-Pois Accuracy vs. Perturbation Budget',
                 'Perturbation Budgets',
                  'Accuracy',
                   traces_names = ['FGSM(Critic->Cls]', 'FGSM(Cls->Critic)']
                   )

