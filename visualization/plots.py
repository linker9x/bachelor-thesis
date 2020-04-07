import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import manifold


def plots(name, filepath):
    df = pd.read_csv(filepath)
    df_features = df.drop('class', axis=1)
    feature_names = list(df_features.columns.values)

    df['label'] = df['class'].apply(lambda i: str(i))

    # # selected features
    # df_feat = df[feature_names]
    # # array of selected features
    # data_subset = df_feat.values

    # selected features and class / labels
    X = df_features
    X['class'] = df['class']
    X['label'] = df['label']

    num_of_classes = df['class'].nunique()

    # tsne = manifold.TSNE(n_components=2, verbose=1, n_iter=10000, learning_rate=10, random_state=1)
    # lle = manifold.LocallyLinearEmbedding(n_neighbors=10, n_components=2, eigen_solver='auto', method='standard')
    # isomap = manifold.Isomap(n_neighbors=10, n_components=2)
    # mds = manifold.MDS(n_components=2, max_iter=100, n_init=1)
    # se = manifold.SpectralEmbedding(n_components=2, n_neighbors=10)
    # pca_tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    #pca = PCA(n_components=2)
    tsne = manifold.TSNE(n_components=2, verbose=1, n_iter=10000, perplexity=5, learning_rate=10, random_state=1)

    # embedding = [tsne, lle, isomap, mds, se, pca_tsne, pca]
    # emb_names = ['tsne', 'lle', 'isomap', 'mds', 'se', 'pca_tsne', 'pca']

    #embedding = [tsne, pca]
    #emb_names = ['tsne', 'pca']

    embedding = [tsne]
    emb_names = ['tsne']

    for i in range(4):
        for e_name, e in zip(emb_names, embedding):
            X_copy = X
            results = e.fit_transform(X_copy)
            label_x = name + ' ' + e_name + ' p=5 x-axis'
            label_y = name + ' ' + e_name + ' p=5 y-axis'

            X_copy[label_x] = results[:, 0]
            X_copy[label_y] = results[:, 1]

            plt.figure(figsize=(8, 5))

            sns.scatterplot(
                x=label_x, y=label_y,
                hue="class",
                palette=sns.color_palette("hls", num_of_classes),
                data=X_copy,
                legend="full",
                alpha=0.7
            )
            plt.savefig('./images/' + name + '_' + e_name + '_' + str(i) + '_2D.png', bbox_inches='tight')

    # if(threeD):
    #     tsne = manifold.TSNE(n_components=3, verbose=1, n_iter=10000, learning_rate=10, random_state=1)
    #     lle = manifold.LocallyLinearEmbedding(n_neighbors=10, n_components=3, eigen_solver='auto', method='standard')
    #     isomap = manifold.Isomap(n_neighbors=10, n_components=3)
    #     mds = manifold.MDS(n_components=3, max_iter=100, n_init=1)
    #     se = manifold.SpectralEmbedding(n_components=3, n_neighbors=10)
    #     pca_tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
    #     pca = PCA(n_components=3)
    #
    #     embedding = [tsne, lle, isomap, mds, se, pca_tsne, pca]
    #     emb_names = ['tsne', 'lle', 'isomap', 'mds', 'se', 'pca_tsne', 'pca']
    #     for e_name, e in zip(emb_names, embedding):
    #         X_copy = X
    #         results = e.fit_transform(X_copy)
    #
    #         X_copy['one'] = results[:, 0]
    #         X_copy['two'] = results[:, 1]
    #         X_copy['three'] = results[:, 2]
    #
    #         ax = plt.figure(figsize=(8, 5)).gca(projection='3d')
    #         ax.scatter(
    #             xs=X_copy["one"],
    #             ys=X_copy["two"],
    #             zs=X_copy["three"],
    #             c=X_copy["class"]
    #             #     cmap=['blue', 'red', 'green']
    #         )
    #         ax.set_xlabel(name + ' ' + e_name + ' x-axis')
    #         ax.set_ylabel(name + ' ' + e_name + ' y-axis')
    #         ax.set_zlabel(name + ' ' + e_name + ' z-axis')
    #         plt.savefig('./images/' + name + '_' + e_name + '_3D.png', bbox_inches='tight')


def post_plots(name, filepath, selected_feats):
    df = pd.read_csv(filepath)
    df_features = df[selected_feats]
	
    feature_names = list(df_features.columns.values)

    df['label'] = df['class'].apply(lambda i: str(i))

    # # selected features
    # df_feat = df[feature_names]
    # # array of selected features
    # data_subset = df_feat.values

    # selected features and class / labels
    X = df_features
    X['class'] = df['class']
    X['label'] = df['label']

    num_of_classes = df['class'].nunique()

    # tsne = manifold.TSNE(n_components=2, verbose=1, n_iter=10000, learning_rate=10, random_state=1)
    # lle = manifold.LocallyLinearEmbedding(n_neighbors=10, n_components=2, eigen_solver='auto', method='standard')
    # isomap = manifold.Isomap(n_neighbors=10, n_components=2)
    # mds = manifold.MDS(n_components=2, max_iter=100, n_init=1)
    # se = manifold.SpectralEmbedding(n_components=2, n_neighbors=10)
    # pca_tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    #pca = PCA(n_components=2)
    tsne = manifold.TSNE(n_components=2, verbose=1, n_iter=10000, perplexity=5, learning_rate=10, random_state=1)

    # embedding = [tsne, lle, isomap, mds, se, pca_tsne, pca]
    # emb_names = ['tsne', 'lle', 'isomap', 'mds', 'se', 'pca_tsne', 'pca']

    #embedding = [tsne, pca]
    #emb_names = ['tsne', 'pca']

    embedding = [tsne]
    emb_names = ['tsne']

    for i in range(4):
        for e_name, e in zip(emb_names, embedding):
            X_copy = X
            results = e.fit_transform(X_copy)
            label_x = name + ' ' + e_name + ' p=5 x-axis'
            label_y = name + ' ' + e_name + ' p=5 y-axis'

            X_copy[label_x] = results[:, 0]
            X_copy[label_y] = results[:, 1]

            plt.figure(figsize=(8, 5))

            sns.scatterplot(
                x=label_x, y=label_y,
                hue="class",
                palette=sns.color_palette("hls", num_of_classes),
                data=X_copy,
                legend="full",
                alpha=0.7
            )
            plt.savefig('./images/' + name + '_' + e_name + '_' + str(i) + '_post_2D.png', bbox_inches='tight')

    # if(threeD):
    #     tsne = manifold.TSNE(n_components=3, verbose=1, n_iter=10000, learning_rate=10, random_state=1)
    #     lle = manifold.LocallyLinearEmbedding(n_neighbors=10, n_components=3, eigen_solver='auto', method='standard')
    #     isomap = manifold.Isomap(n_neighbors=10, n_components=3)
    #     mds = manifold.MDS(n_components=3, max_iter=100, n_init=1)
    #     se = manifold.SpectralEmbedding(n_components=3, n_neighbors=10)
    #     pca_tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
    #     pca = PCA(n_components=3)
    #
    #     embedding = [tsne, lle, isomap, mds, se, pca_tsne, pca]
    #     emb_names = ['tsne', 'lle', 'isomap', 'mds', 'se', 'pca_tsne', 'pca']
    #     for e_name, e in zip(emb_names, embedding):
    #         X_copy = X
    #         results = e.fit_transform(X_copy)
    #
    #         X_copy['one'] = results[:, 0]
    #         X_copy['two'] = results[:, 1]
    #         X_copy['three'] = results[:, 2]
    #
    #         ax = plt.figure(figsize=(8, 5)).gca(projection='3d')
    #         ax.scatter(
    #             xs=X_copy["one"],
    #             ys=X_copy["two"],
    #             zs=X_copy["three"],
    #             c=X_copy["class"]
    #             #     cmap=['blue', 'red', 'green']
    #         )
    #         ax.set_xlabel(name + ' ' + e_name + ' x-axis')
    #         ax.set_ylabel(name + ' ' + e_name + ' y-axis')
    #         ax.set_zlabel(name + ' ' + e_name + ' z-axis')
    #         plt.savefig('./images/' + name + '_' + e_name + '_3D.png', bbox_inches='tight')



def top_plots(ds_name, methods, position):
    if len(methods) > 8:
        print('Not enough colors')
        return None
    act_intervals = [5, 10, 15, 20, 25, 35, 45, 55, 65, 75, 125, 150, 175, 200, 250]
    per_intervals = [.02, .04, .06, .08, .1, .15, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
    markerfaces = ['#ed89a2', '#8eafe2', '#f9c377', '#c78be5',
                   '#96d396', '#524c4c', '#4753de', '#805630']
    lines = ['#ed0e45', '#4286f4', '#ef9210', '#9411d6',
             '#349633', '#000000', '#0512a3', '#613105']

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(121)
    ax1.set_ylim([0, 1.05])
    ax1.grid(b=True, which='major', color='#666666', linestyle='-')
	
    ax2 = fig.add_subplot(122)
    ax2.set_ylim([0, 1.05])
    ax2.grid(b=True, which='major', color='#666666', linestyle='-')

    for i, alg in enumerate(methods):
        data = pd.read_csv('./results/' + ds_name + '/' + alg + '_NN_avg.csv')
        f1 = data['f1'].values
        max_pos = position[i]

        if alg in ['JACKSTRAW', 'CFS', 'EN', 'HSIC', 'LASSO', 'SS']:
            ax1.plot(per_intervals, f1, linewidth=3, marker='o', markerfacecolor=markerfaces[i],
                     markersize=10, color=lines[i], label=alg, markevery=[max_pos])
            print()
        else:
            ax2.plot(act_intervals, f1, linewidth=3, marker='o', markerfacecolor=markerfaces[i],
                 markersize=10, color=lines[i], label=alg, markevery=[max_pos])


    ax1.set_title('Subset/Ranked Subset Methods')
    ax1.set(xlabel="Fraction of Subset", ylabel="fold-averaged F1 Score")
    ax1.legend()

    ax2.set_title('Ranked Methods')
    ax2.set(xlabel="Number of Features", ylabel="fold-averaged F1 Score")
    ax2.legend()

    fig.suptitle(ds_name + ' Top ' + str(len(methods)) + ' FSMs')

    if not os.path.exists('./results/' + ds_name + '/' + 'images/'):
        os.mkdir('./results/' + ds_name + '/' + 'images/')

    plt.savefig('./results/' + ds_name + '/' + 'images/' + ds_name + '_top_f1.png', bbox_inches='tight')