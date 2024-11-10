import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def visualize_clusters(X, labels1, labels2):

    # 数据降维到二维
    pca = PCA(n_components=2)
    data_reduced = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    scatter1 = axes[0].scatter(data_reduced[:, 0], data_reduced[:, 1],
                               c=labels1 + 1, cmap='viridis', s=100)
    axes[0].set_title("standard GMM")
    axes[0].set_xlabel('feature X')
    axes[0].set_ylabel('feature Y')

    legend1 = axes[0].legend(*scatter1.legend_elements(), title="Clusters")
    axes[0].add_artist(legend1)
    axes[0].grid()

    scatter2 = axes[1].scatter(data_reduced[:, 0], data_reduced[:, 1],
                               c=labels2 + 1, cmap='viridis', s=100)
    axes[1].set_title("my GMM")
    axes[1].set_xlabel('feature X')
    axes[1].set_ylabel('feature Y')

    legend2 = axes[1].legend(*scatter2.legend_elements(), title="Clusters")
    axes[1].add_artist(legend2)
    axes[1].grid()

    plt.suptitle("Comparison of Two Clustering Results")
    plt.show()
