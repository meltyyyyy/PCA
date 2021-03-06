import matplotlib.pyplot as plt
import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def execute():
    cancer = load_breast_cancer()

    scaler = StandardScaler()
    pca = PCA(n_components=2)

    scaler.fit(cancer.data)
    X_scaled = scaler.transform(cancer.data)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    print("Original shape: {}".format(str(X_scaled.shape)))
    print("Reduced shape: {}".format(str(X_pca.shape)))

    plt.figure(figsize=(8, 8))
    mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
    plt.legend(cancer.target_names, loc='best')
    plt.gca().set_aspect('equal')
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.savefig('pca_dr/pca_dr.png')

    print("PCA component shape: {}".format(pca.components_.shape))
    print("PCA components:\n{}".format(pca.components_))

    plt.figure()
    plt.matshow(pca.components_, cmap='viridis')
    plt.yticks([0, 1], ["First component", "Second component"])
    plt.colorbar()
    plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation=60, ha='left')
    plt.xlabel('Feature')
    plt.ylabel('Principal components')
    plt.savefig('pca_dr/pca_heatmap.png')
