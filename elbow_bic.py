from .dependencies import (
    np, plt, sns, KMeans, GaussianMixture, StandardScaler, cdist
)

def compute_elbow_bic(data, max_clusters=10, use_gmm=False):
    """
    Compute the elbow and BIC values for KMeans with varying number of clusters,
    and plot the results.

    Parameters
    ----------
    data : array-like or pd.DataFrame, shape (n_samples, n_features)
        Input data to be clustered.
    max_clusters : int, optional, default: 10
        Maximum number of clusters to evaluate.
    use_gmm : bool, optional, default: False
        If True, use Gaussian Mixture Model for BIC computation.
        If False, use log-likelihood and KMeans model for BIC computation.

    Returns
    -------
    results : dict
        A dictionary containing the range of cluster numbers evaluated, 
        distortions, and BIC values for each number of clusters using the chosen model.
    """
    ks = range(1, max_clusters + 1)
    distortions = []
    bics = []
    
    X_scaled = StandardScaler().fit_transform(data)

    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
        distortions.append(kmeans.inertia_)

        if use_gmm:
            # BIC with Gaussian Mixture Model
            gmm = GaussianMixture(n_components=k, random_state=42).fit(X_scaled)
            bic = gmm.bic(X_scaled)
        else:
            # BIC with log-likelihood and KMeans model
            log_likelihood = np.sum(np.log(np.min(cdist(X_scaled, kmeans.cluster_centers_, 'euclidean'), axis=1)))
            n_features = data.shape[1]
            n_samples = data.shape[0]
            bic = -2 * log_likelihood + np.log(n_samples) * n_features * k
        
        bics.append(bic)

    results = {
        'ks': ks,
        'distortions': distortions,
        'bics': bics
    }

    return results

def find_elbow(distortions):
    x1, y1 = 1, distortions[0]
    x2, y2 = len(distortions), distortions[-1]

    distances = []
    for i, d in enumerate(distortions):
        x0 = i + 1
        y0 = d
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = ((y2 - y1)**2 + (x2 - x1)**2)**0.5
        distances.append(numerator / denominator)

    return distances.index(max(distances)) + 1

# Create a sample dataset
data, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# Compute elbow and BIC
ks, distortions, bics = compute_elbow_bic(data, max_clusters=10)


def plot_elbow_bic_seaborn(ks, distortions, bics, optimal_k, title='Elbow and BIC Plot'):
    """
    Plot the elbow and BIC curve using Seaborn library.

    Parameters
    ----------
    ks : list
        The range of cluster numbers evaluated.
    distortions : list
        The distortions (inertia) for each number of clusters.
    bics : list
        The BIC values for each number of clusters.
    optimal_k : int
        The optimal number of clusters.
    title : str, optional, default: 'Elbow and BIC Plot'
        Title of the plot.
    """
    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        'Number of Clusters': ks,
        'Distortion': distortions,
        'BIC': bics
    })

    # Set the Seaborn theme and color palette
    sns.set_theme()
    sns.set_palette("pastel")

    # Create the figure and axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the elbow curve
    sns.lineplot(data=results_df, x='Number of Clusters', y='Distortion', ax=ax1, label='Distortion')
    ax1.set_ylabel('Distortion')

    # Create a secondary axis for BIC
    ax2 = ax1.twinx()
    sns.lineplot(data=results_df, x='Number of Clusters', y='BIC', ax=ax2, color='g', label='BIC')
    ax2.set_ylabel('BIC')

    # Highlight and label the optimal number of clusters
    ax1.axvline(optimal_k, color='r', linestyle='--')
    ax1.text(optimal_k, distortions[optimal_k - 1], f'Optimal k = {optimal_k}', fontsize=12, color='r')

    # Set the title and legend
    plt.title(title)
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    plt.show()