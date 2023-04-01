import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs

def bic_elbow(signal, decomposition_method='PCA', n_components_range=np.arange(2, 11)):
    """
    Perform a BIC elbow test to find the optimal number of components for a signal decomposition method.

    Parameters
    ----------
    signal : `hyperspy.signals.Signal2D` or `hyperspy.signals.SignalND`
        The hyperspy signal to perform the BIC elbow test on.
    decomposition_method : str, optional
        The signal decomposition method to use. Default is 'PCA'.
    n_components_range : array-like, optional
        The range of possible numbers of components to fit the signal model for.
        Default is np.arange(2, 11).

    Returns
    -------
    optimal_n_components : int
        The optimal number of components that minimizes the BIC score.

    Notes
    -----
    This function uses the Bayesian Information Criterion (BIC) to select the optimal number of components
    for the given signal decomposition method. The BIC score is computed for each number of components
    in the range, and the elbow point in the BIC curve is identified as the optimal number of components.
    """
    bic_scores = []
    for n_components in n_components_range:
        # Fit the signal model
        if decomposition_method == 'PCA':
            model = signal.decomposition.pca(n_components=n_components)
        elif decomposition_method == 'NMF':
            model = signal.decomposition.nmf(n_components=n_components)
        elif decomposition_method == 'ICA':
            model = signal.decomposition.ica(n_components=n_components)
        else:
            raise ValueError(f"Unknown decomposition method: {decomposition_method}")

        # Compute the log-likelihood and number of model parameters
        log_likelihood = model.score(signal)
        n_samples, n_features = signal.data.shape
        n_params = n_components * (n_features + 1)

        # Compute the BIC score
        bic = -2 * log_likelihood + n_params * np.log(n_samples)
        bic_scores.append(bic)

    # Plot the BIC scores as a function of the number of components
    plt.plot(n_components_range, bic_scores, 'o-')
    plt.xlabel('Number of components')
    plt.ylabel('BIC score')
    plt.title(f'BIC elbow test for {decomposition_method} decomposition')
    plt.show()

    # Find the elbow point
    diff = np.diff(bic_scores)
    elbow_index = np.argmax(diff) + 1
    optimal_n_components = n_components_range[elbow_index]
    print(f'The optimal number of components is {optimal_n_components}')

    return optimal_n_components

# USAGE EXAMPLE
# signal = hs.load('path/to/signal.hdf5')
# optimal_n_components = bic_elbow(signal)
