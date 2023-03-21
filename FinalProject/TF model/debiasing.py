def bias_regularization_encoder(model, D, N, var_ratio, lmbda, norm=True):
    """
    Compute bias regularization loss term as described in the Bordia paper for the encoder of a given model.

    Args:
        model: A Keras model object with an encoder.
        D: A tensor of shape (number of gender pairs, 2) containing gendered words pairs.
        N: A tensor of shape (number og gender-neutral words, 1) containing gender-neutral words.
        var_ratio (float): A float between 0 and 1 that determines how much gender variance to capture.
        lmbda (float): A float that determines the strength of the regularization.
        norm (bool): A boolean that indicates whether to normalize the weights before computing the bias regularization loss.

    Returns:
        A tensor representing the bias regularization loss term for the encoder.
    """
    # W: weights matrix of embedding
    # shape: (vocabulary size, embedding features)
    W = model.encoder.get_weights()[0]
    if norm:
        W = W / tf.norm(W, axis=1, keepdims=True)

    # C: differences btw gendered words (gender pairs)
    # shape: (number of gender pairs, embedding length)
    C = []
    for idx in range(D.shape[0]):
        idxs = tf.reshape(D[idx], [-1])
        u = W[idxs[0],:] # vector for female word
        v = W[idxs[1],:] # vector for male counterpart
        C.append(tf.reshape((u - v)/2, [1, -1])) # compute difference
    C = tf.concat(C, axis=0)

    # get principal components
    # Singular Value Decomposition: decomposes a matrix into three parts
    # returns: left singular vectors, singular values & right singular vectors of the input tensor
    S, U, V = tf.linalg.svd(C, full_matrices=True)
    # shape of S: (min(number of gender pairs, embedding length)) -> 1D vector

    # Find k such that we capture 100*var_ratio% of the gender variance
    # in our case k is always 0
    var = S**2
    norm_var = var/ tf.reduce_sum(var)
    cumul_norm_var = tf.math.cumsum(norm_var, axis=0)
    k_idx = tf.math.argmin(cumul_norm_var[cumul_norm_var >= var_ratio])

    # gender subspace B
    B = V[:, :k_idx.numpy()+1] # all rows, first k columns
    loss = tf.pow(tf.norm(tensor=tf.matmul(tf.gather(W, N-1), B), ord=2), 2)

    return lmbda * loss
