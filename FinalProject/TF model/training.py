import pprint
import tqdm
from get_sets import get_gender_pairs

def training_loop(model, train_ds, val_ds, epochs, vocabulary_size, var_ratio, lmbda, debiasing, alpha, beta, batch_sizes, train_summary_writer, val_summary_writer):
    """
    Trains a given model on the provided training dataset, and evaluates it on a validation dataset for a given number
    of epochs. Calculates training and validation metrics, and logs them to tensorboard.

    Args:
        model (tensorflow.keras.Model): The model to be trained.
        train_ds (tensorflow.data.Dataset): The training dataset.
        val_ds (tensorflow.data.Dataset): The validation dataset.
        epochs (int): The number of epochs to train the model for.
        vocabulary_size (int): The size of the vocabulary used in the model.
        var_ratio (float): The variance ratio used for the calculation of the gender word pairs.
        lmbda (float): The lambda value used for regularization.
        debiasing (bool): Whether to apply bias regularization (default=True).
        alpha (float): The L2 regularization parameter (default=0).
        beta (float): The slowness regularization parameter (default=0).        batch_sizes (list): List of integers representing the batch sizes for the training, validation and testing dataset.
        train_summary_writer (tensorflow.summary.SummaryWriter): The log writer for training metrics.
        val_summary_writer (tensorflow.summary.SummaryWriter): The log writer for validation metrics.
    """
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}:")

        gender_words, D, N, eos_idx = get_gender_pairs(vocabulary_size)

        # Training:
        hidden = model.initialize_state(batch_sizes[0])
        cell = model.initialize_state(batch_sizes[0])
        for data in tqdm.tqdm(train_ds, position=0, leave=True):

            hidden = [tf.stop_gradient(h) for h in hidden]

            metrics, hidden, cell = model.train_step(data, hidden, cell, D, N, var_ratio, lmbda, debiasing, alpha, beta)

            # logging the validation metrics to the log file which is used by tensorboard
            with train_summary_writer.as_default():
                for metric in model.metrics:
                    tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch)

        # print the metrics
        print([f"{key}: {value.numpy()}" for (key, value) in metrics.items()])

        # reset all metrics (requires a reset_metrics method in the model)
        model.reset_metrics()

        # Validation:
        hidden = model.initialize_states(batch_sizes[1])
        cell = model.initialize_states(batch_sizes[1])
        for data in val_ds:
            metrics, hidden, cell = model.test_step(data, hidden, cell)

            # logging the validation metrics to the log file which is used by tensorboard
            with val_summary_writer.as_default():
                for metric in model.metrics:
                    tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch)

            hidden = [tf.stop_gradient(h) for h in hidden]

        print([f"val_{key}: {value.numpy()}" for (key, value) in metrics.items()])

        # reset all metrics
        model.reset_metrics()

        print("\n")
