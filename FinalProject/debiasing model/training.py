import tqdm

from get_sets import get_gender_pairs

def training_loop(model, train_ds, val_ds, args, tokenizer, serialize, train_summary_writer, val_summary_writer):
    """
    Trains a given model on the provided training dataset, and evaluates it on a validation dataset for a given number
    of epochs. Calculates training and validation metrics, and logs them to tensorboard.

    Args:
        model (tensorflow.keras.Model): The model to be trained.
        train_ds (tensorflow.data.Dataset): The training dataset.
        val_ds (tensorflow.data.Dataset): The validation dataset.
        args (argparse.ArgumentParser): ArgumentParser containing all the hyperparamters and necessary arguments.
        tokenizer (tensorflow.keras.preprocessing.text.Tokenizer): The tokenizer of the training data.
        serialize (SerializeEmbedding): Serializer for embedding weights matrix.
        train_summary_writer (tensorflow.summary.SummaryWriter): The log writer for training metrics.
        val_summary_writer (tensorflow.summary.SummaryWriter): The log writer for validation metrics.
    """
    gender_words, D, N = get_gender_pairs(args.path, args.vocabulary_size, tokenizer)

    stored_loss = 100000000
    epochs_since_best_val_set = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}:")

        # Training:
        for data in tqdm.tqdm(train_ds, position=0, leave=True):

            metrics = model.train_step(data, D, N, args.debiasing, args.var_ratio, args.lmbda, args.clip)

            # serialize embeddings
            serialize.on_train_batch_end()

            # logging the validation metrics to the log file which is used by tensorboard
            with train_summary_writer.as_default():
                for metric in model.metrics:
                    tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch)

        # print the metrics
        print([f"{key}: {value.numpy()}" for (key, value) in metrics.items()])

        # reset all metrics (requires a reset_metrics method in the model)
        model.reset_metrics()

        # Validation:
        for data in val_ds:
            metrics = model.test_step(data)

            # logging the validation metrics to the log file which is used by tensorboard
            with val_summary_writer.as_default():
                for metric in model.metrics:
                    tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch)

        print([f"val_{key}: {value.numpy()}" for (key, value) in metrics.items()])

        # reset all metrics
        model.reset_metrics()

        # serialize embeddings
        serialize.on_epoch_end()

        print("\n")

        # check early stopping
        if args.patience > 0:
            if metrics["loss"] < stored_loss:
                stored_loss = metrics["loss"]
                epochs_since_best_val_set = 0
            else:
                epochs_since_best_val_set += 1
                if epochs_since_best_val_set >= args.patience:
                    epoch = epochs


def testing(model, test_ds):
    """
    Runs a testing loop for the given model on the provided test dataset.

    Args:
        model: An tf.keras.Model to be tested.
        test_ds: The dataset on which the model should be tested.
    """
    for data in test_ds:
        metrics = model.test_step(data)

    print([f"test_{key}: {value.numpy()}" for (key, value) in metrics.items()])
