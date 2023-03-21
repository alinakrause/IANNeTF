import pprint
import tqdm
from get_sets import get_gender_pairs

def training_loop(model, train_ds, val_ds, arguments, train_summary_writer, val_summary_writer):
    """
    Trains a given model on the provided training dataset, and evaluates it on a validation dataset for a given number
    of epochs. Calculates training and validation metrics, and logs them to tensorboard.

    Args:
        model (tensorflow.keras.Model): The model to be trained.
        train_ds (tensorflow.data.Dataset): The training dataset.
        val_ds (tensorflow.data.Dataset): The validation dataset.
        args (argparse.ArgumentParser): ArgumentParser containing all the hyperparamters.
        train_summary_writer (tensorflow.summary.SummaryWriter): The log writer for training metrics.
        val_summary_writer (tensorflow.summary.SummaryWriter): The log writer for validation metrics.
    """
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}:")

        gender_words, D, N, eos_idx = get_gender_pairs(args.vocabulary_size)

        # Training:
        hidden = model.initialize_state(args.train_bsz)
        cell = model.initialize_state(args.train_bsz)
        for data in tqdm.tqdm(train_ds, position=0, leave=True):

            hidden = [tf.stop_gradient(h) for h in hidden]

            metrics, hidden, cell = model.train_step(data, hidden, cell, D, N, args)

            # logging the validation metrics to the log file which is used by tensorboard
            with train_summary_writer.as_default():
                for metric in model.metrics:
                    tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch)

        # print the metrics
        print([f"{key}: {value.numpy()}" for (key, value) in metrics.items()])

        # reset all metrics (requires a reset_metrics method in the model)
        model.reset_metrics()

        # Validation:
        hidden = model.initialize_states(args.val_bsz)
        cell = model.initialize_states(args.val_bsz)
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
