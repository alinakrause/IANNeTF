import datetime

def config_name():
    """
    Define log paths for training and validation, and create log writers for each.

    Returns:
        train_summary_writer (tf.summary.FileWriter): A log writer for training metrics.
        val_summary_writer (tf.summary.FileWriter): A log writer for validation metrics.
    """
    # Define where to save the log
    config_name= "config_name"
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    train_log_path = f"logs/{config_name}/{current_time}/train"
    val_log_path = f"logs/{config_name}/{current_time}/val"

    # log writer for training metrics
    train_summary_writer = tf.summary.create_file_writer(train_log_path)

    # log writer for validation metrics
    val_summary_writer = tf.summary.create_file_writer(val_log_path)

    return train_summary_writer, val_summary_writer
