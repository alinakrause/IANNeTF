import pickle

class SerializeEmbedding():
    """A class that provides functions for serializing the embeddings weights after 500 batches and at the end of each epoch.

    Arguments:
        embedding (EncoderDecoder): An instance of the EncoderDecoder class.

    Methods:
        on_train_batch_end: Updates the batch counter and serializes the embedding weights matrix after every 500 batches.
        on_epoch_end: Serializes embedding weights matrix after each training epoch.
    """

    def __init__(self, embedding):
        """
        Initializes an instance of the class with the provided embedding layer and a batch counter variable.

        Args:
            embedding (EncoderDecoder): An instance of the EncoderDecoder class.
        """
        super().__init__()

        self.emb = embedding
        self.batch_counter = 0


    def on_train_batch_end(self):
        """Updates the batch counter and serializes the embedding weights matrix after every 500 batches."""
        self.batch_counter += 1
        if self.batch_counter == 500:
            self.batch_counter = 0

            # get word embeddings matrix
            word_embeddings = self.emb.weights[0]

            # serialize
            emb_file = open(os.path.join(path, "word_embedding_{bias}".format(bias = "debiased" if debiasing else "biased")), 'wb')
            pickle.dump(word_embeddings, emb_file)
            emb_file.close()


    def on_epoch_end(self):
        """Serializes embedding weights matrix after each training epoch."""

        # get word embeddings matrix
        word_embeddings = self.emb.weights[0]

        # serialize
        emb_file = open(os.path.join(path, "word_embedding_{bias}".format(bias = "debiased" if debiasing else "biased")), 'wb')
        pickle.dump(word_embeddings, emb_file)
        emb_file.close()
