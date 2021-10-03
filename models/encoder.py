"""Encoder module."""
import logging
import numpy as np
from onnxruntime import InferenceSession as Session
from models.tokenizer import BertTokenizer
from config import MODEL_PATH, VOCAB_PATH


class Encoder:
  """
      Encoder 

      Embeds documents into numerical vectors.

      Args
      ----------
      logger: logging.Logger
              logging.Logger object to log messages.
  """
  def __init__(self, logger: logging.Logger):
 
    # Validate logger
    if not isinstance(logger, logging.Logger):
      raise TypeError("logger needs to be an instance of a logging.Logger object.")

    self.model, self.tokenizer = self.load_model(logger)
    
    if self.model:
      logger.info(f"Loaded {MODEL_PATH} model successfully.") 

    else:
      raise ValueError(
        f"Failed to load {MODEL_PATH} model from local file directory.")


  def load_model(self, logger: logging.Logger):
    """
        Method to load a pretrained model from local file directory.

        Args
        ----------
        logger: logging.Logger
                logging.Logger object to log messages.        

        Returns
        ----------
        model: a onnx model
    """
    model = None

    # Try to load model from local file directory
    try:   
      model = Session(MODEL_PATH)
      tokenizer = BertTokenizer(VOCAB_PATH)

    except OSError:
      logger.error(f"Failed to load model {MODEL_PATH} from local file directory.")

    return model, tokenizer


  def encode(self, documents: list) -> np.ndarray:
    """
        Method to compute document embeddings.

        Args
        ----------
        documents: list of strings
                list/array of string documents to be embedded into numerical vectors.

        Returns
        ----------
        embeddings: np.ndarray
                numerical vector representation of the input documents. 
    """
    encoded_inputs = self.tokenizer.batch_encode_plus(documents)
    return self.model.run(None, encoded_inputs)[0]