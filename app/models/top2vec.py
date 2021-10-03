"""Top2Vec module."""
import umap
import hdbscan
import logging
import numpy as np
import pandas as pd
from typing import Tuple
from models.encoder import Encoder
from sklearn.cluster import dbscan
from sklearn.preprocessing import normalize


class Top2Vec:
  """
      Top2Vec

      Creates jointly embedded topic and document vectors.

      Args
      ----------
      documents: list of strings or np.ndarray of np.str
              input text corpus.

      encoder: Encoder
              embedding model to be used for generating the document embeddings.

      logger: logging.Logger
              logging.Logger object to log messages.

      umap_args: dict (Optional, default None)
              pass custom arguments to UMAP.

      hdbscan_args: dict (Optional, default None)
              pass custom arguments to HDBSCAN.

      batch_size: int (Optional, default 32)
              number of documents passed to the model per iteration.
  """
  def __init__(self, 
               documents: list, 
               encoder: Encoder, 
               logger: logging.Logger, 
               umap_args: dict = None, 
               hdbscan_args: dict = None, 
               batch_size: int = 32
               ):

    # Validate documents
    if not isinstance(documents, list):
      raise TypeError("documents need to be a list of strings.")

    if not all(isinstance(document, str) for document in documents):
      raise TypeError("documents need to be a list of strings.")

    # Validate encoder
    if not isinstance(encoder, Encoder):
      raise TypeError("encoder needs to be an instance of an Encoder object.")

    # Validate logger
    if not isinstance(logger, logging.Logger):
      raise TypeError("logger needs to be an instance of a logging.Logger object.")

    self.documents = documents
    self.document_vectors = self.embed_documents(encoder, batch_size)

    # Apply UMAP dimensionality reduction to the document vectors
    logger.info("Creating lower dimension embedding of documents.")

    if not umap_args:
      umap_args = {
          "n_neighbors": 15,
          "n_components": 5,
          "metric": "cosine"
      }

    umap_model = umap.UMAP(**umap_args).fit(self.document_vectors)

    # Find dense areas of document vectors
    logger.info("Finding dense areas of documents.")

    if not hdbscan_args:
      hdbscan_args = {
          "min_cluster_size": 15,
          "metric": "euclidean",
          "cluster_selection_method": "eom"
      }

    cluster = hdbscan.HDBSCAN(**hdbscan_args).fit(umap_model.embedding_)

    # Calculate topic vectors from dense areas of documents
    logger.info("Finding topics.")

    # Create topic vectors
    self.create_topic_vectors(cluster.labels_)

    # Deduplicate topics
    self.deduplicate_topics()
    
    # Assign topic to documents
    self.doc_top, self.doc_dist = self.calculate_documents_topic(batch_size)

    # Calculate topic sizes
    self.topic_sizes = self.calculate_topic_sizes()

    # Re-order topics
    self.reorder_topics()


  def embed_documents(self, encoder: Encoder, batch_size: int = 64) -> np.ndarray:
    """
        Method to embed the input text corpus.

        Args
        ----------
        encoder: Encoder
              embedding model to be used for generating the document embeddings.

        batch_size: int (Optional, default 64)
                number of documents passed to the model per iteration.

        Returns
        ----------
        embeddings: np.ndarray
                numerical vector representation of the input text corpus.    
    """
    document_vectors = []
    for start_index in range(0, len(self.documents), batch_size):
      document_vectors.append(
        encoder.encode(self.documents[start_index: start_index + batch_size]))
      
    return self.l2_normalize(np.array(np.vstack(document_vectors)))

  
  @staticmethod
  def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """
        Method to scale input vectors individually to unit l2 norm (vector length).

        Args
        ----------
        vectors: np.ndarray
                the data to normalize.

        Returns
        ----------
        normalized vectors: np.ndarray
                normalized input vectors.
    """
    if vectors.ndim == 2:
      return normalize(vectors)
    return normalize(vectors.reshape(1, -1))[0]

  
  def create_topic_vectors(self, cluster_labels: np.ndarray) -> None:
    """
        Method to calculate the topic vectors based on the arithmetic mean of all the 
        document vectors in the same dense cluster.

        Args
        ----------
        cluster_labels: np.ndarray
                cluster assigned to each document based on HDBSCAN algorithm.

        Returns
        ----------
        None
    """
    unique_labels = set(cluster_labels)
    if -1 in unique_labels:
      unique_labels.remove(-1)

    self.topic_vectors = self.l2_normalize(
        np.vstack([self.document_vectors[np.where(cluster_labels == label)[0]]
                   .mean(axis = 0) for label in unique_labels]))
    

  def deduplicate_topics(self) -> None:
    """
        Method to merge duplicate topics.

        Returns
        ----------
        None
    """
    _, labels = dbscan(X = self.topic_vectors,
                       eps = 0.1,
                       min_samples = 2,
                       metric = "cosine")
    
    duplicate_clusters = set(labels)

    if len(duplicate_clusters) > 1 or -1 not in duplicate_clusters:

      # Unique topics
      unique_topics = self.topic_vectors[np.where(labels == -1)[0]]

      if -1 in duplicate_clusters:
        duplicate_clusters.remove(-1)

      # Merge duplicate topics
      for unique_label in duplicate_clusters:
        unique_topics = np.vstack(
            [unique_topics, self.l2_normalize(self.topic_vectors[np.where(labels == unique_label)[0]]
                                              .mean(axis = 0))])
        
      self.topic_vectors = unique_topics


  def calculate_documents_topic(self, batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """
        Method to compute the topic and score of each document.

        Args
        ----------
        batch_size: int (Optional, default 64)
                number of documents passed to the model per iteration.

        Returns
        ----------
        (document_topics, document_scores): tuple of a pair of np.ndarray
                the topic assigned to and score of each document. 
    """
    doc_top, doc_dist = [], []
    for start_index in range(0, len(self.documents), batch_size):
      res = np.inner(self.document_vectors[start_index: start_index + batch_size], 
                     self.topic_vectors)
      doc_top.extend(np.argmax(res, axis = 1))
      doc_dist.extend(np.max(res, axis = 1))
    
    return np.array(doc_top), np.array(doc_dist)

  
  def calculate_topic_sizes(self) -> pd.Series:
    """
        Method to calculate the topic sizes.

        Returns
        ----------
        topic_sizes: pd.Series
                number of documents belonging to each topic.
    """
    return pd.Series(self.doc_top).value_counts()


  def reorder_topics(self) -> None:
    """
        Method to sort the topics in descending order based on topic size.

        Returns
        ----------
        None
    """
    self.topic_vectors = self.topic_vectors[self.topic_sizes.index]
    old2new = dict(zip(self.topic_sizes.index, range(self.topic_sizes.index.shape[0])))
    self.doc_top = np.array([old2new[i] for i in self.doc_top])
    self.topic_sizes.reset_index(drop=True, inplace=True)