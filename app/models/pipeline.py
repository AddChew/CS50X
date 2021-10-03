"""Pipeline module."""
import os
import logging
import numpy as np
import pandas as pd
from typing import Tuple
from zipfile import ZipFile
from models.encoder import Encoder
from models.top2vec import Top2Vec
from config import ZIPPED_PATH, RESULTS_PATH, SUMMARY_PATH


class Pipeline:
    """
        Pipeline
        Passes a text corpus csv file through top2vec model and then saves
        the results and summary to excel files.
        Args
        ----------
        corpus: pd.DataFrame
                input corpus

        feedback: str
                name of the column containing the documents.

        field1: str
                name of the column used to group the documents.

        field2: str
                name of the column used to group the documents.
    """
    def __init__(self, 
                 corpus: pd.DataFrame, 
                 feedback: str, 
                 field1: str, 
                 field2: str, 
                 ):
        
        logger = self.setup_logger()
        encoder = Encoder(logger)

        self.df = corpus
        self.feedback = feedback
        self.field1 = field1
        self.field2 = field2

        self.resultsWriter = pd.ExcelWriter(RESULTS_PATH)
        self.summaryWriter = pd.ExcelWriter(SUMMARY_PATH)

        self.run_pipeline(encoder, logger, RESULTS_PATH, SUMMARY_PATH)
        logger.info(f"Results saved sucessfully to {RESULTS_PATH}.")
        logger.info(f"Summary saved sucessfully to {SUMMARY_PATH}.")

        # Zip the results
        with ZipFile(ZIPPED_PATH, "w") as zipper:
            for file in (RESULTS_PATH, SUMMARY_PATH):
                zipper.write(file, os.path.basename(file))
        logger.info(f"Zipped Top2VecResults saved successfully to {ZIPPED_PATH}.")

        
    def save_no_field_results(self, 
                              encoder: Encoder, 
                              logger: logging.Logger, 
                              resultsPath: str, 
                              summaryPath: str
                              ) -> None:
        """
            Method to write the results and summary to excel files when no grouping field
            is selected.

            Args
            ----------
            encoder: Encoder
                    embedding model to be used for generating the document embeddings.

            logger: logging.Logger
                    logging.Logger object to log messages.

            resultsPath: string
                    file path to save the results to.

            summaryPath: string
                    file path to save the summary to.

            Returns
            ----------
            None
        """
        top2vec = Top2VecPipeline(self.df, self.feedback, encoder, logger)
        results, summary = top2vec.get_results_summary()

        logger.info("Saving top2vec model results.")
        results.to_excel(resultsPath, index = False)
        summary.to_excel(summaryPath, index = False)

    
    def save_one_field_results(self, 
                               encoder: Encoder, 
                               logger: logging.Logger, 
                               field: str, 
                               unique_values: np.ndarray,
                               ) -> None:
        """
            Method to write the results and summary to excel files when one grouping field
            is selected.

            Args
            ----------
            encoder: Encoder
                    embedding model to be used for generating the document embeddings.

            logger: logging.Logger
                    logging.Logger object to log messages.

            field: string
                    name of the column used to group the documents.

            unique_values: np.ndarray
                    unique values in the field column.

            Returns
            ----------
            None
        """
        for i, unique_value in enumerate(unique_values):
            sheet = self.df[self.df[field] == unique_value]
            top2vec = Top2VecPipeline(sheet, self.feedback, encoder, logger)
            results, summary = top2vec.get_results_summary()

            logger.info("Saving top2vec model results.")
            results.to_excel(self.resultsWriter, sheet_name = f"Sheet{i + 1}", index = False)
            summary.to_excel(self.summaryWriter, sheet_name = f"Sheet{i + 1}", index = False)

    
    def save_two_fields_results(self, encoder: Encoder, logger: logging.Logger) -> None:
        """
            Method to write the results and summary to excel files when two grouping fields
            are selected.

            Args
            ----------
            encoder: Encoder
                    embedding model to be used for generating the document embeddings.

            logger: logging.Logger
                    logging.Logger object to log messages.

            Returns
            ----------
            None
        """
        i = 1
        for unique_value1 in self.df[self.field1].unique():
            for unique_value2 in self.df[self.field2].unique():
                sheet = self.df[
                    (self.df[self.field1] == unique_value1) & (self.df[self.field2] == unique_value2)]
                top2vec = Top2VecPipeline(sheet, self.feedback, encoder, logger)
                results, summary = top2vec.get_results_summary()

                logger.info("Saving top2vec model results.")
                results.to_excel(self.resultsWriter, sheet_name = f"Sheet{i}", index = False)
                summary.to_excel(self.summaryWriter, sheet_name = f"Sheet{i}", index = False)
                i += 1
    

    def run_pipeline(self, 
                     encoder: Encoder, 
                     logger: logging.Logger, 
                     resultsPath: str, 
                     summaryPath: str
                     ) -> None:
        """
            Method to pass documents through top2vec pipeline and then save
            the results and summary to excel files.

            Args
            ----------
            encoder: Encoder
                    embedding model to be used for generating the document embeddings.

            logger: logging.Logger
                    logging.Logger object to log messages.

            resultsPath: string
                    file path to save the results to.

            summaryPath: string
                    file path to save the summary to.

            Return
            ----------
            None
        """
        if not self.field1 and not self.field2:
            self.save_no_field_results(encoder, logger, resultsPath, summaryPath)
            return
        
        with self.resultsWriter, self.summaryWriter:
            if self.field1 and self.field2:
                self.save_two_fields_results(encoder, logger)
            
            elif self.field1:
                self.save_one_field_results(
                    encoder, logger, self.field1, self.df[self.field1].unique())

            else:
                self.save_one_field_results(
                    encoder, logger, self.field2, self.df[self.field2].unique())


    @staticmethod
    def setup_logger(name: str = "top2vec") -> logging.Logger:
        """
            Method to setup a logger to log runtime messages.

            Args
            ----------
            name: str (Optional, default 'top2vec')
                    the name assigned to the logger.
            
            Returns
            ----------
            logger: logging.Logger
                    logger for logging runtime messages.
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(sh)

        return logger


class Top2VecPipeline:
    """
        Top2VecPipeline 

        Passes text corpus dataframe through top2vec model and then output the results 
        and summary as dataframes.  

        Args
        ----------
        df: pd.DataFrame
                input text corpus.

        feedback: string
                name of the column containing the documents.

        encoder: Encoder
                embedding model to be used for generating the document embeddings.

        logger: logging.Logger
                logging.Logger object to log messages.
    """
    def __init__(self, df: pd.DataFrame, feedback: str, encoder: Encoder, logger: logging.Logger):

        # Validate feedback column
        if feedback not in df.columns:
            raise ValueError(f"{feedback} column not found in dataframe.")
        
        self.results = df
        self.documents = df[feedback].values.tolist()
        self.run_pipeline(encoder, logger, feedback)
        
    
    def run_pipeline(self, encoder: Encoder, logger: logging.Logger, feedback: str) -> None:
        """
            Method to pass the input text corpus through Top2Vec algorithm and then save
            the results and extractive summaries to dataframes.

            Args
            ----------
            encoder: Encoder
                    embedding model to be used for generating the document embeddings.

            logger: logging.Logger
                    logging.Logger object to log messages.

            feedback: string
                    name of the column containing the documents.

            Returns
            ----------
            None
        """
        try:
            # Initialise top2vec model
            model = Top2Vec(self.documents, encoder, logger)

            # Append model results to dataframe
            self.results["Topic"], self.results["Score"] = model.doc_top, model.doc_dist
            
            # Sort results by topic and score
            self.results.sort_values(
                by = ["Topic", "Score"], ascending = [True, False], inplace = True)

            # Drop duplicates
            lowercase = f"{feedback}_lower"
            summary = self.results.copy()
            summary[lowercase] = summary[feedback].str.replace(r"[^a-zA-Z0-9]", "").str.lower()
            summary = summary.drop_duplicates(subset = [lowercase])
            self.summary = summary.groupby("Topic").head(5).drop(columns = [lowercase])
        
        except ValueError:
            self.summary = self.results


    def get_results_summary(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
            Method to get top2vec model results and extractive summary of each topic.

            Returns
            ----------
            (results, summary): tuple of a pair of pd.DataFrame
                    top2vec model results and extractive summary of each topic.
        """
        return self.results, self.summary