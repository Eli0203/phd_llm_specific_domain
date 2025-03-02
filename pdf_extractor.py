import logging
from typing import Union 
from pydantic import BaseModel
import pymupdf4llm

# Set up logging
logging.basicConfig(level=logging.INFO) # logging.DEBUG for verbose
logger = logging.getLogger(__name__)    # create a logger

# Create a class PDFExtractor
class PDFExtractorLLM(BaseModel):
    """
    PDFExtractor class
    """
    path: str
    llama_reader = pymupdf4llm.LlamaMarkdownReader()
    @classmethod
    def extractor_pdf(self):
        """
        Extracts the text from a PDF file
        """
        llama_docs = self.llama_reader.load_data(self.path)
        logger.info(f"Extracted {len(llama_docs)} documents from {self.path}")
        return llama_docs
