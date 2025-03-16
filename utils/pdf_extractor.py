import logging
from typing import Union, ClassVar
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
    llama_reader : ClassVar[pymupdf4llm.LlamaMarkdownReader] = pymupdf4llm.LlamaMarkdownReader()
    @classmethod
    def extractor_pdf(self,path: str) -> Union[str, None]:
        """
        Extracts the text from a PDF file
        """
        llama_docs = self.llama_reader.load_data(path)
        logger.info(f"Extracted {len(llama_docs)} documents from {path}")
        return llama_docs
