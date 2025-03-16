import logging
import glob
from utils.llm import LLM
from utils.pdf_extractor import PDFExtractorLLM

logging.basicConfig(level=logging.DEBUG) # logging.DEBUG for verbose
logger = logging.getLogger(__name__)    # create a logger

if __name__ == "__main__":
    logger.info("Getting PDF files")
    pdf_files = glob.glob("data/*.pdf")
    logger.info(f"Found {len(pdf_files)} PDF files")
    logger.info("Extracting text from PDF file")
    # Create a PDFExtractor object
    pdf_extractor = PDFExtractorLLM()
    # Extract the text from the PDF file
    pdf_text = [pdf_extractor.extractor_pdf(path=pdf) for pdf in pdf_files]
    logger.info(f"Extracted {len(pdf_text)} documents from the PDF file")
    # Create a LLM object
    logger.info("Creating embeddings for the extracted text")
    llm = LLM()
    # Create embeddings for the extracted text
    embeddings = [llm.create_embedding(text) for text in pdf_text]
    logger.info(f'Embeddings created:\n{embeddings}')