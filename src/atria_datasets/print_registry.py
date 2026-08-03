from atria_core.logger import get_logger

from atria_datasets.registry import datasets

logger = get_logger(__name__)

if __name__ == "__main__":
    logger.info(f"Dataset registry loaded successfully:\n{datasets.list()}")
