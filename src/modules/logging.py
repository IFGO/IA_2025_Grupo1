import logging
from pathlib import Path


logs_dir = Path(__file__).resolve().parents[2] / 'logs'
logs_dir.mkdir(exist_ok=True)


def get_logger(progname: str) -> logging.Logger:
    """
    Cria um logger para o programa.
    Todos os logs são salvos no diretório logs/.

    Args:
        progname (str): Nome do programa.

    Returns:
        logging.Logger: Logger configurado.
    """
    logger = logging.getLogger(progname)

    # Evita adicionar outro handler caso o logger esteja sendo reaproveitado
    if not logger.handlers:
        log_file_path = logs_dir / f'{progname}.log'
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')

        handler = logging.FileHandler(log_file_path, mode='a')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
