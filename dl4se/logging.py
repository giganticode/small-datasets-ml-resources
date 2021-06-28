import logging
from rich.logging import RichHandler, get_console

logging.basicConfig(
    level="NOTSET", format='%(message)s', datefmt="[%X]", handlers=[RichHandler()]
)

console = get_console()
logger = logging.getLogger('dl4se')
