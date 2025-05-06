import logging
import sys

lib_logger = logging.getLogger("pydantic_visible_fields")
lib_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s %(name)-25s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
lib_logger.addHandler(handler)
