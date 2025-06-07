from .pdf_parser import PDFParser
from .parser import Parser

Pasers = {
    'MinerU': PDFParser,
}


def get_parser(name: str = "MinerU") -> Parser:
    if name not in Pasers:
        msg = f"unknown parser: {name}" + "\n" \
            f"supported parsers are {[k for k in Pasers]}"
        raise Exception(msg)
    p = Pasers[name]
    return p()
