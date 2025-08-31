from .parser import Parser
from .pdf_parser import PDFParser

Pasers: dict[str, type[PDFParser]] = {
    "MinerU": PDFParser,
}


def get_parser(name: str = "MinerU") -> Parser:
    if name not in Pasers:
        msg = f"unknown parser: {name}" + f"\nsupported parsers are {[k for k in Pasers]}"
        raise Exception(msg)
    p = Pasers[name]
    return p()
