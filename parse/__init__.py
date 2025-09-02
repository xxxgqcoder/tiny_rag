from .parser import Parser
from .pdf_parser import PDFParser

Parsers: dict[str, type[Parser]] = {
    "MinerU": PDFParser, # type: ignore
}


def get_parser(name: str = "MinerU") -> Parser:
    if name not in Parsers:
        msg = f"unknown parser: {name}" + f"\nsupported parsers are {[k for k in Parsers]}"
        raise Exception(msg)
    p = Parsers[name]
    return p()
