from .parser import ImageParser, Parser, TextParser
from .pdf_parser import PDFParser

Parsers: dict[str, type[Parser]] = {
    "MinerU": PDFParser,  # type: ignore
}


def get_parser(name: str = "MinerU") -> Parser:
    if name not in Parsers:
        msg = f"unknown parser: {name}" + f"\nsupported parsers are {[k for k in Parsers]}"
        raise Exception(msg)
    p = Parsers[name]
    return p()


def get_parser_by_file_type(file_type: str) -> Parser:
    if file_type in ["pdf"]:
        return PDFParser()
    if file_type in ["txt", "md"]:
        return TextParser()
    if file_type in ["png", "jpg", "jpeg", "bmp", "gif"]:
        return ImageParser()

    # default
    return TextParser()
