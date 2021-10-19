from decouple import config


def get_alphabet(name="basic"):
    """Create specified alphabet

     Args:
         name: string. Name of the alphabet to create.

    Returns:
        alphabet: list. List containing all characters from specified alphabet.
    """
    if name == "basic":
        alphabet = basic()
    elif name == "basic-lower":
        alphabet = basic_lower()
    elif name == "ascii":
        alphabet = ascii()
    elif name == "ascii-lower":
        alphabet = ascii_lower()
    else:
        raise Exception(f"alphabet name {name} unknown")

    return alphabet


def basic():
    """Get basic alphabet for English character set
    with leading special char to encode partial decryptions"""
    alphabet = "".join([chr(i) for i in range(65, 123) if i not in range(91, 97)])
    alphabet = chr(config("SPECIAL_CHAR", cast=int)) + alphabet
    return alphabet


def basic_lower():
    """Get basic alphabet for lowercase English character set
    with leading special char to encode partial decryptions"""

    alphabet = "".join([chr(i) for i in range(97, 123)])
    alphabet = chr(config("SPECIAL_CHAR", cast=int)) + alphabet
    return alphabet


def ascii():
    """Get alphabet for ASCI character set
    with leading special char to encode partial decryptions"""

    alphabet = "".join([chr(i) for i in range(128) if chr(i) != " "])
    alphabet = chr(config("SPECIAL_CHAR", cast=int)) + alphabet
    return alphabet


def ascii_lower():
    """Get alphabet for lowercase ASCI character set
    with leading special char to encode partial decryptions"""

    alphabet = "".join(
        [chr(i) for i in range(128) if i not in range(65, 91) and chr(i) != " "]
    )
    alphabet = chr(config("SPECIAL_CHAR", cast=int)) + alphabet
    return alphabet
