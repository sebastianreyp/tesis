from clean_text import clean
import re
from unicodedata import normalize

s = "hólaéá"

s = re.sub(
        r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1", 
        normalize( "NFD", s), 0, re.I
    )

# -> NFC
s = normalize( 'NFC', s)

print( s )


# print(clean(s))