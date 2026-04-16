"""Indian state/UT name-to-code mapping with aliases.

Provides bidirectional lookup between full state names, Hindi names,
common abbreviations, and ISO 3166-2:IN style 2-letter codes used by
Sarvam APIs and Chroma metadata filters.
"""

from __future__ import annotations

_STATE_ENTRIES: list[tuple[str, list[str]]] = [
    # (code, [canonical_name, English aliases, Hindi aliases, Bengali aliases, romanised variants])
    ("AP", ["Andhra Pradesh", "andhra", "AP", "आंध्र प्रदेश", "অন্ধ্রপ্রদেশ", "andhra pradesh"]),
    ("AR", ["Arunachal Pradesh", "arunachal", "AR", "अरुणाचल प्रदेश", "অরুণাচলপ্রদেশ"]),
    ("AS", ["Assam", "AS", "असम", "আসাম", "asom"]),
    ("BR", ["Bihar", "BR", "बिहार", "বিহার"]),
    ("CG", ["Chhattisgarh", "chattisgarh", "CG", "छत्तीसगढ़", "ছত্তিশগড়"]),
    ("GA", ["Goa", "GA", "गोवा", "গোয়া"]),
    ("GJ", ["Gujarat", "gujrat", "GJ", "गुजरात", "গুজরাট"]),
    ("HR", ["Haryana", "HR", "हरियाणा", "হরিয়ানা"]),
    ("HP", ["Himachal Pradesh", "himachal", "HP", "हिमाचल प्रदेश", "হিমাচলপ্রদেশ"]),
    ("JH", ["Jharkhand", "jharkhand", "JH", "झारखंड", "ঝাড়খণ্ড"]),
    ("KA", ["Karnataka", "karnataka", "KA", "कर्नाटक", "কর্ণাটক"]),
    ("KL", ["Kerala", "kerala", "KL", "केरल", "কেরালা", "কেরল"]),
    ("MP", ["Madhya Pradesh", "MP", "मध्य प्रदेश", "মধ্যপ্রদেশ"]),
    ("MH", ["Maharashtra", "maharashtra", "MH", "महाराष्ट्र", "মহারাষ্ট্র"]),
    ("MN", ["Manipur", "MN", "मणिपुर", "মণিপুর"]),
    ("ML", ["Meghalaya", "ML", "मेघालय", "মেঘালয়"]),
    ("MZ", ["Mizoram", "MZ", "मिजोरम", "মিজোরাম"]),
    ("NL", ["Nagaland", "NL", "नागालैंड", "নাগাল্যান্ড"]),
    ("OD", ["Odisha", "orissa", "OD", "OR", "ओडिशा", "ওডিশা"]),
    ("PB", ["Punjab", "PB", "पंजाब", "পাঞ্জাব"]),
    ("RJ", ["Rajasthan", "rajasthan", "RJ", "राजस्थान", "রাজস্থান"]),
    ("SK", ["Sikkim", "SK", "सिक्किम", "সিকিম"]),
    ("TN", ["Tamil Nadu", "tamilnadu", "tamil nadu", "TN", "तमिलनाडु", "তামিলনাড়ু"]),
    ("TS", ["Telangana", "telangana", "TS", "तेलंगाना", "তেলেঙ্গানা"]),
    ("TR", ["Tripura", "TR", "त्रिपुरा", "ত্রিপুরা"]),
    ("UK", ["Uttarakhand", "uttaranchal", "UK", "UA", "उत्तराखंड", "উত্তরাখণ্ড"]),
    ("UP", ["Uttar Pradesh", "UP", "उत्तर प्रदेश", "উত্তরপ্রদেশ"]),
    (
        "WB",
        [
            "West Bengal",
            "bengal",
            "WB",
            "पश्चिम बंगाल",
            "পশ্চিমবঙ্গ",
            "paschimbanga",
            "poshchim bongo",
        ],
    ),
    # Union Territories
    ("AN", ["Andaman and Nicobar Islands", "andaman", "AN", "अंडमान और निकोबार", "আন্দামান ও নিকোবর"]),
    ("CH", ["Chandigarh", "CH", "चंडीगढ़", "চণ্ডীগড়"]),
    (
        "DN",
        [
            "Dadra and Nagar Haveli and Daman and Diu",
            "dadra",
            "daman",
            "DN",
            "DD",
            "दादरा और नगर हवेली",
            "দাদরা ও নগর হাভেলি",
        ],
    ),
    ("DL", ["Delhi", "new delhi", "NCT", "DL", "दिल्ली", "দিল্লি"]),
    ("JK", ["Jammu and Kashmir", "jammu", "kashmir", "J&K", "JK", "जम्मू और कश्मीर", "জম্মু ও কাশ্মীর"]),
    ("LA", ["Ladakh", "LA", "लद्दाख", "লাদাখ"]),
    ("LD", ["Lakshadweep", "LD", "लक्षद्वीप", "লক্ষদ্বীপ"]),
    ("PY", ["Puducherry", "pondicherry", "PY", "पुडुचेरी", "পুদুচেরি"]),
]

# Build lookup tables at import time
_NAME_TO_CODE: dict[str, str] = {}
_CODE_TO_NAME: dict[str, str] = {}

for _code, _names in _STATE_ENTRIES:
    _CODE_TO_NAME[_code] = _names[0]  # canonical name
    for _name in _names:
        _NAME_TO_CODE[_name.lower().strip()] = _code


def state_name_to_code(name: str | None) -> str | None:
    """Convert a state name, abbreviation, or alias to a 2-letter code.

    Returns ``None`` if the name cannot be resolved.

    >>> state_name_to_code("Maharashtra")
    'MH'
    >>> state_name_to_code("RJ")
    'RJ'
    >>> state_name_to_code("tamil nadu")
    'TN'
    """
    if not name:
        return None
    key = name.lower().strip()
    return _NAME_TO_CODE.get(key)


def state_code_to_name(code: str) -> str | None:
    """Convert a 2-letter state code to its canonical name.

    >>> state_code_to_name("WB")
    'West Bengal'
    """
    return _CODE_TO_NAME.get(code.upper().strip())


def normalize_state(raw: str | None) -> str | None:
    """Return the canonical state name for any form of input.

    Returns the input unchanged if no mapping exists.
    """
    if not raw:
        return raw
    code = state_name_to_code(raw)
    if code:
        return _CODE_TO_NAME[code]
    return raw


# Convenience export
INDIAN_STATES = _CODE_TO_NAME
