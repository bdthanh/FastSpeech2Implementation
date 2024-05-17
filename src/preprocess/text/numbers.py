""" Copied and modified from https://github.com/keithito/tacotron """

import inflect
import re

_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")

def _remove_commas(m):
    """Removes commas from between digits, e.g. 15,000 -> 15000"""
    return m.group(1).replace(",", "")


def _expand_decimal_point(m):
    """Expands decimals, e.g. 42.57 -> forty-two point fifty-seven"""
    return m.group(1).replace(".", " point ")


def _expand_pounds(m):
    """Expands pounds, e.g. £42.57 -> forty-two pounds fifty-seven pence"""
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " pounds"  # Unexpected format
    pounds = int(parts[0]) if parts[0] else 0
    pence = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if pounds and pence:
        pound_unit = "pound" if pounds == 1 else "pounds"
        pence_unit = "penny" if pence == 1 else "pence"
        return "%s %s, %s %s" % (pounds, pound_unit, pence, pence_unit)
    elif pounds:
        pound_unit = "pound" if pounds == 1 else "pounds"
        return "%s %s" % (pounds, pound_unit)
    elif pence:
        pence_unit = "penny" if pence == 1 else "pence"
        return "%s %s" % (pence, pence_unit)
    else:
        return "zero pounds"


def _expand_dollars(m):
    """Expands dollars, e.g. $42.57 -> forty-two dollars fifty-seven cents"""
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    else:
        return "zero dollars"


def _expand_ordinal(m):
    """Expands ordinals, e.g. 1st -> first, 2nd -> second, etc."""
    return _inflect.number_to_words(m.group(0))


def _expand_number(m):
    """Converts an integer number into words"""
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + " hundred"
        else:
            return _inflect.number_to_words(
                num, andword="", zero="oh", group=2
            ).replace(", ", " ")
    else:
        return _inflect.number_to_words(num, andword="")


def normalize_numbers(text):
    """Normalize numbers in text data"""
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, _expand_pounds, text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text
