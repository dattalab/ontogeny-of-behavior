import re

_wks_re = re.compile(r'(\d+)wks')


def age_map_male(string: str) -> int:
    '''returns age in weeks'''
    res = _wks_re.search(string)
    if res is not None:
        return int(res.group(1))
    if '3m' in string:
        return 12
    if '6m' in string:
        return 24
    if '9m' in string:
        return 36
    if '12m' in string:
        return 52
    if '18m' in string:
        return 78
    if '22m' in string:
        return 91


def age_map_female(string: str) -> int:
    '''returns age in weeks'''
    if '3m' in string:
        return 12
    if '6m' in string:
        return 24
    if '9m' in string:
        return 36
    if '12m' in string:
        return 52
    if '18m' in string:
        return 72
    if '23m' in string:
        return 92
    if '3w' in string:
        return 3
    if '5w' in string:
        return 5
    if '7w' in string:
        return 7
    if '9w' in string:
        return 9
    if '90w' in string:
        return 90