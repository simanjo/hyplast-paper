import time
from typing import Tuple
from urllib.error import HTTPError
from urllib.request import urlopen
from urllib.parse import quote
from functools import cache

import pandas as pd


def unify_doi(doi: str, short: bool=False):
    if pd.isna(doi):
        return pd.NA
    if not isinstance(doi, str):
        raise TypeError(f'Wrong tpye for doi, expected string: {doi}')
    *_, publisher, issue = doi.split('/')
    if short:
        return publisher.strip() + '/' + issue.strip()
    else:
        return 'https://doi.org/' + publisher.strip() + '/' + issue.strip()


# Copyright rapelpy CC-BY-SA
# Modified from https://stackoverflow.com/a/54932071
@cache
def get_smiles(name: str, pubchem_only: bool=True) -> str | None:
    name = name.replace('.', ',')
    name = name.replace('α', 'alpha')
    pubchem_url = None
    smiles = None
    if pubchem_only:
        pubchem_url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/' + quote(name) + '/property/CanonicalSMILES/TXT'
    else:
        url = 'https://cactus.nci.nih.gov/chemical/structure/' + quote(name) + '/smiles'
        try:
            smiles = urlopen(url).read().decode('utf8').strip()
            smiles = str(smiles)
        except HTTPError as e:
            print(f'Encountered an error while parsing {name}: {e}')
            print('Trying pubchem for fallback')
            pubchem_url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/' + quote(name) + '/property/CanonicalSMILES/TXT'
    if pubchem_url:
        try:
            smiles = urlopen(pubchem_url).read().decode('utf8').strip()
            smiles = str(smiles)
            # if there are multiple smiles only take the first
            smiles = smiles.split('\n')[0]
        except HTTPError as e:
            print(f'Encountered an HTTP error while parsing {name} in pubchem: {e}')
            if 'HTTP Error 400: PUGREST.BadRequest' in str(e):
                # sleep shortly to throttle requests
                time.sleep(0.2)
                print('Retrying with smiles field')
                pubchem_url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/' + quote(name) + '/property/SMILES/TXT'
                try:
                    smiles = urlopen(pubchem_url).read().decode('utf8').strip()
                    smiles = str(smiles)
                except HTTPError as e:
                    print(f'Could not parse {name} in pubchem: {e}')
                    smiles = None
    # sleep shortly to throttle requests
    time.sleep(0.2)
    return smiles


def unify_sorbent(name: str, base_only=False) -> str:
    if name is None or not isinstance(name, str):
        raise TypeError('Please provide a string as argument.', name)
    name = name.lower()

    modified = False

    if base_only or 'pristine' in name or 'virgin' in name:
        pass
    elif 'aged' in name \
      or 'ozonized' in name \
      or 'treated' in name \
      or 'gealtert' in name \
      or '0d' in name \
      or 'biofilm' in name \
      or 'residual' in name \
      or 'beached' in name \
      or 'hps' in name \
      or 'ups' in name \
      or 'bps' in name \
      or 'chlorinated' in name \
      or 'cps' in name:
        modified = True

    uni_name = 'modified ' if modified else ''

    if 'pvc' in name:
        uni_name += 'PVC'
    elif 'twp' in name:
        uni_name += 'TWP'
    elif 'pbat' in name:
        uni_name += 'PBAT'
    elif 'pla' in name:
        uni_name += 'PLA'
    elif 'pp' in name:
        uni_name += 'PP'
    elif 'pet' in name:
        uni_name += 'PET'
    elif 'pa' in name:
        uni_name += 'PA'
    elif 'ps' in name:
        uni_name += 'PS'
    elif 'pc' in name:
        uni_name += 'PC'
    elif 'pmma' in name:
        uni_name += 'PMMA'
    elif 'pe' in name:
        uni_name += 'PE'
    elif 'ethylene vinyl acetat' == name:
        uni_name += 'EVA'
    else:
        raise ValueError(f'Please provide a known abbreviation/name for the sorbent - "{name}" encountered')

    return uni_name


def label_modifications(data):
    data_copy = data.copy(deep=True)
    data_copy[('cleaned_sorbent')] = data_copy['sorbent_name'].apply(unify_sorbent)
    mods_array = []
    for i, (ref, group) in enumerate(data_copy.groupby('doi', as_index=False)):
        mods = _calc_mods(group)
        mods = mods.apply(lambda x: x if x == 0 else 10*i+x)
        mods_array.append(mods)
    return pd.concat(mods_array)


def _calc_mods(group):
    mod_grouped = group.groupby('cleaned_sorbent', as_index=False)
    modified_sorbs = [sorb for sorb in mod_grouped.groups.keys() if 'modified' in sorb]
    sorbs = [sorb for sorb in mod_grouped.groups.keys() if not 'modified' in sorb]

    if (mod_count:=len(modified_sorbs)) == 0:
        # returning a scalar confuses pandas here
        return group.apply(lambda _: 0, axis='columns')
    elif mod_count * 2 == len(mod_grouped):
        if not len(mod_grouped.groups[sorbs[0]]) == len(mod_grouped.groups[modified_sorbs[0]]):
            assert len(sorbs) == 1
            # group by original name, each is assumed to be a single modifaction type
            # hence number the groups with ascending numbers in arbitrary order
            mods = group.groupby([('sorbent_name')], as_index=False).ngroup()
            # then swap w/o modifcation to 0 to guarantee said value
            unmod_idx = mod_grouped.groups[sorbs[0]]
            swap_val = mods.loc[unmod_idx].iloc[0]
            mods.where(mods!=0, swap_val, inplace=True)
            mods.loc[unmod_idx] = 0
            return mods
        else:
            # all modfications are assumed to be same, unmodified is labelled 0
            return group.apply(
                lambda row: int('modified' in row['cleaned_sorbent']),
                axis='columns'
            )
    else:
        # two special cases: 2 aged MP w/o standard MP and 1 Biofilm w/ 2 standard MP
        # icn both cases: 0 is unmodified 1 is modified
        return group.apply(
            lambda row: int('modified' in row['cleaned_sorbent']),
            axis='columns'
        )

def _calc_sorbent_stats(h=0, c=0, o=0, n=0, cl=0, zero=0.001):
    H = 1.00794
    C = 12.0107
    O = 15.9994
    N = 14.0067
    CL = 35.45

    c_weight_percent = (c*C / (h*H + c*C + o*O + n*N + cl*CL)) * 100
    hc_ratio = zero if h*c == 0 else h/c
    oc_ratio = zero if o*c == 0 else o/c

    return c_weight_percent, hc_ratio, oc_ratio


def calculate_sorbent_stats(name: str) -> Tuple[float, float, float]:

    if (lname:=name.lower()) == 'pvc': # assume C2H3Cl
        return _calc_sorbent_stats(c=2, h=3, cl=1)
    elif lname == 'twp':
        raise ValueError("Can't calculate sorbent statistics for tire wear particles of unspecified composition.")
    elif lname == 'pbat': # assume C10H10O4.C6H10O4.C4H10O2 or C8H6O4.C6H10O4.C4H10O2 TODO
        return _calc_sorbent_stats(c=10+6+4, h=10+10+10, o=4+4+2)
    elif lname == 'pla': # assume C3H4O2
        return _calc_sorbent_stats(c=3, h=4, o=2)
    elif lname == 'pp': # assume C3H6
        return _calc_sorbent_stats(c=3, h=6)
    elif lname == 'pet': # assume C10H8O4
        return _calc_sorbent_stats(c=10, h=8, o=4)
    elif lname == 'pa': # assume C6H11NO
        return _calc_sorbent_stats(c=6, h=11, o=1, n=1)
    elif lname == 'ps': # assume C8H8
        return _calc_sorbent_stats(c=8, h=8)
    elif lname == 'pe': # assume C2H4
        return _calc_sorbent_stats(c=2, h=4)
    elif lname == 'eva': # assume C4H6O2.C2H4
        return _calc_sorbent_stats(c=4+2, h=6+4, o=2)
    elif lname == 'pmma': # assume C5H8O2
        return _calc_sorbent_stats(c=5, h=8, o=2)
    elif lname == 'pc': # assume C16H14O3
        return _calc_sorbent_stats(c=16, h=14, o=3)
    elif lname == 'pvc': # assume C2H3Cl
        return _calc_sorbent_stats(c=2, h=3, cl=1)
    else:
        raise ValueError(f"Can't calculate sorbent statistics for {name}.")
