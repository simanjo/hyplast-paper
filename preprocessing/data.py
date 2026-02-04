#%%
from functools import reduce
import operator

import pandas as pd

RAW_DATA = {
	# Dict containing the raw data values with doi as key
	# and values again dicts with keys given by the final tables format
	# following a JSON-like format, with explicit
	# '_explode_by' and '_keys' keys signifying unfoldable substructures
	# EXAMPLE:
    # 'DOI': {
    #     '_explode_by': ['sorbat_name', 'sorbent_name'],
	#     'ph':,
    #     'T':,
    #     'SLR (kg/L)': ,
    #     'salinity': ,
    #     'conc_range (µg/L)': {
	# 		'_keys': ['sorbat_name'],
    #         'sorbat1': ,
    #         'sorbat2': ,
	# 	},
	# 	'sorbat_name': ['sorbat1', 'sorbat2'],
    #     'SSA (m^2/g)': {
	# 		'_keys': ['sorbent_name'],
    #         'sorbent1': 7.04,
	# 		'sorbent2': 5.10,
	# 	},
    #     'sorbent_name': ['sorbent1', 'sorbent2'],
    #     'isotherm_data': {
    #         'sorbat1': {
    #             'sorbent1': {
	# 				'kf': ,
	# 				'n': ,
	# 				'r2': 
	# 			},
	# 			'sorbent2': {
	# 				'kf':,
	# 				'n': ,
	# 				'r2':
	# 			},
	# 		},
    #         'sorbat2': {
    #             'sorbent1': {
	# 				'kf': ,
	# 				'n': ,
	# 				'r2': 
	# 			},
	# 			'sorbent2': {
	# 				'kf':,
	# 				'n': ,
	# 				'r2':
	# 			},
	# 		},
    #     'scaling_factor': {
    #         'kf': '10**(6-3n)'
	# 	}
	# },
	}

SORBAT_ABBR = {
	# Dict containing the sorbat abbreveations occuring in the raw data
}


def _explode_data(entry):
	if not isinstance(entry, dict):
		print('Not a dict: ', entry)
		return entry
	if '_explode_by' not in entry.keys():
		print('no way to explode by')
		return entry
	
	# fix concentration range first:
	conc_range = entry['conc_range (µg/L)']
	if isinstance(conc_range, dict):
		keys = conc_range.pop('_keys')
		if len(keys) == 1:
			entry['low conc_range (µg/L)'] = {
				k: v[0] for k, v in conc_range.items()
			} | {'_keys': keys}
			entry['high conc_range (µg/L)'] = {
				k: v[1] for k, v in conc_range.items()
			} | {'_keys': keys}
		else:
			assert len(keys) == 2
			entry['low conc_range (µg/L)'] = {
				ko: {
					k: v[0] for k, v in v_dict.items()
				} for ko, v_dict in conc_range.items()
			} | {'_keys': keys}
			entry['high conc_range (µg/L)'] = {
				ko: {
					k: v[1] for k, v in v_dict.items()
				} for ko, v_dict in conc_range.items()
			} | {'_keys': keys}
		# reset _keys
		conc_range['_keys'] = keys
	else:
		entry['low conc_range (µg/L)'] = conc_range[0]
		entry['high conc_range (µg/L)'] = conc_range[1]
	
	expl = entry['_explode_by']
	assert isinstance(expl, list), 'Wrong specification of variables to explode by' + expl

    # copy simple entries
	single_entries = pd.Series()
	multi_keys = []
	for key, val in entry.items():
		if key in ['_explode_by', 'conc_range (µg/L)']:
			continue
		if isinstance(val, dict):
			multi_keys.append(key)
			continue
		single_entries[key] = val
		
	df = pd.DataFrame(single_entries).T
	
    # explode multi entries
	for expl_key in expl:
		df = df.explode(expl_key)
	for mkey in multi_keys:
		if mkey == 'isotherm_data' or mkey == 'scaling_factor':
			pass
		else:
			df[mkey] = pd.NA
			loc_keys = entry[mkey].pop('_keys')
			if len(loc_keys) == 1:
				col = loc_keys[0]
				for key_val in entry[col]:
					df.loc[df[col] == key_val, mkey] = entry[mkey][key_val]
			else:
				assert len(loc_keys) == 2
				col1 = loc_keys[0]
				col2 = loc_keys[1]
				for key_val1 in entry[col1]:
					for key_val2 in entry[col2]:
						df.loc[
							(df[col1] == key_val1) \
							& (df[col2] == key_val2),
							mkey] = entry[mkey][key_val1][key_val2]
			entry[mkey]['_keys'] = loc_keys
	df.reset_index(inplace=True, drop=True)

    # helper function to extract isotherm data
    # from arbitrarily deep nested dicts
	def _add_isotherm(row):
		col_vals = [row[key] for key in entry['_explode_by']]
		try:
			iso_data = reduce(operator.getitem, col_vals, entry['isotherm_data'])
		except KeyError as e:
			print('No isotherm data for: ', col_vals, entry['isotherm_data'])
			iso_data = {'kf': pd.NA, 'n': pd.NA, 'r2': pd.NA}
		scaling = entry['scaling_factor']
		scaling_str = ''
		if 'n' in scaling.keys():
			assert scaling['n'] == '1/n', \
				f'Only inversion allowed for n-rescaling, got {scaling['n']}'
			iso_data['scaled_n'] = 1/iso_data['n'] # type: ignore
			scaling_str += 'n: 1/n'
		else:
			iso_data['scaled_n'] = iso_data['n'] # type: ignore
		if 'kf' in scaling.keys() and not pd.isna(iso_data['kf']): # type: ignore
			kf_scale = scaling['kf']
			if '_keys' in scaling.keys():
				assert len(ks:=scaling['_keys']) == 1
				kf_scale = kf_scale[row[ks[0]]]
			if isinstance(kf_scale, tuple):
				log, kf_exp = kf_scale
				assert log == '10**kf'
				kf_exp = kf_exp.replace('3n', '(3*n)')#
				kf_exp = kf_exp.replace('n', str(iso_data['scaled_n'])) # type: ignore
				scaling_str = f'kf: {kf_scale} -- ' + scaling_str
				iso_data['scaled_kf'] = (10 ** iso_data['kf']) * eval(kf_exp)	# type: ignore
			else:
				assert isinstance(kf_scale, str), kf_scale
				if 'n' in kf_scale:
					# kf_scale = 'lambda n: ' + kf_scale
					kf_scale = kf_scale.replace('3n', '(3*n)')#
					kf_scale = kf_scale.replace('n', str(iso_data['scaled_n'])) # type: ignore
					scaling_str = f'kf: {kf_scale} -- ' + scaling_str
				if 'kf' in kf_scale:
					assert kf_scale == '10**kf'
					iso_data['scaled_kf'] = 10 ** iso_data['kf'] # type: ignore
					scaling_str = 'kf: 10**kf' + scaling_str
				else:
					iso_data['scaled_kf'] = iso_data['kf'] * eval(kf_scale)	# type: ignore
					scaling_str = kf_scale
		else:
			iso_data['scaled_kf'] = iso_data['kf'] # type: ignore
		iso_data['scaling'] = scaling_str # type: ignore
		return pd.Series(iso_data)

	df[['kf', 'n', 'r2', 'scaled_n', 'scaled_kf', 'scaling']] \
		= df.apply(_add_isotherm, axis=1)
	df.dropna(subset=['kf', 'n'], inplace=True)

	# unfold abbreviations in sorbat names
	df['sorbat_name'] = df['sorbat_name'].apply(
		lambda name: SORBAT_ABBR.get(name, name)
	)
	return df


# %%
import os
from pathlib import Path

from preprocessing.utils import get_smiles


def save_raw_data(save_path, raw_data=RAW_DATA, overwrite=False):
	
	DATA_WO_DOIS = pd.concat([
		_explode_data(data) for data in raw_data.values() # type: ignore
	]).reset_index(drop=True)
	# force column ordering for backwardscapability
	DATA_WO_DOIS = DATA_WO_DOIS[[
		'ph', 'T', 'SLR (kg/L)', 'salinity', 'sorbat_name',
		'sorbent_name', 'SSA (m^2/g)', 'low conc_range (µg/L)',
		'high conc_range (µg/L)', 'kf', 'n', 'r2', 'scaled_n',
		'scaled_kf', 'scaling'
	]]

	DATA = pd.concat([
		_explode_data(data).assign(doi=doi) for doi, data in raw_data.items() # type: ignore
	]).reset_index(drop=True)
	
	# force column ordering for backwardscapability
	DATA = DATA[[
		'ph', 'T', 'SLR (kg/L)', 'salinity', 'sorbat_name',
		'sorbent_name', 'SSA (m^2/g)', 'low conc_range (µg/L)',
		'high conc_range (µg/L)', 'kf', 'n', 'r2', 'scaled_n',
		'scaled_kf', 'scaling', 'doi'
	]]


	DATA_W_SMILES = DATA.copy(deep=True)
	DATA_W_SMILES['smiles'] = DATA_W_SMILES['sorbat_name'].apply(get_smiles)

	os.makedirs(save_path, exist_ok=True)

	if not (
		csv_path:=Path(save_path) / 'raw_hyplast.csv'
    ).is_file() or overwrite:
		DATA.to_csv(csv_path, index=False)

	if not (
		csv_path_wo_doi:=Path(save_path) / 'raw_hyplast_wo_doi.csv'
    ).is_file() or overwrite:
		DATA_WO_DOIS.to_csv(csv_path_wo_doi, index=False)

	if not (
		csv_path_w_smiles:=Path(save_path) / 'raw_hyplast_w_smiles.csv'
    ).is_file() or overwrite:
		DATA_W_SMILES.to_csv(csv_path_w_smiles, index=False)

# %%
