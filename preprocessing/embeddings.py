# %%
from unittest.mock import patch
from pathlib import Path

import pandas as pd
import numpy as np

from rdkit.Chem import AllChem, MACCSkeys, DataStructs
from rdkit import Chem
import pubchempy
from mol2vec.features import mol2alt_sentence, sentences2vec
import gensim

def _calc_rdkit(smiles, fpsize=1024):
    # RDKIT embedding
	fpgen = AllChem.GetRDKitFPGenerator(fpSize=fpsize)
	# HACK/NOT recommended... use side-effect ridden conversion of
	# fingerprint to numpy array with prior initialization of empty arrays
	# also need convers
	fps_dict = {
		sm: np.zeros((0,), dtype=int)
		for sm in smiles
	}
	fps = pd.DataFrame.from_dict({
		sm: arr
		for sm, arr in fps_dict.items()
		if DataStructs.ConvertToNumpyArray(
			fpgen.GetFingerprint(Chem.MolFromSmiles(sm)),
			arr
		) == None
	}, orient='index', columns=[f'rdkit_{i}' for i in range(fpsize)])
	# drop constant columns
	fps.drop(
		columns=[
			col for col in fps.columns
			if fps[col].nunique(dropna=False)==1
		],inplace=True
	)
	return fps


def _calc_maccs(smiles):
	fps = pd.DataFrame.from_dict({
		sm: np.asarray(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(sm)))
		for sm in smiles
	}, orient='index', columns=[f'maccs_{i}' for i in range(167)])
	# drop constant columns
	fps.drop(
		columns=[
			col for col in fps.columns
			if fps[col].nunique(dropna=False)==1
		],inplace=True
	)
	return fps


def _calc_pubchem(smiles):
	compound_dict = {
		sm: pubchempy.get_compounds(sm, 'smiles')[0]
		for sm in smiles
	}
	pubchem_fps = pd.DataFrame.from_dict({
		sm: np.asarray([int(bit) for bit in comp.cactvs_fingerprint])
		for sm, comp in compound_dict.items()
	}, orient='index', columns=[f'pubchem_{i}' for i in range(881)])
	# drop constant columns
	pubchem_fps.drop(
		columns=[
			col for col in pubchem_fps.columns
			if pubchem_fps[col].nunique(dropna=False)==1
		],inplace=True
	)
	return pubchem_fps


def _calc_mol2vec(smiles, model_path):
	if not model_path.is_file():
		print('Trying to fetch the pretrained mol2vec 300dim model from github.')
		model_url = 'https://github.com/samoturk/mol2vec/raw/refs/heads/master/examples/models/model_300dim.pkl'
		import requests
		import tqdm
		r = requests.get(model_url, stream=True)
		with open(model_path, 'wb') as fh:
			for chunk in tqdm.tqdm(r.iter_content(chunk_size=None)):
				if chunk:
					fh.write(chunk)
					fh.flush()

	model = gensim.models.word2vec.Word2Vec.load(str(model_path))

	# we need to patch mol2vec, as it does not support gensim v4
	# follows the idea of the fixing commit in https://github.com/forgee-master/mol2vec-updated
	# the goal is to replace vocab.keys() with index_to_key used inside sentences2vec
	# we double patch first vocab (effectively with self, but practically with
	# our models KeyedVectors instance) and then the non-existing keys()
	with patch.object(gensim.models.keyedvectors.KeyedVectors, 'vocab', model.wv):
		with patch.object(
			gensim.models.keyedvectors.KeyedVectors, 'keys',
			lambda self: self.index_to_key, create=True
		):
			mol2vec_fps = pd.DataFrame.from_dict({
				# result is list of converted sentences, as there is only 1
				# we are safe to choose the first one
				sm: sentences2vec(
					[mol2alt_sentence(Chem.MolFromSmiles(sm), radius=1)],
					model, unseen='UNK'
				)[0] for sm in smiles
			}, orient='index', columns=[f'mol2vec_{i}' for i in range(300)])
	# drop constant columns
	mol2vec_fps.drop(
		columns=[
			col for col in mol2vec_fps.columns
			if mol2vec_fps[col].nunique(dropna=False)==1
		],inplace=True
	)
	return mol2vec_fps


def _calc_abrahams(smiles, abrahams_csv):
	abrahams_df = pd.read_csv(abrahams_csv)
	abrahams_df['Input SMILES'] = abrahams_df['Input SMILES'].apply(Chem.CanonSmiles)
    
	def get_single_smiles(sm):
		try:
			params = abrahams_df.loc[
                abrahams_df['Input SMILES'] == Chem.CanonSmiles(sm)
            ].iloc[0]
		except BaseException as e:
			print(sm)
			return None
		error = params['Error']
		if not error == '-':
			if len(splitted:=error.split('--')) > 1:
				ref = splitted[-1].strip().split(' ')
				if not ref[0].strip() == 'See':
					print(ref)
					print(splitted)
					assert False
				alt_smiles = ref[-1].strip()
				params = abrahams_df.loc[
					abrahams_df['Input SMILES'] == Chem.CanonSmiles(alt_smiles)
				].iloc[0]
			else:
				print(f'Unrecognized error: {error}, skipping smiles {smiles}')
		return params[['E', 'S', 'A', 'B', 'V', 'L']]

	return pd.DataFrame({
		sm: get_single_smiles(sm) for sm in smiles
	}).T


def add_embeddings(csv_path, embedding, overwrite=False):
	if not Path(csv_path).is_file():
		raise ValueError(f'{csv_path} does not point to a file.')

	data = pd.read_csv(csv_path)
	unique_smiles = np.unique(np.asarray(data['smiles']))

	if (embedding:=embedding.lower()) == 'maccs':
		fp_dict = _calc_maccs(unique_smiles)
	elif embedding == 'mol2vec':
		fp_dict = _calc_mol2vec(unique_smiles, Path(csv_path).parent / 'mol2vec_300dim.pkl')
	elif embedding == 'rdkit':
		fp_dict = _calc_rdkit(unique_smiles)
	elif embedding == 'pubchem':
		fp_dict = _calc_pubchem(unique_smiles)
	elif embedding == 'abrahams':
		abrahams_csv = Path(csv_path).parent / 'RMG_SoluteML.csv'
		fp_dict = _calc_abrahams(unique_smiles, abrahams_csv)
		fp_dict.columns = fp_dict.columns.map(lambda c: f'abrahams_{c}')
	else:
		raise ValueError(f'Please specify a valid embedding, was {embedding}.')

	save_path = Path(csv_path).parent / f'hyplast_{embedding}.csv'
	if overwrite or not save_path.is_file():
		data.join(
			fp_dict, on='smiles', validate='many_to_one'
		).to_csv(save_path, index=False)

# %%
