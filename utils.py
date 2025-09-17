import numpy as np 
import pandas as pd
from tqdm import tqdm
import torch
from scipy.stats import truncnorm

def get_variant(reference, mutations, start_index=1):
    
    '''
    Given a reference sequence and a list of mutations, returns the variant sequence.
    '''

    if isinstance(reference, str):
        reference = np.array(list(reference))

    if mutations[0]=='WT':
        return reference.copy()
    
    else:
        variant = reference.copy()
        for mut in mutations:
            pos = int(mut[1:-1])-start_index
            variant[pos] = mut[-1]
        return variant

def one_hot_encode_variants(variant_list, wt_seq, break_char=','):
    
    '''
    One-hot encodes a list of variants given on a wild-type sequence. Automatically drops positions that are always the WT position.
    
    Input:
    - variant_list: list of variants, e.g. ['Y54F', 'Y54F,A56S', 'Y54F,A56S,L58P']
    - wt_seq: wild-type sequence, e.g. 'MDASSSSVTIFGQK.' If a string is provided, it will be converted to a numpy array.
    - break_char: character that separates mutations in the variant list, e.g. ',' for 'Y54F,A56S,L58P' or '_' for 'Y54F_A56S_L58P'.
    Output:
    - X_data_df: DataFrame with one-hot encoded variants, where each column corresponds to a mutation at a specific position in the wild-type sequence.
    '''
    
    if isinstance(wt_seq, str):
        wt_seq = np.array(list(wt_seq))

    all_sequences = pd.DataFrame(np.array([get_variant(wt_seq, i.split(break_char)) for i in variant_list]), columns=np.arange(1,len(wt_seq)+1))
    one_hot_sequences = pd.get_dummies(all_sequences).astype('int')
    all_available_mutations = one_hot_sequences.columns

    pos, c = np.unique([i.split('_')[0] for i in all_available_mutations], return_counts=True)
    pos_mutation_counts = pd.Series(c, index=pos)

    X_data_df = one_hot_sequences[one_hot_sequences.columns[np.array([pos_mutation_counts.loc[c.split('_')[0]]>1 for c in one_hot_sequences.columns])]]
    X_data_df.index = variant_list

    return X_data_df


def filter_variants(dataset, filter_out, max_num_mut=1000):

    keep_list = []
    
    for i in dataset.index:
        
        if i.count(',')<max_num_mut:
            keep=True
            for j in filter_out:
                if j in i:
                    keep=False
        else:
            keep=False
            
        if keep:
            keep_list.append(i)
            
    return dataset.loc[keep_list]

def count_mutations(dataset):
    var_dict = {}
    for i in dataset.index:
        for n in i.split(','):
            if n in var_dict:
                var_dict[n] += 1
            else:
                var_dict[n] = 1

    return pd.Series(var_dict)

def drop_variants_with_rare_mutations(dataset, n_min=2):
    
    mutation_counts = count_mutations(dataset)
    
    muts = mutation_counts.loc[mutation_counts>n_min]
    
    variants = []
    for i in tqdm(dataset.index):
        muts_in_var = i.split(',')
        
        if np.all(np.isin(muts_in_var, muts.index)):
            variants.append(i)
            
    return variants
    
def get_variant(reference, mutations, start_index=1):

    if mutations[0]=='WT':
        return reference.copy()
    
    else:
        variant = reference.copy()
        for mut in mutations:
            pos = int(mut[1:-1])-start_index
            variant[pos] = mut[-1]
        return variant

def make_tensors(X_data_df, input_df, y_col, yerr_col=None, drop_wt_cols=True, non_wt_columns=None):
    
    '''Prepare dataframe of analyzed data into tensors for inference.
    
        Inputs:
            X_data_df: One-hot encoded dataframe giving variant sequences, indexed by variant names.
            input_df: DataFrame with experimental (y) data, indexed by variant names.
            y_col: Name of the column in input_df containing the experimental measurement to regress against.
            yerr_col: Name of the column in input_df containing the error information.
                (Defaults to None, in which case no error information is used)
            drop_wt_cols: If True, will drop columns corresponding to the wild-type sequence.
                (Defaults to True)
            non_wt_columns: If provided, will use these columns instead of dropping WT columns.
                (Defaults to None, in which case WT columns are dropped if 'WT' is in the index of X_data_df).'''
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if drop_wt_cols:
        if non_wt_columns is None:
            
            if 'WT' in X_data_df.index:
                # If 'WT' is in the index
                non_wt_columns = X_data_df.columns[np.where(X_data_df.loc['WT']!=1)[0]]
                
            else:
                print("Cannot drop WT columns, 'WT' not in index and non_wt_columns not provided.")
                non_wt_columns = X_data_df.columns
            
        X_data = torch.tensor(X_data_df[non_wt_columns].values).float().to(device)
        
    else:
        X_data = torch.tensor(X_data_df.values).float().to(device)
        
    y_data = torch.tensor(input_df.loc[X_data_df.index][y_col]).float().to(device)
    
    if yerr_col is None:
        return X_data, y_data
    else:
    
        y_err = torch.tensor(input_df.loc[X_data_df.index][yerr_col]).float().to(device)
        return X_data, y_data, y_err
        

def calibration_curve(model, X_test, y_test, y_err, num_samples=100,
                      ci_min=10, n_ci_steps=20):
    """
    Generate a calibration curve for the model.
    
    Args:
        model (FlyroModel): The trained FlyroModel instance.
        X_test (torch.Tensor): Test feature matrix.
        y_test (torch.Tensor): Test target values.
        y_err_test (torch.Tensor): Test target errors.
        num_samples (int): Number of samples to draw for predictions (default=1000).
        
    Returns:
        tuple: Tuple containing predicted means and standard deviations.
    """
    raw_predictions = model.predict(X_test, samples=num_samples, summarize=False, y_err=y_err)
    
    ci_range = np.linspace(10, 100, n_ci_steps)
    empirical_coverage = []
    
    for ci in ci_range:
        
        lower_bound = np.percentile(raw_predictions, (100 - ci) / 2, axis=0)
        upper_bound = np.percentile(raw_predictions, 100 - (100 - ci) / 2, axis=0)
        
        empirical_coverage.append(np.mean((y_test.detach().numpy() < upper_bound)&(y_test.detach().numpy() > lower_bound)))
        
    return pd.Series(empirical_coverage, index=ci_range)
        
def summarize_predictions(predictions, ids=None, ci=95, label=''):
    """
    Summarize predictions with confidence intervals.
    
    Args:
        predictions (np.ndarray): Array of predicted values.
        ids (list, optional): List of IDs for the predictions (default=None).
        ci (int): Confidence interval percentage (default=95).
        
    Returns:
        pd.DataFrame: DataFrame with mean predictions and confidence intervals.
    """
    mean_predictions = predictions.mean(axis=0)
    lower_bound = np.percentile(predictions, (100 - ci) / 2, axis=0)
    upper_bound = np.percentile(predictions, 100 - (100 - ci) / 2, axis=0)
    
    if ids is None:
        ids = np.arange(len(predictions.T))
    
    df = pd.DataFrame({
        'mut': ids,
        f'y{label}_pred': mean_predictions,
        f'y{label}_pred_lower_bound': lower_bound,
        f'y{label}_pred_upper_bound': upper_bound
    }).set_index('mut')
    
    return df
    
def truncated_normal(loc, scale, min, max):

    min_z_score = (min - loc) / scale
    max_z_score = (max - loc) / scale

    return truncnorm(loc=loc, scale=scale, a=min_z_score, b=max_z_score)

from collections import Counter, defaultdict

def parse_variant(variant_str, sep=','):
    return variant_str.split(sep)

def compute_mutation_counts(variants, sep=','):
    counts = Counter()
    for var in variants:
        muts = parse_variant(var, sep=sep)
        counts.update(muts)
    return counts

def filter_variants(variants, min_count=3, verbose=False, sep=','):
    current_variants = set(variants)
    while True:
        mutation_counts = compute_mutation_counts(current_variants, sep=sep)
        bad_mutations = {mut for mut, count in mutation_counts.items() if count < min_count}

        # Identify variants to remove
        to_remove = set()
        for var in current_variants:
            muts = parse_variant(var, sep=sep)
            if any(m in bad_mutations for m in muts):
                to_remove.add(var)

        if verbose:
            print(f"Removing {len(to_remove)} variants due to under-represented mutations")

        if not to_remove:
            break

        current_variants -= to_remove

    return sorted(current_variants)