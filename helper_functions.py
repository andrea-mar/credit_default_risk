import pandas as pd


def select_features(df_features: pd.DataFrame, pca_df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes columns containing 'BILL_AMT' from the first DataFrame (df_features) 
    and appends columns from the second DataFrame (pca_df) while aligning 
    by index.

    Parameters:
        df_features (pd.DataFrame): The input DataFrame to filter and modify.
        pca_df (pd.DataFrame): The DataFrame containing PCA components to 
                               append to df_features.

    Returns:
        pd.DataFrame: A new DataFrame with filtered columns from df_features 
                      and PCA components from pca_df appended.
    """
    df = df_features.loc[:, ~df_features.columns.str.contains('BILL_AMT')].reset_index(drop=True)
    df = pd.concat([df, pca_df], axis=1)
    return df