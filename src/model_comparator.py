import pandas as pd
import glob
import os

def calculate_composite_score(df, metrics=['accuracy', 'precision', 'recall', 'f1']):
    
    df_to_score = df[[col for col in metrics if col in df.columns]]
    
    if df_to_score.empty:
        return df

    df['Composite_Score'] = df_to_score.mean(axis=1)
    
    return df

def compare_data_variants_from_multiple_csv(folder_path):
    file_pattern = os.path.join(folder_path, '*.csv')
    csv_files = glob.glob(file_pattern)

    if not csv_files:
        print("No CSV files found to load. Please check the 'RESULTS_FOLDER' path.")
        return

    list_df = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            list_df.append(df)
        except:
            continue

    if not list_df:
        print("No data was loaded from the CSV files.")
        return

    df_all_results = pd.concat(list_df, ignore_index=True)
    
    metrics_to_score = ['accuracy', 'precision', 'recall', 'f1']
    
    df_with_score = calculate_composite_score(df_all_results, metrics_to_score)
    
    cols_to_average = metrics_to_score + ['Composite_Score']
    
    df_aggregated = df_with_score.groupby(['data_variant'])[cols_to_average].mean().reset_index()

    df_sorted_variants = df_aggregated.sort_values(by='Composite_Score', ascending=False)
    
    if df_sorted_variants.empty:
        print("No valid results found for comparison.")
        return

    best_variant = df_sorted_variants.iloc[0]

    print("--- OVERALL BEST DATA VARIANT (Aggregated Mean Score) ---")
    print(f"BEST_DATA_VARIANT: {best_variant['data_variant']}")
    print(f"MEAN_COMPOSITE_SCORE: {best_variant['Composite_Score']:.4f}")
    
    print("\n----------------------------------------------------")
    print("--- ALL DATA VARIANT RESULTS (Aggregated Mean Scores) ---")
    
    print(df_sorted_variants[['data_variant', 'Composite_Score', 'accuracy', 'precision', 'recall', 'f1']].to_string(index=False, float_format="%.4f"))


RESULTS_FOLDER = 'model_results/' 

compare_data_variants_from_multiple_csv(RESULTS_FOLDER)