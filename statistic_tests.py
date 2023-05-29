import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu, kruskal, ks_2samp
import config


def main():
    # Define column names for the statistics DataFrame
    cols = [
        "Collection",
        "Params",
        "Params2",
        "Model",
        "test_rmse",
        "test_mae",
        "train_rmse",
        "train_mae",
        "test_predict",
        'PValueAvg',
        "Wilcoxon_PValue",
        "Kruskal_PValue",
        "MannWhitney_PValue",
        "Kolmogorov_PValue",
    ]

    dfs = {}

    # Load tables for each NFT collection
    for collection in config.NFT_COLLECTIONS:
        dfs[collection] = pd.read_csv(f"data/analysis/tables/{collection}.csv")

    dfs_stats = []

    # Perform statistical analysis for each NFT collection
    for collection, df in dfs.items():
        dfs_stats.append(pd.DataFrame(columns=cols))

        # Remove unnecessary columns from the DataFrame
        df.drop(columns=[
            "FloorEthPrice",
            "VolumeInEth",
            "Sales",
        ], inplace=True)

        # Group the DataFrame by "Model" column
        df_grouped = df.groupby(by=["Model"])

        # Perform statistical tests for each model and pair of parameters
        for model, df_model in df_grouped:
            # Convert "test_predict" column values to lists of floats
            df_model.test_predict = [list(map(float, x.strip('[]').split(', '))) for x in df_model.test_predict]

            for i1, line1 in df_model.iterrows():
                for i2, line2 in df_model.iterrows():
                    if i1 == i2 or line1.test_predict == line2.test_predict:
                        continue

                    # Perform statistical tests and calculate p-values
                    wilcoxon_stat = wilcoxon(line1.test_predict, line2.test_predict)
                    kruskal_stat = kruskal(line1.test_predict, line2.test_predict)
                    mannwhitneyu_stat = mannwhitneyu(line1.test_predict, line2.test_predict)
                    ks_2samp_stat = ks_2samp(line1.test_predict, line2.test_predict)

                    # Calculate average p-value
                    line1[f'PValueAvg'] = round(
                        (sum([wilcoxon_stat.pvalue, kruskal_stat.pvalue, mannwhitneyu_stat.pvalue,
                              ks_2samp_stat.pvalue]) / 4), 4)

                    # Assign individual p-values to respective columns
                    line1[f'Wilcoxon_PValue'] = round(wilcoxon_stat.pvalue, 4)
                    line1[f'Kruskal_PValue'] = round(kruskal_stat.pvalue, 4)
                    line1[f'MannWhitney_PValue'] = round(mannwhitneyu_stat.pvalue, 4)
                    line1[f'Kolmogorov_PValue'] = round(ks_2samp_stat.pvalue, 4)

                    # Assign second parameter value for comparison
                    line1['Params2'] = line2.Params

                    # Append the statistical results to the statistics DataFrame
                    dfs_stats[-1].loc[len(dfs_stats[-1])] = line1

        # Remove unnecessary columns from the statistics DataFrame
        dfs_stats[-1].drop(columns=["test_predict"], inplace=True)
        dfs_stats[-1].rename(columns={"Params": "Params1"}, inplace=True)

        # Save the statistics DataFrame to a CSV file
        dfs_stats[-1].to_csv(f"data/analysis/statistics/{collection}.csv", float_format='%.40f', index=False)


if __name__ == '__main__':
    # Entry point of the program
    main()
