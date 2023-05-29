import pandas as pd
import json

import config
import utils


def main():
    # Read JSON data from file
    with open("data/analysis/results.json", 'r', encoding='utf8') as f:
        json_data = f.read()

    # Parse JSON data
    data = json.loads(json_data)

    # Define column names for the DataFrame
    cols = [
        "Collection",
        "Params",
        "Model",
        "FloorEthPrice",
        "VolumeInEth",
        "Sales",
        "test_rmse",
        "test_mae",
        "train_rmse",
        "train_mae",
        "test_predict"
    ]

    print("START")

    # Create an empty DataFrame with specified columns
    df = pd.DataFrame(columns=cols)

    # Iterate over data and populate the DataFrame
    for collection, data_collection in data.items():
        for param, data_param in data_collection.items():
            for model, data_model in data_param['models'].items():
                # Check parameter conditions
                price_param = "FloorEthPrice" in param
                volume_param = "VolumeInEth" in param
                sales_param = "Sales" in param

                # Create a new row for the DataFrame
                new_row = [
                    collection,
                    param,
                    model,
                    price_param,
                    volume_param,
                    sales_param,
                    data_model['stats']['test_rmse'],
                    data_model['stats']['test_mae'],
                    data_model['stats']['train_rmse'],
                    data_model['stats']['train_mae'],
                    data_model['stats']['test_predict'],
                ]
                # Append the new row to the DataFrame
                df.loc[len(df)] = new_row

    # Save the DataFrame to a CSV file
    df.to_csv("data/analysis/tables/data.csv", index=False)

    # Group the DataFrame by "Collection" column
    dfs = df.groupby(by=["Collection"])

    # Process each group separately
    for name, name_df in dfs:
        # Save group-specific DataFrame to a CSV file
        name_df.to_csv(f"data/analysis/tables/{name}.csv", index=False)

        # Read price DataFrame from file
        price_df = pd.read_csv(f"data/analysis/prices/actual/{name}.csv")

        # Determine indices to replace and remove from the price DataFrame
        replace_start_index = int(len(price_df) * config.TRAIN_SIZE)
        remove_end_index = replace_start_index + config.LOOKBACK

        # Remove specified indices from the price DataFrame
        price_df.drop(range(replace_start_index, remove_end_index), inplace=True)

        # Remove the last rows from the price DataFrame
        price_df = price_df[:-config.LOOKBACK]

        # Calculate a weighted score in the group-specific DataFrame
        name_df['weighted_score'] = 0.6 * name_df['test_rmse'] + 0.4 * name_df['test_mae']

        # Find the row with the minimum weighted score
        best_row = name_df[name_df['weighted_score'] == name_df['weighted_score'].min()]

        # Update "Prediction_Start" column in the price DataFrame
        price_df['Prediction_Start'] = False
        price_df.loc[replace_start_index - config.LOOKBACK, 'Prediction_Start'] = True

        # Set predicted values in the price DataFrame
        price_df.loc[replace_start_index - config.LOOKBACK:, 'FloorEthPrice'] = best_row['test_predict'].values[0]

        # Save the updated price DataFrame to a CSV file
        price_df.to_csv(f"data/analysis/prices/predicted/{name}.csv", index=False)

        # Perform additional actions using the updated price DataFrame
        df_calls = utils.add_indicators_and_decisions(price_df, "FloorEthPrice")

        # Save the resulting DataFrame to a CSV file
        df_calls.to_csv(f"data/analysis/actions/predicted/{name}.csv", index=False)

    print("END")


if __name__ == '__main__':
    # Entry point of the program
    main()
