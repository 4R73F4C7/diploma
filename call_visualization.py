import pandas as pd

import config
import utils


def main():
    # Specify the types of data to process
    types = ["actual", "predicted", "prices"]

    # Iterate over the types of data
    for _type in types:
        # Iterate over the NFT collections specified in the config
        for collection in config.NFT_COLLECTIONS:
            # Read the CSV file corresponding to the current data type and collection
            df = pd.read_csv(f"data/analysis/actions/{_type if _type != 'prices' else 'actual'}/{collection}.csv")

            # Convert the 'TimeStamp' column to datetime format
            df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], unit='s')

            # Call a utility function to draw the collection data (assumed to be implemented in the "utils" module)
            utils.draw_collection(df, collection, type=_type, show=False, save=True)


if __name__ == '__main__':
    # Call the main function if this script is executed directly
    main()
