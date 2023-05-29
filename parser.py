import json
from typing import List, Optional
import cloudscraper
import pandas as pd

import requests
from typing import Dict, Any

import config


def item_to_list(item: Dict[str, Any]):
    # Convert item dictionary to a list in a specific order
    return [item['floorEthPrice'], item['maxEthPrice'], item['minEthPrice'], item['sales'], item['timeStamp'],
            item['volumeInEth']]


class Parser:
    def __init__(self, collections: List[str] = None):
        # Initialize Parser object with a list of collections
        if not collections:
            collections = list()
        self.collections: List[str] = collections
        self.collections_data = {collection: list() for collection in self.collections}
        self.session = cloudscraper.create_scraper()

    def parse_from_file(self, file_path: str):
        # Parse collections data from a file
        try:
            with open(file_path, 'r', encoding='utf8') as f:
                json_data = f.read()
        except:
            json_data = "{}"
        self.collections_data = json.loads(json_data)

    def parse(self, file_path: Optional[str] = None, data_file_path: Optional[str] = None):
        # Parse collections data either from a file or by calling parse_collection() for each collection
        if data_file_path:
            self.parse_from_file(data_file_path)
        else:
            for collection in self.collections:
                self.parse_collection(collection)
        if file_path:
            self.save_to_file(file_path)

    def parse_collection(self, collection: str):
        # Parse data for a specific collection using OpenSea API
        try:
            response = self.session.get(
                f"https://api.pro.opensea.io/analytics/{collection}/volumeAndSales?duration=All+time")
            self.collections_data[collection] = response.json()['data']
            print(f"Got collection [{collection}] data: [{response.status_code}]")
        except requests.exceptions.ConnectionError:
            print("Connection error")
            return

    def save_to_file(self, file_path: str = "data.csv", by_collection_path: str = None):
        # Save collections data to a file in either CSV or JSON format
        if file_path.endswith("csv"):
            df = pd.DataFrame(
                columns=['Collection', 'FloorEthPrice', 'MaxEthPrice', 'MinEthPrice', 'Sales', 'TimeStamp',
                         'VolumeInEth'])
            for collection in self.collections:
                for item in self.collections_data[collection]:
                    df.loc[len(df)] = [collection] + item_to_list(item)
                    # df = df.append([collection] + utils.item_to_list(item), ignore_index=True)
            df.to_csv(file_path, index=False)
        elif file_path.endswith("json"):
            with open(file_path, "w", encoding='utf-8') as f:
                f.write(json.dumps(self.collections_data))
        else:
            raise ValueError("File format not supported")
        if by_collection_path:
            for collection in self.collections:
                df = pd.DataFrame(
                    columns=['Collection', 'FloorEthPrice', 'MaxEthPrice', 'MinEthPrice', 'Sales', 'TimeStamp',
                             'VolumeInEth'])
                for item in self.collections_data[collection]:
                    df.loc[len(df)] = [collection] + item_to_list(item)
                df.to_csv(f"{by_collection_path}/{collection}.csv", index=False)


def main():
    # Create a Parser object with NFT collections from the config module
    parser = Parser(config.NFT_COLLECTIONS)
    # Parse collections data from a file, if provided, or call parse_collection() for each collection
    parser.parse("data/data.json")
    # Save collections data to a CSV file and separate CSV files for each collection
    parser.save_to_file("data/data.csv", by_collection_path="data/analysis/prices/actual/")


if __name__ == '__main__':
    # Call the main function if this script is executed directly
    main()
