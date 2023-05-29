import json

try:
    # Attempt to open the "config.json" file for reading
    with open("config.json", 'r', encoding='utf8') as f:
        # Load the JSON content from the file into the "setup" dictionary
        setup = json.load(f)
    # Print a message indicating that the config file was found and its setup will be used
    print("Config file found. Using its setup.")
    # Print the content of the "setup" dictionary
    print(setup)
except:
    # If an exception occurs (e.g., file not found or invalid JSON format), use default values
    setup = {
        "collections": [
            "boredapeyachtclub",
        ],
        "lookback": 10,
        "train_size": 0.8
    }
    # Print a message indicating that the config file was not found, and default values will be used
    print("Config file not found. Using default values. Created config.json, containing them.")
    # Print the default values in the "setup" dictionary
    print(setup)
    # Create a new config file with the default values
    with open("config.json", 'w', encoding='utf8') as f:
        # Write the contents of the "setup" dictionary to the file in JSON format
        json.dump(setup, f)

# Set variables based on the values in the "setup" dictionary
NFT_COLLECTIONS = setup['collections']
LOOKBACK = setup['lookback']
TRAIN_SIZE = setup['train_size']
TEST_SIZE = 1 - TRAIN_SIZE
