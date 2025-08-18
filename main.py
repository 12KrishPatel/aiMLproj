import pandas as pd
import io

def load_spambase(data_path, names_path):
    # Loads the spambase dataset into a Pandas DB
    col_names = []

    try:
        with open(names_path, 'r') as f:
            for line in f:
                if ":" in line:
                    col_name = line.split(":")[0].strip()
                    col_names.append(col_name)
    except FileNotFoundError:
        print(f"Error: ' {names_path} not found.")
        return None
    except Exception as e:
        print(f"Error occured while reading '{names_path}':{e}")
        return None
    
    # Make sure col names are there
    if not col_names:
        print(f"No col names found in '{col_names}'")
        pass

    
    # Load data into Pandas DB
    try:
        df = pd.read_csv(data_path, header=None, names=col_names)
        return df
    except FileNotFoundError:
        print(f"Error: '{data_path}' not found.")
        return None
    except Exception as e:
        print(f"Error occured while reading '{data_path}':{e}")
        return None
    
if __name__ == "__main__":
    data_file = 'spambase.data'
    names_file = 'spambase.names'

    spambase_df = load_spambase(data_file, names_file)

    if spambase_df is not None:
        print("--- Spambase DataFrame Loaded Successfully! ---")
        print("\n--- DataFrame Head ---")
        print(spambase_df.head())
        print("\n--- DataFrame Info ---")
        print(spambase_df.info())
        print("\n--- DataFrame Shape ---")
        print(f"Rows: {spambase_df.shape[0]}, Columns: {spambase_df.shape[1]}")
    else:
        print("\nFailed to load Spambase DataFrame.")
