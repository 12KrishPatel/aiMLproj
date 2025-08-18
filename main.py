import pandas as pd
import io
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore

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
    else:
        print("\nFailed to load Spambase DataFrame.")

    # Seperating features into X and Y
    # Target COL
    target_col = "capital_run_length_total"
    drop_col = '| UCI Machine Learning Repository'

    X = spambase_df.drop([drop_col, target_col], axis=1)
    Y = spambase_df[target_col]
    
    # Split data into training and testing
    # Stratify = y to maintain spam/ham in both sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42, stratify=Y)

    # Scale features AFTER splitting to prevent data leaks
    scaler = StandardScaler()

    xtrain_scaled = scaler.fit_transform(x_train)
    xtest_scaled = scaler.transform(x_test)
    # Convert scaled numpy arrays back to pandas DB
    xtrain_scaled_df = pd.DataFrame(xtrain_scaled, columns=x_train.columns)
    xtest_scaled_df = pd.DataFrame(xtest_scaled, columns=x_test.columns)

    # Train the model
    model = LogisticRegression(random_state=42, solver='liblinear')
    print(f"\nTraining {type(model).__name__} model...")

    # Train using scaled data
    model.fit(xtrain_scaled_df, y_train)

    print(f"{type(model).__name__} model training complete!")


    y_pred = model.predict(xtest_scaled_df)

    print("\nFirst 10 actual labels (y_test):")
    print(y_test.head(10).tolist()) # Convert to list for cleaner printing
    print("\nFirst 10 predicted labels (y_pred):")
    print(y_pred[:10].tolist())