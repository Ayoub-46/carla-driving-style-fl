import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    """Loads and preprocesses the dataset."""
    df = pd.read_csv(path)
    
    # Separate features and labels
    # Label column is 'label' (Cautious, Normal, Aggressive)
    X = df.drop(columns=['label'])
    y = df['label']

    # Encode labels to 0, 1, 2
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    return X, y, le

def get_client_partitions(X, y, num_clients=2):
    """
    Splits data into partitions for each client.
    Returns a list of tuples: [(X_train, y_train, X_test, y_test), ...]
    """
    # Simple IID split (random shuffle)
    # For non-IID (more realistic), you would split by user ID or similar.
    
    partitions = []
    
    # Split total data into chunks
    chunk_size = len(X) // num_clients
    
    for i in range(num_clients):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        
        X_part = X.iloc[start:end]
        y_part = y[start:end]
        
        # Split each client's partition into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_part, y_part, test_size=0.2, random_state=42
        )
        
        partitions.append((X_train, y_train, X_test, y_test))
        
    return partitions