import kagglehub
from kagglehub import KaggleDatasetAdapter

DATASET = "zoya77/multi-modal-data-for-supply-chain-risk-management"

def load_data():
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        DATASET,
        file_path=""
    )
    
    print("Loaded dataset:", df.shape)
    print("Columns:", df.columns)

    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())
