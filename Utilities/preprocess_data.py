import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,MinMaxScaler
from imblearn.over_sampling import SMOTE

# ------------------------ 1. Load Dataset ------------------------
def load_data(file_path):
    """
    Loads the dataset from a CSV file.

    Parameters:
    - file_path(str):Path to the CSV file.

    Returns:
    - DataFrame: Loaded dataset as a Pandas DataFrame.
    """
    df=pd.read_csv(file_path)
    print("\nFirst 5 Rows:\n",df.head())
    
    print("\nDataset Info:\n",df.info())
    return df

# ------------------------ 2. Handle Missing Values ------------------------
def handle_missing_values(df):
    """
    Fills missing values in the dataset.

    - Numerical columns: Filled with the median.
    - Categorical columns: Filled with the most frequent(mode) value.

    Parameters:
    - df(DataFrame):Input dataset.

    Returns:
    - DataFrame: Dataset with missing values handled.
    """
    
    # Fill numeric columns
    df.fillna(df.median(numeric_only=True),inplace=True)  
    # Fill categorical columns
    
    # Fill categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        mode_value = df[col].mode()
        if not mode_value.empty:
            df[col] = df[col].fillna(mode_value.iloc[0])
        else:
            df[col].fillna('Unknown', inplace=True)  # Modify this depending on your dataset's needs
            print(f"Warning: No mode found for column: {col}, defaulting to 'Unknown'")
    
    
    return df 
    return df
# ------------------------ 3. Encode Categorical Features ------------------------
def encode_categorical(df):
    """
    Converts categorical(object-type) features into numerical values using Label Encoding.

    Parameters:
    - df(DataFrame):Input dataset.

    Returns:
    - DataFrame: Dataset with categorical features encoded.
    """
    le=LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col]=le.fit_transform(df[col])
    return df

# ------------------------ 4. Process Date-Time Features ------------------------
def process_datetime(df,date_column):
    """
    Extracts useful features from a datetime column.

    - Extracts year,month,day,and hour.
    - Removes the original datetime column.

    Parameters:
    - df(DataFrame):Input dataset.
    - date_column(str):Name of the datetime column.

    Returns:
    - DataFrame: Dataset with datetime features extracted.
    """
    if date_column in df.columns:
        df[date_column]=pd.to_datetime(df[date_column])
        df['year']=df[date_column].dt.year
        df['month']=df[date_column].dt.month
        df['day']=df[date_column].dt.day
        df['hour']=df[date_column].dt.hour
        df.drop(columns=[date_column],inplace=True)  # Drop original timestamp
    return df

# ------------------------ 5. Remove Outliers ------------------------
def removeOutliers(df):
    """
    using the Interquartile Range(IQR) Removes_outliers.

    Parameters:
    - df(DataFrame):Input dataset.

    Returns:
    - DataFrame: Dataset with outliers removed.
    """
    for col in df.select_dtypes(include=['number']).columns:
        Q1=df[col].quantile(0.25)
        Q3=df[col].quantile(0.75)
        IQR=Q3 - Q1
        lower_bound=Q1 - 1.5 * IQR
        upper_bound=Q3 + 1.5 * IQR
        df=df[(df[col]>=lower_bound)&(df[col]<= upper_bound)]
    return df

# ------------------------ 6. Scale Features ------------------------
def scale_features(df,scaling_method="standard"):
    """
    Scales numerical features using StandardScaler or MinMaxScaler.

    Parameters:
    - df(DataFrame):Input dataset.
    - scaling_method(str):'standard' for StandardScaler,'minmax' for MinMaxScaler.

    Returns:
    - DataFrame: Scaled dataset.
    """
    scaler=StandardScaler() if scaling_method == "standard" else MinMaxScaler()
    df[df.select_dtypes(include=['number']).columns]=scaler.fit_transform(df.select_dtypes(include=['number']))
    return df

# ------------------------ 7. Split Data into Train & Test ------------------------
def partitionData(df,target_column,test_size=0.25):
    """
    Splits the dataset into training and testing sets.

    Parameters:
    - df(DataFrame):Input dataset.
    - target_column(str):Column to be predicted.
    - test_size(float):Proportion of dataset used for testing.

    Returns:
    - X_train,X_test,y_train,y_test==>Split dataset.
    """
    X=df.drop(columns=[target_column])
    y=df[target_column]
    return train_test_split(X,y,test_size=test_size,random_state=123)

# ------------------------ 9. Drop Unwanted Columns ------------------------
def drop_notImportcolumns(df,columns_to_drop):
    """
    Drops unnecessary columns from the dataset.

    Parameters:
    - df(DataFrame):Input dataset.
    - columns_to_drop(list):List of column names to be dropped.

    Returns:
    - DataFrame: Dataset with unwanted columns removed.
    """
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]  
    df.drop(columns=columns_to_drop, inplace=True)
    return df

# ------------------------ 10. Main Preprocessing Function ------------------------
def preprocess_data(file_path,target_column,drop_columns=None,date_column=None,scale_method="standard",balance_classes=False):
    """
    Performs end-to-end preprocessing on the dataset.

    Steps:
    1. Load dataset.
    2. Drop unwanted columns(if any).
    3. Handle missing values.
    4. Encode categorical features.
    5. Process datetime features(if applicable).
    6. Remove outliers.
    7. Scale numerical features.
    8. Split into train & test sets.
    9. Handle class imbalance(if required).

    Parameters:
    - file_path(str):Path to the dataset.
    - target_column(str):Column to be predicted.
    - drop_columns(list,optional):Columns to drop.
    - date_column(str,optional):Name of the date column.
    - scale_method(str,optional):'standard' or 'minmax' scaling.
    - balance_classes(bool,optional):Whether to apply SMOTE for class imbalance.

    Returns:
    - X_train,X_test,y_train,y_test: Preprocessed data ready for modeling.
    """
    ##### print for debugging only 
    print(f'1. Load dataset.')
    df=load_data(file_path)
    print(f'2. Drop unwanted columns')
    if drop_columns:
        df=drop_notImportcolumns(df,drop_columns)
        
    #print(f'3. Handle missing values.')
    df=handle_missing_values(df)
    #print(f'4. Encode categorical features.')
    df=encode_categorical(df)
    #print(f'5. Process datetime features(if applicable)')
    if date_column:
        df=process_datetime(df,date_column)
    #print("6. Remove outliers.")
    df=removeOutliers(df)
    #print(f'7. Scale numerical features.')
    df=scale_features(df,scale_method)
    #print(f'8. Split into train & test sets.')
    X_train,X_test,y_train,y_test=partitionData(df,target_column)
    
    #print(f'9. Handle class imbalance(if required).')
    # if balance_classes:
    #     X_train,y_train=handleImbalance(X_train,y_train)

    print("\n Preprocessing Complete!")
    return X_train,X_test,y_train,y_test
