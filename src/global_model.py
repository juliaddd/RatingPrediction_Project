import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

def load_data(path: str):
    df = pd.read_csv(path)
    return df

def preprocess_data(df: pd.DataFrame):
    # Delete unnecessary columns
    # Score related columns and columns with little impact
    df.drop(labels=['Ranked','Score-10','Score-9','Score-8','Score-7','Score-6',
                    'Score-5','Score-4','Score-3','Score-2','Score-1', 'Ranked',
                    'Name','English name','Japanese name','Aired','Producers',
                     'Licensors','Duration','Members','Watching','On-Hold', 'MAL_ID'], inplace=True)

    # Converting into numerical columns
    numeric_cols = ['Score', 'Episodes']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handling NaN
    df["Episodes"] = df['Episodes'].fillna(0)
    df.dropna(subset=["Score"], inplace=True)

    # Multi-hot encoding for Genres
    genre_dummies = df['Genres'].str.get_dummies(sep=', ')
    df = pd.concat([df, genre_dummies], axis=1)
    df.drop(columns=['Genres'], inplace=True)

    # Extracting Year from Premier
    df[['Season', 'Year']] = df['Premiered'].str.split(' ', expand=True)
    df.drop(columns=['Season', 'Premiered'], inplace=True)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0)

    # One-hot encoding for Rating, Type and top Studious
    type_dummies = pd.get_dummies(df['Type'], prefix='Type')
    df = pd.concat([df, type_dummies], axis=1)

    rating_dummies = pd.get_dummies(df['Rating'].str.split(' - ', expand=True)[0], prefix='Rating')
    df = pd.concat([df, rating_dummies], axis=1)

    top_studios = df['Studios'].value_counts().nlargest(10).index
    df['Studios'] = df['Studios'].apply(lambda x: x if x in top_studios else 'Other')
    studio_dummies = pd.get_dummies(df['Studios'], prefix='Studio')
    df = pd.concat([df, studio_dummies], axis=1)

    df.drop(columns=['Type', 'Rating', 'Studios'], inplace=True)

    # Bool columns to int
    bool_cols = df.select_dtypes('bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df

def train_global_model(df: pd.DataFrame):
    X = df.drop(columns=['Score'])
    y = df['Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2:", r2_score(y_test, y_pred))

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances.nlargest(20).plot(kind='barh', figsize=(8, 6))
    plt.title("Top-20 Feature Importances")
    plt.show()

    # Cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    print("R2:", scores)
    print("Mean R2:", np.mean(scores))

    return model