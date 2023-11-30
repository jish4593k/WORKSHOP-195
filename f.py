import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tkinter import Tk, filedialog

def load_data_gui():
    """Load data interactively using Tkinter GUI."""
    root = Tk()
    root.title("Data Loading GUI")

    file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])

    if file_path:
        data = pd.read_csv(file_path)
        root.destroy()
        return data
    else:
        root.destroy()
        return None

def preprocess_data(df):
    """Preprocess data by filtering stars based on solar distance and gravity."""
    star_dist = df[df['solar_distance'] <= 100]
    star_dist.reset_index(inplace=True, drop=True)

    final_stars = star_dist[(star_dist['solar_gravities'] >= 150) & (star_dist['solar_gravities'] <= 350)]
    final_stars.reset_index(inplace=True, drop=True)

    return final_stars

def visualize_filtered_stars(df):
    """Visualize the filtered stars."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='solar_distance', y='solar_gravities', data=df)
    plt.title('Filtered Stars: Solar Distance vs. Solar Gravity')
    plt.xlabel('Solar Distance')
    plt.ylabel('Solar Gravity')
    plt.show()

def perform_regression(X, y):
    """Perform linear regression using TensorFlow and Keras."""
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build a simple neural network for regression using Keras
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    # Evaluate the model on the test set
    mse = model.evaluate(X_test_scaled, y_test)
    print(f'Mean Squared Error: {mse}')

    # Visualize the predictions vs. actual values
    predictions = model.predict(X_test_scaled)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=predictions.flatten())
    plt.title('Actual vs. Predicted Solar Gravity')
    plt.xlabel('Actual Solar Gravity')
    plt.ylabel('Predicted Solar Gravity')
    plt.show()

def main():
    print('Welcome to Advanced Data Analysis')

    # Load data interactively using Tkinter GUI
    data = load_data_gui()

    if data is not None:
        # Preprocess data by filtering stars
        filtered_stars = preprocess_data(data)

        # Visualize the filtered stars
        visualize_filtered_stars(filtered_stars)

        # Perform regression using TensorFlow and Keras
        X = filtered_stars[['solar_distance']]
        y = filtered_stars['solar_gravities']
        perform_regression(X, y)

if __name__ == "__main__":
    main()
