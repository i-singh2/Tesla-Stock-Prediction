# TeslaPrediction

This repository contains code and resources for predicting Tesla stock prices using machine learning and a Linear Regression Model. The project includes a Jupyter notebook to show and explain how to model was made and evaluated, and a Python script for generating Today's prediction, using the last trading day's data

## Files

- `tesla_stock_data.csv`: Contains the dataset used for training and testing the model, up to the date of submission
- `tesla_stock_model.ipynb`: A Jupyter notebook for exploratory data analysis and model development.
- `prediction.py`: MAIN FILE - A Python script that gathers and preprocesses data, trains and tests the model, calculates error and generates the prediction for the day's stock price, as well as provides advice whether you should buy, hold or sell. 

## Dependencies

To run the code in this repository, you need the following dependencies:

- Python 3.8 or higher
- Jupyter Notebook
- yfinance
- pandas
- ta
- scikit-learn
- matplotlib

You can install the required dependencies using the following command in your terminal:

```bash
pip install yfinance pandas ta matplotlib scikit-learn
```

## Setup and Usage

### Running the Jupyter Notebook (to see model development)

1. Clone the repository or download as ZIP:

    ```bash
    git clone https://github.com/i-singh2/Tesla-Stock-Prediction.git
    ```

2. Install the required dependencies as mentioned above

3. Go to the correct directory the project is stored in and start Jupyter Notebook:

    ```bash
    jupyter notebook
    ```

4. Open the `tesla_stock_model.ipynb` notebook and run the cells one by one to perform exploratory data analysis and model development.

### Running the Python Script

1. Clone the repository or download as ZIP:

    ```bash
    git clone https://github.com/i-singh2/Tesla-Stock-Prediction.git
    ```

2. Install the required dependencies as mentioned above

3. Run the script

4. When the script is running