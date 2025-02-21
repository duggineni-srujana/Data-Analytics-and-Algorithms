from KNNcustom import KNNcustom
from CustomSVM import CustomSVM
from LinearRegressioncustom import LinearRegressioncustom
from NaiveBayescustom import NaiveBayescustom
import pandas as pd
import numpy as np
from shiny import App, ui, render, reactive
from sklearn.metrics import accuracy_score
import joblib

# Load pre-trained models from .pkl files
model_filenames = {
    "KNN": "knn_model.pkl",
    "Random Forest": "RF_model.pkl",
    "SVM": "svm_model.pkl",
    "Logistic Regression": "LinReg_model.pkl",
    "Naive Bayes": "NaiveBayes_model.pkl",
    "KMeans": "kmeans.pkl",
}
scaler = joblib.load("scaler.pkl")  
encoder = joblib.load("le.pkl")
# Load models into a dictionary with exception handling
models = {}
for name, filename in model_filenames.items():
    try:
        models[name] = joblib.load(filename)
    except Exception as e:
        print(f"Error loading model '{name}' from {filename}: {e}")

# UI Definition
app_ui = ui.page_fluid(
    ui.tags.style("""
        body {
            font-family: 'Arial', sans-serif;
        }
        .container {
            max-width: 800px;
            margin: 20px auto;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1); /* Add a subtle shadow */
            background-color: #f9f9f9; /* Light background color */
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
        }
        .model-selection {
            margin-bottom: 20px;
        }
        .input-area {
            margin-bottom: 20px;
        }
        .results-area {
            margin-top: 20px;
        }
        .btn-primary { /* Style for the prediction button */
            background-color: #4CAF50;
            border-color: #4CAF50;
        }
        .btn-primary:hover {
            background-color: #45a049;
            border-color: #45a049;
        }
    """),
    ui.tags.div(  # Main container
        ui.tags.div(  # Header section
            ui.h3("Name : Srujana Duggineni"),
            ui.h3("MSc Data Science."),
            ui.h3("College name : South East Technological University"),
            class_="header"  
        ),
        ui.hr(),
        ui.tags.div(  # Model selection
            ui.h4("ML Model Selector"),
            ui.input_select("model", "Choose a Model:", choices=list(models.keys()), selected="Logistic Regression"),
            class_="model-selection" 
        ),
        ui.hr(),
        ui.tags.div(  # Input area
            ui.input_file("file", "Upload CSV for Predictions", accept=[".csv"], multiple=False),
            ui.br(),
            ui.h4("Enter Data Manually"),
            ui.input_text("input_features", "Enter feature values (comma-separated)", placeholder="5.1,3.5,1.4,0.2"),
            ui.br(),
            ui.input_action_button("predict_btn", "Make Prediction", class_="btn-primary"),
            class_="input-area"  
        ),
        ui.hr(),
        ui.tags.div(  # Results area
            ui.h4("Model Accuracy:"),
            ui.output_text("accuracy_output"),
            ui.br(),
            ui.h4("Prediction Result:"),
            ui.output_text("prediction_output"),
            class_="results-area"  
        ),
        class_="container" 
    )
)


# Server Logic
def server(input, output, session):

    @output
    @render.text
    def accuracy_output():
        selected_model = input.model()
        model = models.get(selected_model)

        if model is None:
            return f"Error: Model '{selected_model}' is not available."

        file = input.file()
        if file is None:
            return "Please upload a CSV file or enter data manually."

        try:
            df = pd.read_csv(file["datapath"])
            df.info()
            if 'target' in df.columns:
                X_test = df.drop(columns=["target"])
            else:
                X_test = df

            if X_test.empty:
                return "Error: Uploaded file does not contain valid features."
            
            if X_test.empty:
                return "Error: Uploaded file does not contain valid features."
            X_test_encoded = encoder.transform(X_test)  
            X_test_scaled = scaler.transform(X_test_encoded) 


            y_pred = model.predict(X_test_scaled)
            accuracy = "N/A"
            if "target" in df.columns:
                accuracy = accuracy_score(df["target"], y_pred)

            return f"Selected Model: {selected_model}\nAccuracy: {accuracy:.4f}" if accuracy != "N/A" else f"Selected Model: {selected_model}\nAccuracy: Not Available"
        except Exception as e:
            return f"Error processing CSV file: {e}"

    @output
    @render.text
    def prediction_output():
        selected_model = input.model()
        model = models.get(selected_model)

        if model is None:
            return f"Error: Model '{selected_model}' is not available."

        file = input.file()
        if file:
            try:
                df = pd.read_csv(file["datapath"])
                X_test = df.drop(columns=["target"], errors="ignore")

                if X_test.empty:
                    return "Error: No valid features found in the uploaded CSV."

                predictions = model.predict(X_test)
                return f"Predictions: {list(predictions)}"
            except Exception as e:
                return f"Error processing CSV file: {e}"

        manual_input = input.input_features()
        if manual_input:
            try:
                features = np.array([float(x) for x in manual_input.split(",")]).reshape(1, -1)
                prediction = model.predict(features)[0]
                return f"Prediction: {prediction}"
            except ValueError:
                return "Invalid input format. Enter values as comma-separated numbers."
            except Exception as e:
                return f"Error processing input: {e}"

        return "No input data provided."


# Run the application
app = App(app_ui, server)

if __name__ == "__main__":
    print("Starting Shiny application...")
    app.run()