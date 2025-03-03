import joblib
from shiny import App, ui, render, reactive
import time
# Model and Preprocessor Loading 
models = {}

NB = {
    "KNN": "https://github.com/duggineni-srujana/Data-Analytics-and-Algorithms/blob/main/Notebooks/knn.ipynb",
    "Random Forest": "https://github.com/duggineni-srujana/Data-Analytics-and-Algorithms/blob/main/Notebooks/RandomForest_DecisionTree.ipynb",
    "SVM": "https://github.com/duggineni-srujana/Data-Analytics-and-Algorithms/blob/main/Notebooks/SVM.ipynb",
    "Linear Regression": "https://github.com/duggineni-srujana/Data-Analytics-and-Algorithms/blob/main/Notebooks/LinearRegression.ipynb",
    "Naive Bayes": "https://github.com/duggineni-srujana/Data-Analytics-and-Algorithms/blob/main/Notebooks/NBayes.ipynb",
    "KMeans": "https://github.com/duggineni-srujana/Data-Analytics-and-Algorithms/blob/main/Notebooks/kmeans.ipynb",
}

doc = {
    "KNN": "https://github.com/duggineni-srujana/Data-Analytics-and-Algorithms/blob/main/Logs/Log_KNN.pdf",
    "Random Forest": "https://github.com/duggineni-srujana/Data-Analytics-and-Algorithms/blob/main/Logs/Log_RandomForest_DecisionTree.pdf",
    "SVM": "https://github.com/duggineni-srujana/Data-Analytics-and-Algorithms/blob/main/Logs/Log_SVM.pdf",
    "Linear Regression": "https://github.com/duggineni-srujana/Data-Analytics-and-Algorithms/blob/main/Logs/Log_LinearRegression.pdf",
    "Naive Bayes": "https://github.com/duggineni-srujana/Data-Analytics-and-Algorithms/blob/main/Logs/Log_NaiveBayes.pdf",
    "KMeans": "https://github.com/duggineni-srujana/Data-Analytics-and-Algorithms/blob/main/Logs/Log_Kmeans.pdf",
}


# model_filenames = {  # Consistent naming
#     "KNN": "knn_model.pkl",
#     "Random Forest": "RF_model.pkl",
#     "SVM": "svm_model.pkl",
#     "Linear Regression": "LinReg_model.pkl",
#     "Naive Bayes": "NaiveBayes_model.pkl",
#     "KMeans": "kmeans.pkl",
# }

# # Load Models, Scalers, and Encoders
# for name, filename in model_filenames.items():
#     try:
#         models[name] = joblib.load(filename)
#     except Exception as e:
#         print(f"Error loading model '{name}': {e}")


# Define the UI
app_ui = ui.page_fluid(   
    
    # Add Google Fonts to the page header
    ui.tags.head(
        ui.tags.link(rel="stylesheet", href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap")
    ),
    
    ui.tags.style("""
        body {
            font-family: 'Lora', serif;  # Apply the stylish font to the entire body
            background-color: #e8f5e9;  # Add a lively gradient background
            color: #333;
            line-height: 1.6;
            font-style: italic;
        }
        .container {
            max-width: 900px;
            margin: 30px auto;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
            background-color: #fff;
        }
        .header {
            text-align: center;
            margin-bottom: 25px;
            color: #4CAF50;  # Use a lively green color for header
        }
        .section {
            margin-bottom: 30px;
        }
        h1, h2, h3, h4 {
            font-family: 'Playfair Display', serif; /* or Dancing Script, cursive */
            font-weight: 600;
            color: #4a6572; /* Primary color */
        }

        h1 {
            font-size: 2.5em;
            text-align: center;
            margin-bottom: 20px;
        }

        h2 {
            font-size: 2em;
            margin-bottom: 15px;
        }

        h3 {
            font-size: 1.5em;
            margin-bottom: 10px;
        }

        .btn-primary {
            background-color: #4a6572;  # A vibrant red-orange color for primary buttons
            border-color: #ff6f61;
            color: white;
            font-weight: bold;
            border-radius: 6px;
            padding: 12px 20px;
            transition: all 0.3s ease-in-out;  # Smooth hover effect
            cursor: pointer;
        }
        .btn-primary:hover {
            background-color: #344955;  # Darker shade on hover
            border-color: #ff5f50;
            cursor: pointer;
        }
        .btn-secondary {
            background-color: #66bb6a;  # Bright blue for secondary buttons
            border-color: #03a9f4;
            color: white;
            font-weight: bold;
            border-radius: 6px;
            padding: 12px 20px;
            transition: all 0.3s ease-in-out;
        }
        .btn-secondary:hover {
            background-color: #4caf50;  # Darker blue on hover
            border-color: #0288d1;
            cursor: pointer;
        }
        hr {
            border-top: 1px solid #eee;
            margin: 20px 0;
        }
        .algorithm-list {
            list-style-type: disc;
            padding-left: 20px;
            color: #5e35b1;  # Lively purple for list items
        }
        .algorithm-list li {
            margin-bottom: 8px;
        }
        .page-title {
            color: #ff8c00;  # Orange for page titles
            font-size: 32px;
            text-align: center;
        }
        .section-title {
            color: #d32f2f;  # Red color for section titles
            font-size: 24px;
            margin-bottom: 10px;
        }
        .highlight {
            color: #f57c00;  # Lively orange-yellow for highlights
        }
        .model-card {
            background-color: #f0f8ff;  /* Light blue for cards */
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
             margin-bottom: 20px;
            transition: transform 0.3s ease-in-out;
        }
        .model-card:hover {
            transform: translateY(-5px);  # Slight zoom effect on hover for cards
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        }
        .about-study {  /* Style for the "About Study" section */
        font-family: 'Lora', serif, cursive; /* Use Comic Sans MS */
        font-size: 18px; /* Slightly larger font size */
        line-height: 1.6; /* Increased line spacing */
        margin-bottom: 20px; /* Add some margin below */
        font-style: italic;
        }
        .about-study strong { /* Style for the bold text */
            font-weight: bold;
        }
        .nav-tabs .nav-link.active {
            background-color: #007bff;
            color: white;
        }
    """),
        ui.div(
        ui.h1("Welcome!"),
        style="""
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background-color: #ffffff;
            display: flex;
            padding: 20px;
            border-radius: 10px;
            z-index: 1000;
            max-width: 600px;
            width: 90%;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            text-align: center;
            transition: opacity 0.5s ease-in-out;
        """,
        id="welcome-screen"
        ),

        
        
        
        ui.navset_tab(
        # Main Page - Home
        ui.nav_panel(
            "Home",
            ui.h1("Machine Learning Algorithm Study"),
            ui.div(
                #ui.img(src=".static/photo.jpeg", class_="profile-photo", style="width:120px; height:120px; border-radius:50%;"),
                ui.div(
                    ui.p("👩‍🎓 Name: Srujana Duggineni"),
                    ui.p("📚 MSc Data Science, South East Technological University"),
                    ui.p("🖥️ Experienced in Python, R, SQL, and Data Visualization tools."),
                    ui.p("📊 Skilled in building predictive models and working with large datasets."),
                    ui.p("📧 Email :   ",  # Text before the link
                            ui.tags.a(
                                "srujanaduggineni@gmail.com",  
                                href="mailto:srujanaduggineni@gmail.com" ),
                        ),
                    ui.p("🔗 LinkedIn:   ",ui.tags.a(
                            "Srujana Duggineni",  
                            href="https://www.linkedin.com/in/srujana-duggineni-2110/",  
                            target="_blank"),
                    ),
                    ui.p("📂 Portfolio: ", 
                        ui.tags.a("Dive Into My Work", href="https://srujana-duggineni.shinyapps.io/portfolio_mlalgorithms/", target="_blank", style="color: #007bff; font-weight: bold;")),
                    ),
                
                style="display: flex; gap: 20px; align-items: center;"
            ),
            ui.hr(),

            # About Study
         ui.h3("About Study", class_="about-study"),
            ui.div(  # Container for styled paragraph
                ui.p("""
                    This project explores the implementation and comparison of multiple machine learning algorithms that I have worked on across various problem domains. Through this work, I aim to provide insights into how to choose the most suitable machine learning algorithm for different types of data and tasks, offering practical solutions for predictive modeling and unsupervised learning challenges.
                    """,
                    class_="about-study"
            ),
            ui.p(
                 
                "In the future, I plan to add additional models to this project, continuously expanding the scope to explore more advanced techniques and refine the comparisons, ensuring the project remains up to date with the latest advancements in machine learning.",
                class_="about-study"
            )
            ),
            ui.hr(),
            # Combined Model List and Logs
            ui.h4(" Models and Logs"),
            ui.tags.ul(
                ui.tags.li(
                    ui.a("Linear Regression", href=NB["Linear Regression"], target="_blank", style="color:#ff8c00;"),
                    " | ",  # Separator
                    ui.a("Log", href=doc["Linear Regression"], target="_blank", style="color:#ff8c00;")
                ),
                ui.tags.li(
                    ui.a("Support Vector Machine (SVM)", href=NB["SVM"], target="_blank", style="color:#ff8c00;"),
                    " | ",
                    ui.a("Log", href=doc["SVM"], target="_blank", style="color:#ff8c00;")
                ),
                ui.tags.li(
                    ui.a("Random Forest & Decision Tree", href=NB["Random Forest"], target="_blank", style="color:#ff8c00;"),
                    " | ",
                    ui.a("Log", href=doc["Random Forest"], target="_blank", style="color:#ff8c00;")
                ),
                ui.tags.li(
                    ui.a("K-Nearest Neighbors (KNN) Regression", href=NB["KNN"], target="_blank", style="color:#ff8c00;"),
                    " | ",
                    ui.a("Log", href=doc["KNN"], target="_blank", style="color:#ff8c00;")
                ),
                ui.tags.li(
                    ui.a("K-Means Clustering", href=NB["KMeans"], target="_blank", style="color:#ff8c00;"),
                    " | ",
                    ui.a("Log", href=doc["KMeans"], target="_blank", style="color:#ff8c00;")
                ),
                ui.tags.li(
                    ui.a("Naive Bayes", href=NB["Naive Bayes"], target="_blank", style="color:#ff8c00;"),
                    " | ",
                    ui.a("Log", href=doc["Naive Bayes"], target="_blank", style="color:#ff8c00;")
                ),
            ),
            
            # Navigation Links
            ui.h4("Explore More:"),
            ui.tags.ul(
                ui.tags.li(ui.a("Model List", href="#model-list")),
                ui.tags.li(ui.a("Performance Evaluation", href="#performance-evaluation")),
                ui.tags.li(ui.a("Data Preprocessing", href="#data-preprocessing"))
            ),
            

            ui.hr(),
        ),
        # Model List Page
        ui.nav_panel(
            "Model List",
            ui.h2("Machine Learning Models"),
            ui.p("Here are the models used in this study:"),
            ui.tags.ul(
                ui.tags.li("Linear Regression"),
                ui.tags.li("Support Vector Machine (SVM)"),
                ui.tags.li("Random Forest & Decision Tree"),
                ui.tags.li("K-Nearest Neighbors (KNN) Regression"),
                ui.tags.li("K-Means Clustering"),
                ui.tags.li("Naive Bayes")
            ),
        ),
        # Performance Evaluation Page
        ui.nav_panel(
            "Performance Evaluation",
            ui.h2("Performance Evaluation"),
            ui.p("Different performance metrics were used for different types of models."),
            ui.h3("Regression Metrics"),
            ui.tags.ul(
                ui.tags.li("Mean Squared Error (MSE)"),
                ui.tags.li("Mean Absolute Error (MAE)"),
                ui.tags.li("R² Score")
            ),
            ui.h3("Classification Metrics"),
            ui.tags.ul(
                ui.tags.li("Precision"),
                ui.tags.li("Recall"),
                ui.tags.li("F1-Score")
            ),
            ui.h3("Clustering Metrics"),
            ui.tags.ul(
                ui.tags.li("Silhouette Score")
            ),
        ),
        # Data Preprocessing Page
        ui.nav_panel(
            "Data Preprocessing",
            ui.h2("Data Preprocessing"),
            ui.p("Before training, the dataset undergoes preprocessing steps such as:"),
            ui.tags.ul(
                ui.tags.li("Handling missing values"),
                ui.tags.li("Encoding categorical variables"),
                ui.tags.li("Feature scaling"),
                ui.tags.li("Outlier detection"),
                ui.tags.li("Feature selection"),
                ui.tags.li("Splitting into training and test sets")
            ),
        ),
    ) 
)




# Server Logic
def server(input, output, session):


    @reactive.effect
    def remove_welcome_screen():
        time.sleep(1)  # Show "Welcome!" for 2 seconds
        ui.remove_ui("#welcome-screen")  # Remove welcome screen
   

    @output
    @render.ui
    def resource_links():
        selected_model = input.model_selector()
        github_link = NB.get(selected_model, "#")
        pdf_link = doc.get(selected_model, "#")
        # Style the <a> tags (buttons) directly!
        button_style = "padding: 5px 10px; border: none; border-radius: 4px; background-color: #007bff; color: white; text-decoration: none; margin-right: 10px;"  # Example button style

        return ui.TagList(
            ui.tags.a("View GitHub Code", href=github_link, target="_blank", class_="btn-secondary", style=button_style),
            ui.tags.a("View Model Log", href=pdf_link, target="_blank", class_="btn-secondary", style=button_style),
        )

    
   
# Run the Shiny App
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()