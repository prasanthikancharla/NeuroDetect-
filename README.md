import tkinter as tk
from tkinter import filedialog, messagebox, Text
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import numpy as np

class ParkinsonsDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Parkinson's Disease Detection")
        self.root.geometry("1440x810")  # Set GUI resolution to 3/4 of 1920x1080
        self.root.configure(bg='white')
        
        self.create_widgets()
        self.df = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.results = {}

    def create_widgets(self):
        title_frame = tk.Frame(self.root, bg='blue')
        title_frame.pack(fill=tk.X)

        title_label = tk.Label(title_frame, text="Parkinson's Disease Detection", font=("Helvetica", 20, "bold"), bg='blue', fg='white')
        title_label.pack(pady=20)

        main_frame = tk.Frame(self.root, bg='white')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        btn_frame = tk.Frame(main_frame, bg='white')
        btn_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        self.result_text = Text(main_frame, width=80, height=35, bg='lightgrey', font=("Helvetica", 12))
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        btn_style = {'font': ("Helvetica", 12), 'bg': 'lightblue', 'fg': 'black', 'width': 20, 'height': 2}
        
        tk.Button(btn_frame, text="Load Dataset", command=self.load_dataset, **btn_style).pack(pady=5)
        tk.Button(btn_frame, text="Decision Tree", command=self.run_decision_tree, **btn_style).pack(pady=5)
        tk.Button(btn_frame, text="Random Forest", command=self.run_random_forest, **btn_style).pack(pady=5)
        tk.Button(btn_frame, text="Logistic Regression", command=self.run_logistic_regression, **btn_style).pack(pady=5)
        tk.Button(btn_frame, text="SVM", command=self.run_svm, **btn_style).pack(pady=5)
        tk.Button(btn_frame, text="Naive Bayes", command=self.run_naive_bayes, **btn_style).pack(pady=5)
        tk.Button(btn_frame, text="KNN", command=self.run_knn, **btn_style).pack(pady=5)
        tk.Button(btn_frame, text="Comparison Graph", command=self.show_comparison_graph, **btn_style).pack(pady=5)
        tk.Button(btn_frame, text="Exit", command=self.root.quit, **btn_style).pack(pady=5)

    def load_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.df = pd.read_csv(file_path)
            self.perform_eda()
            self.preprocess_data()

    def perform_eda(self):
        messagebox.showinfo("EDA", "Performing Exploratory Data Analysis (EDA)...")
        plt.figure(figsize=(10, 6))
        sns.countplot(x='status', data=self.df)
        plt.title('Class Distribution')
        plt.show()

        plt.figure(figsize=(20, 20))
        sns.heatmap(self.df.corr(numeric_only=True), annot=True)
        plt.title('Correlation Matrix')
        plt.show()

        plt.figure(figsize=(15, 15))
        sns.pairplot(self.df.select_dtypes(include=[np.number]), hue='status')
        plt.title('Pairplot')
        plt.show()

    def preprocess_data(self):
        if 'name' in self.df.columns:
            self.df = self.df.drop(columns=['name'])  # Drop non-numeric column
        X = self.df.drop(columns=['status'])
        y = self.df['status']
        
        # Handling missing values if any
        X = X.fillna(X.mean())

        # Balance the dataset using SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    def run_algorithm(self, model, model_name):
        if self.X_train is None or self.y_train is None:
            messagebox.showerror("Error", "Please load and preprocess the dataset first.")
            return

        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)

        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        self.show_metrics(model_name, accuracy, precision, recall, f1, cm)

    def show_metrics(self, model_name, accuracy, precision, recall, f1, cm):
        metrics_text = f"{model_name} Metrics\n\nAccuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}"
        self.result_text.insert(tk.END, metrics_text + '\n\n')

        plt.figure(figsize=(10, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        metrics = [accuracy, precision, recall, f1]
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        plt.figure(figsize=(10, 6))
        sns.barplot(x=metrics_names, y=metrics)
        plt.title(f'{model_name} Metrics')
        plt.show()

    def run_decision_tree(self):
        model = DecisionTreeClassifier()
        self.run_algorithm(model, "Decision Tree")

    def run_random_forest(self):
        model = RandomForestClassifier()
        self.run_algorithm(model, "Random Forest")

    def run_logistic_regression(self):
        model = LogisticRegression(max_iter=1000)
        self.run_algorithm(model, "Logistic Regression")

    def run_svm(self):
        model = SVC()
        self.run_algorithm(model, "SVM")

    def run_naive_bayes(self):
        model = GaussianNB()
        self.run_algorithm(model, "Naive Bayes")

    def run_knn(self):
        model = KNeighborsClassifier()
        self.run_algorithm(model, "KNN")

    def show_comparison_graph(self):
        if not self.results:
            messagebox.showinfo("Error", "No results to compare. Please run the algorithms first.")
            return
        algorithms = list(self.results.keys())
        accuracies = [self.results[algo]['accuracy'] for algo in algorithms]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=algorithms, y=accuracies)
        plt.title('Algorithm Comparison')
        plt.xlabel('Algorithm')
        plt.ylabel('Accuracy')
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = ParkinsonsDetectionApp(root)
    root.mainloop()

