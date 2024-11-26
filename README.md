**mlflow tracking:** mlflow server --backend-store-uri sqlite:///trackdata/database.db --default-artifact-root trackdata/artifacts --host 0.0.0.0 --port 5000

**be:** uvicorn main:app --reload --host 0.0.0.0 --port 8000

**fe:** streamlit run app.py
