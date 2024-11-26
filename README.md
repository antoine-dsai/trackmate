**mlflow tracking:** mlflow server --backend-store-uri sqlite:///track/database.db --default-artifact-root track/artifacts --host 0.0.0.0 --port 5000

**be:** uvicorn main:app --reload --host 0.0.0.0 --port 8000

**fe:** streamlit run app.py
