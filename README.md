# Student Service Request Monitoring System

Streamlit deployment of the Section B midterm Decision Tree model.

## Run locally
1. Use Python 3.13.
2. Open terminal in this folder.
3. Run `py -3.13 -m pip install -r requirements.txt`
4. Run `py -3.13 -m streamlit run app.py`
5. In Dataset, upload your original `student_service_requests.csv`.
6. In Prediction, click **Train and save midterm model pipeline**.

## Midterm model reused
- Target: `SLA_Breached`; 1 means breach and 0 means no breach.
- Label generated only for completed records when `Processing_Days > SLA_Days`.
- Features: Service_Type, Requester_Type, Requested_Channel, Priority, Assigned_Office, Queue_Length, Required_Documents_Complete, SLA_Days, Submission_Weekday, Queue_Category.
- Preprocessing: OneHotEncoder(handle_unknown='ignore') for categorical data; numerical data passed through.
- Classifier: DecisionTreeClassifier(random_state=42, class_weight='balanced').
- Split: 80/20, random_state=42, stratified when possible.

## Important
The uploads included the notebook code and assignment PDF, but not the original CSV or a fitted model. Therefore this project does not invent training records or metrics. Upload the same midterm CSV and train the pipeline. For Cloud deployment with a preloaded model, train locally and commit `models/trained_model.joblib`. Never commit private student records.

The notebook derived the target from completed records. The Streamlit app adds an explicit train/save button because the uploaded notebook did not persist a fitted pipeline. The dashboard review list is historical, not a real-time pending queue. Model predictions support staff review only and must never automatically deny student services.
