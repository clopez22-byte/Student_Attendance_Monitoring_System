import streamlit as st
import pandas as pd
from datetime import datetime
from modules.data_processing import validate_dataset,prepare_model_data,make_prediction_row
from modules.prediction import train_model,save_model,load_model,predict
from modules.exploratory_analysis import build_charts
from modules.dashboard import summary

st.set_page_config(page_title="Student Service Request Monitoring",page_icon="🎓",layout="wide")
st.title("🎓 Student Service Request Monitoring System")
st.caption("Section B | Decision Tree Classification | SLA Breach Prediction")
page=st.sidebar.selectbox("Module",["Home","Dataset","Data Analysis","Prediction","Dashboard","System Design"])
st.sidebar.warning("Predictions are decision-support outputs. Authorized staff must review and record final decisions.")

def current_data(): return st.session_state.get("raw_df")

if page=="Home":
    st.header("Home")
    st.write("A Streamlit decision-support application based on the midterm Student Service Request Monitoring notebook.")
    a,b,c=st.columns(3); a.metric("Classifier","Decision Tree"); b.metric("Target","SLA_Breached"); c.metric("Train/test split","80% / 20%")
    st.markdown("**Modules**\n- Dataset upload and validation\n- Exploratory data analysis\n- Predictive modeling\n- Decision-support dashboard\n- Systems Analysis and Design")
    st.info("Upload your original student_service_requests.csv in Dataset to begin. No substitute training data is included.")

elif page=="Dataset":
    st.header("1. Dataset Upload and Validation")
    f=st.file_uploader("Upload the original student_service_requests.csv",type=["csv"])
    if f:
        try:
            df=pd.read_csv(f); ok,msg=validate_dataset(df)
            if not ok: st.error(msg)
            else:
                st.success(msg); st.session_state["raw_df"]=df
                st.metric("Dataset rows",len(df)); st.dataframe(df.head(20),use_container_width=True)
                st.subheader("Missing-value and data-type summary")
                st.dataframe(pd.DataFrame({"Column":df.columns,"Data type":df.dtypes.astype(str).values,"Missing values":df.isna().sum().values}),use_container_width=True)
                st.write("Duplicate Request_ID values:",int(df["Request_ID"].duplicated().sum()))
                _,md=prepare_model_data(df); st.write("Eligible completed records:",len(md))
                if len(md): st.dataframe(md["SLA_Breached"].value_counts().rename_axis("SLA_Breached").reset_index(name="Count"))
        except Exception as e: st.error(f"Unable to read CSV: {e}")
    else: st.info("Required columns are the exact columns used in the midterm code: Request_ID, Service_Type, Requester_Type, Requested_Channel, Priority, Assigned_Office, Queue_Length, Required_Documents_Complete, Status, Processing_Days, SLA_Days, Submission_Date.")

elif page=="Data Analysis":
    st.header("2. Exploratory Data Analysis")
    df=current_data()
    if df is None: st.warning("Upload the dataset first.")
    else:
        _,md=prepare_model_data(df)
        st.subheader("Descriptive statistics"); st.dataframe(df.describe(include="all").transpose(),use_container_width=True)
        if len(md):
            target=md["SLA_Breached"].map({0:"No SLA Breach",1:"SLA Breach"}).value_counts().rename_axis("Class").reset_index(name="Count")
            st.plotly_chart(__import__("plotly.express",fromlist=[""]).pie(target,names="Class",values="Count",title="SLA Breach Class Distribution"),use_container_width=True)
        else: st.warning("No eligible completed records.")
        for fig in build_charts(df,md).values(): st.plotly_chart(fig,use_container_width=True)

elif page=="Prediction":
    st.header("3. Predictive Modeling")
    st.write("Midterm pipeline: OneHotEncoder(handle_unknown='ignore') + DecisionTreeClassifier(random_state=42, class_weight='balanced').")
    df=current_data()
    if df is not None and st.button("Train and save midterm model pipeline",type="primary"):
        try:
            _,md=prepare_model_data(df); pipe,metrics=train_model(md); save_model(pipe,metrics)
            st.session_state["saved_model"]={"pipeline":pipe,"metrics":metrics}
            st.success("Pipeline saved to models/trained_model.joblib.")
        except Exception as e: st.error(f"Training failed: {e}")
    saved=st.session_state.get("saved_model") or load_model()
    if saved:
        m=saved["metrics"]; st.subheader("Evaluation results")
        cols=st.columns(4)
        for col,key in zip(cols,["Accuracy","Precision (SLA Breach)","Recall (SLA Breach)","F1-score (SLA Breach)"]):
            col.metric(key,f"{m[key]:.4f}")
        st.write("Confusion matrix (rows = actual; columns = predicted; [No, Yes])")
        st.dataframe(pd.DataFrame(m["Confusion Matrix"],index=["Actual No","Actual Yes"],columns=["Predicted No","Predicted Yes"]))
        st.caption(f"Train: {m['Training records']} | Test: {m['Testing records']} | Eligible: {m['Eligible completed records']}")
        st.subheader("New request prediction")
        with st.form("predict_form"):
            x,y=st.columns(2)
            service=x.text_input("Service_Type","Certification"); requester=y.text_input("Requester_Type","Student")
            channel=x.selectbox("Requested_Channel",["Online","Walk-in"]); priority=y.selectbox("Priority",["Standard","Urgent"])
            office=x.text_input("Assigned_Office","Student Services"); queue=y.number_input("Queue_Length",0,10000,5)
            docs=x.selectbox("Required_Documents_Complete",["Yes","No"]); sla=y.number_input("SLA_Days",1,365,3)
            weekday=x.selectbox("Submission_Weekday",["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
            submit=st.form_submit_button("Predict")
        if submit:
            row=make_prediction_row({"Service_Type":service,"Requester_Type":requester,"Requested_Channel":channel,"Priority":priority,"Assigned_Office":office,"Queue_Length":queue,"Required_Documents_Complete":docs,"SLA_Days":sla,"Submission_Weekday":weekday})
            pred,prob=predict(saved,row)
            if pred: st.error("Predicted result: SLA BREACH RISK — staff review recommended.")
            else: st.success("Predicted result: NO SLA BREACH — continue monitoring.")
            if prob is not None: st.metric("Estimated SLA breach probability",f"{prob:.1%}")
            st.write("Prediction timestamp:",datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            st.info("Staff must verify the request and record the final decision. Do not automatically deny or reject a service.")
    else: st.warning("Upload CSV in Dataset, then train and save the model. No trained artifact was supplied with the uploaded files.")

elif page=="Dashboard":
    st.header("4. Decision-Support Dashboard")
    df=current_data()
    if df is None: st.warning("Upload the dataset first.")
    else:
        _,md=prepare_model_data(df); k=summary(df,md); cols=st.columns(4)
        for col,(name,val) in zip(cols,k.items()): col.metric(name,"N/A" if val is None else (f"{val:.1f}%" if isinstance(val,float) else val))
        st.subheader("Historical breach review list")
        st.caption("This is a historical completed-record analysis, not a live queue of pending requests.")
        if len(md):
            risk=md[md.SLA_Breached==1][["Request_ID","Service_Type","Assigned_Office","Queue_Length","SLA_Days","Processing_Days"]].sort_values("Queue_Length",ascending=False)
            st.dataframe(risk,use_container_width=True)
            st.download_button("Download review list",risk.to_csv(index=False),"sla_breach_review.csv","text/csv")
        st.subheader("Recommended staff follow-up")
        st.markdown("1. Verify request status and required documents.\n2. Review queue length and office workload.\n3. Contact the student if details are incomplete.\n4. Record staff review and follow-up.\n5. Apply office policy; do not rely on model output alone.")

elif page=="System Design":
    st.header("5. Systems Analysis and Design")
    st.subheader("Actors")
    st.markdown("- **Student Services Administrator:** uploads data, checks model metrics, reviews dashboard and access.\n- **Student Services Staff:** reviews flagged requests, verifies details, and records follow-up.")
    st.subheader("Five functional requirements")
    st.markdown("1. Validate uploaded CSV schema and show data quality.\n2. Display previews, descriptive statistics, class distribution, and charts.\n3. Train and save the midterm Decision Tree pipeline.\n4. Accept the ten model features and predict SLA breach risk.\n5. Display workload KPIs and historical review records.")
    st.subheader("Three non-functional requirements")
    st.markdown("1. Usability: clear navigation and readable results.\n2. Reliability: handle invalid or incomplete files with clear errors.\n3. Privacy: use synthetic or authorized records and protect student data.")
    st.image("assets/use_case_diagram.png",use_container_width=True)
    st.subheader("Decision workflow")
    st.graphviz_chart('digraph G {rankdir=LR; node [shape=box,style=rounded]; A[label="Upload CSV"]; B[label="Validate data"]; C[label="Prepare completed records"]; D[label="Train/load model"]; E[label="Predict risk"]; F[label="Staff reviews evidence"]; G[label="Record decision and follow-up"]; A->B->C->D->E->F->G;}')
    st.subheader("Prediction checkpoint and limitations")
    st.write("The label is created for completed requests only: Processing_Days > SLA_Days. The ten intake predictors are the features specified in the midterm notebook. Processing_Days, Status, and SLA_Breached are not prediction inputs. Historical model performance may differ on future requests; authorized staff make final decisions.")
