from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import joblib

app = FastAPI(title="MSME Scheme Eligibility API")

# ---------------- CORS ---------------- #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LOAD MODEL ---------------- #

try:
    model = joblib.load("scheme_model.pkl")
    scaler = joblib.load("scheme_scaler.pkl")
    MODEL_AVAILABLE = True
    print("Model loaded successfully")
except Exception as e:
    print("Model loading failed:", e)
    MODEL_AVAILABLE = False


# ---------------- INPUT MODEL ---------------- #

class EnterpriseInput(BaseModel):
    investment_amount: float
    annual_turnover: float
    business_type: int
    sector: int
    years_in_business: float
    number_of_employees: int
    udyam_registered: int
    gst_registered: int
    gender: int
    social_category: int
    minority_status: int
    disability_status: int
    age: int
    rural_urban: int
    state: int
    aspirational_district: int
    north_east_region: int
    exporter: int
    startup_dpiit: int
    green_business: int
    women_owned: int


# ---------------- MSME CLASSIFICATION ---------------- #

def classify_msme(investment: float, turnover: float):

    if investment <= 1e7 and turnover <= 5e7:
        return "Micro"

    elif investment <= 1e8 and turnover <= 5e8:
        return "Small"

    elif investment <= 5e8 and turnover <= 2.5e9:
        return "Medium"

    else:
        return "Not Classified"


# ---------------- RULE BASED SCHEME ENGINE ---------------- #

def recommend_schemes(data: EnterpriseInput) -> List[Dict]:

    schemes = []

    msme_category = classify_msme(
        data.investment_amount,
        data.annual_turnover
    )

    if msme_category == "Micro":
        schemes.append({
            "scheme": "PM Mudra Yojana",
            "reason": "Eligible because enterprise falls under Micro category."
        })

    if data.social_category in [1, 2] or data.women_owned == 1:
        schemes.append({
            "scheme": "Stand-Up India",
            "reason": "Eligible under SC/ST or Women entrepreneur criteria."
        })

    if data.udyam_registered == 1:
        schemes.append({
            "scheme": "CGTMSE",
            "reason": "Udyam registered MSMEs are eligible for credit guarantee coverage."
        })

    if data.startup_dpiit == 1:
        schemes.append({
            "scheme": "Startup India Seed Fund Scheme",
            "reason": "Recognized DPIIT startup."
        })

    if data.exporter == 1:
        schemes.append({
            "scheme": "Export Promotion Capital Goods Scheme",
            "reason": "Exporter MSMEs are eligible for capital goods subsidy."
        })

    if data.green_business == 1:
        schemes.append({
            "scheme": "Credit Linked Capital Subsidy Scheme",
            "reason": "Green / Sustainable businesses qualify for capital subsidy."
        })

    if data.north_east_region == 1:
        schemes.append({
            "scheme": "North East Industrial Development Scheme",
            "reason": "Enterprise located in North-East region."
        })

    if data.aspirational_district == 1:
        schemes.append({
            "scheme": "Aspirational District Programme Benefits",
            "reason": "Enterprise located in notified aspirational district."
        })

    if data.women_owned == 1:
        schemes.append({
            "scheme": "Mahila Udyam Nidhi Scheme",
            "reason": "Women-owned enterprise."
        })

    if not schemes:
        schemes.append({
            "scheme": "General MSME Development Programme",
            "reason": "Eligible under general MSME category."
        })

    return schemes


# ---------------- HEALTH CHECK ---------------- #

@app.get("/")
def health_check():
    return {"status": "MSME Scheme Eligibility Service Running"}


# ---------------- MAIN PREDICTION API ---------------- #

@app.post("/predict")
def predict(data: EnterpriseInput):

    msme_category = classify_msme(
        data.investment_amount,
        data.annual_turnover
    )

    eligible_schemes = recommend_schemes(data)

    ml_prediction = None

    if MODEL_AVAILABLE:

        try:

            # ---------- FEATURE ENGINEERING ---------- #

            investment_turnover_ratio = data.investment_amount / (data.annual_turnover + 1)

            turnover_per_employee = data.annual_turnover / (data.number_of_employees + 1)

            business_maturity_score = (
                3 if data.years_in_business > 10
                else 2 if data.years_in_business > 5
                else 1
            )

            compliance_score = data.udyam_registered + data.gst_registered

            demographic_advantage_score = (
                (1 if data.gender == 1 else 0) +
                (1 if data.social_category > 0 else 0) +
                data.minority_status
            )

            regional_priority_score = (
                data.north_east_region +
                data.aspirational_district +
                data.rural_urban
            )

            capital_intensity_score = data.investment_amount / (data.number_of_employees + 1)

            enterprise_scale_index = (
                data.annual_turnover * 0.6 +
                data.investment_amount * 0.4
            )

            # ---------- FINAL MODEL INPUT (27 FEATURES) ---------- #

            input_data = np.array([[

                data.investment_amount,
                data.annual_turnover,
                data.years_in_business,
                data.number_of_employees,
                data.udyam_registered,
                data.gst_registered,
                data.gender,
                data.social_category,
                data.minority_status,
                data.disability_status,
                data.age,
                data.rural_urban,
                data.state,
                data.aspirational_district,
                data.north_east_region,
                data.exporter,
                data.startup_dpiit,
                data.green_business,
                data.women_owned,

                investment_turnover_ratio,
                turnover_per_employee,
                business_maturity_score,
                compliance_score,
                demographic_advantage_score,
                regional_priority_score,
                capital_intensity_score,
                enterprise_scale_index

            ]])

            input_scaled = scaler.transform(input_data)

            ml_prediction = model.predict(input_scaled).tolist()

        except Exception as e:

            print("Prediction error:", e)
            ml_prediction = None


    return {
        "msme_category": msme_category,
        "eligible_schemes": eligible_schemes,
        "ml_model_output": ml_prediction,
        "message": "Scheme eligibility calculated successfully."
    }
