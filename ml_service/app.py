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
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- OPTIONAL ML MODEL LOAD ---------------- #
# If you train a multi-label model, keep these.
# Otherwise you can remove model/scaler related lines.

import os

try:
    model_path = os.path.join(os.path.dirname(__file__), "scheme_model.pkl")
    scaler_path = os.path.join(os.path.dirname(__file__), "scheme_scaler.pkl")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    MODEL_AVAILABLE = True
except Exception as e:
    print("Model loading failed:", e)
    MODEL_AVAILABLE = False


# ---------------- FEATURE LIST ---------------- #
FEATURES = [
    "investment_amount",
    "annual_turnover",
    "business_type",
    "sector",
    "years_in_business",
    "number_of_employees",
    "udyam_registered",
    "gst_registered",
    "gender",
    "social_category",
    "minority_status",
    "disability_status",
    "age",
    "rural_urban",
    "state",
    "aspirational_district",
    "north_east_region",
    "exporter",
    "startup_dpiit",
    "green_business",
    "women_owned"
]


# ---------------- INPUT MODEL ---------------- #
class EnterpriseInput(BaseModel):
    investment_amount: float
    annual_turnover: float
    business_type: int        # 0=Manufacturing, 1=Service, 2=Trading
    sector: int               # Encoded sector ID
    years_in_business: float
    number_of_employees: int
    udyam_registered: int     # 0 or 1
    gst_registered: int       # 0 or 1
    gender: int               # 0=Male, 1=Female, 2=Other
    social_category: int      # 0=General,1=SC,2=ST,3=OBC
    minority_status: int      # 0 or 1
    disability_status: int    # 0 or 1
    age: int
    rural_urban: int          # 0=Urban,1=Rural
    state: int                # Encoded state ID
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


# ---------------- RULE-BASED SCHEME ENGINE ---------------- #
def recommend_schemes(data: EnterpriseInput) -> List[Dict]:

    schemes = []

    msme_category = classify_msme(
        data.investment_amount,
        data.annual_turnover
    )

    # PM Mudra Yojana
    if msme_category == "Micro":
        schemes.append({
            "scheme": "PM Mudra Yojana",
            "reason": "Eligible because enterprise falls under Micro category."
        })

    # Stand-Up India
    if data.social_category in [1, 2] or data.women_owned == 1:
        schemes.append({
            "scheme": "Stand-Up India",
            "reason": "Eligible under SC/ST or Women entrepreneur criteria."
        })

    # CGTMSE
    if data.udyam_registered == 1:
        schemes.append({
            "scheme": "CGTMSE",
            "reason": "Udyam registered MSMEs are eligible for credit guarantee coverage."
        })

    # Startup India Seed Fund
    if data.startup_dpiit == 1:
        schemes.append({
            "scheme": "Startup India Seed Fund Scheme",
            "reason": "Recognized DPIIT startup."
        })

    # Export Promotion Scheme
    if data.exporter == 1:
        schemes.append({
            "scheme": "Export Promotion Capital Goods Scheme",
            "reason": "Exporter MSMEs are eligible for capital goods subsidy."
        })

    # Green Scheme
    if data.green_business == 1:
        schemes.append({
            "scheme": "Credit Linked Capital Subsidy Scheme",
            "reason": "Green / Sustainable businesses qualify for capital subsidy."
        })

    # North East Scheme
    if data.north_east_region == 1:
        schemes.append({
            "scheme": "North East Industrial Development Scheme",
            "reason": "Enterprise located in North-East region."
        })

    # Aspirational District Scheme
    if data.aspirational_district == 1:
        schemes.append({
            "scheme": "Aspirational District Programme Benefits",
            "reason": "Enterprise located in notified aspirational district."
        })

    # Women Entrepreneur Scheme
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
        input_data = np.array([[ 
            data.investment_amount,
            data.annual_turnover,
            data.business_type,
            data.sector,
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
            data.women_owned
        ]])

        input_scaled = scaler.transform(input_data)
        ml_prediction = model.predict(input_scaled).tolist()

    return {
        "msme_category": msme_category,
        "eligible_schemes": eligible_schemes,
        "ml_model_output": ml_prediction,
        "message": "Scheme eligibility calculated successfully."
    }
