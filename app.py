from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import shap
import math

app = Flask(__name__)

# =========================
# LOAD DATA AND MODELS
# =========================

customer_db = pd.read_csv("customers.csv")

model = pickle.load(open("default_model.pkl", "rb"))
reg_model = pickle.load(open("loan_model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

explainer = shap.TreeExplainer(model)

# =========================
# CONSTANTS
# =========================

INTEREST_RATE = 0.12
ANNUAL_HIKE_RATE = 0.08
SAFE_EMI_RATIO = 0.40

AVERAGE_CAREER_YEARS = 33
MIN_TENURE_MONTHS = 12


# =========================
# HELPER FUNCTIONS
# =========================

def get_float_value(field_name, default=0):
    value = request.form.get(field_name, default)

    if value == "" or value is None:
        return float(default)

    return float(value)


def calculate_required_emi(loan_amount, rate=INTEREST_RATE, months=120):
    r = rate / 12

    if loan_amount <= 0 or months <= 0:
        return 0

    if r == 0:
        return loan_amount / months

    emi = loan_amount * r * ((1 + r) ** months) / (((1 + r) ** months) - 1)

    return emi


def calculate_loan_amount(emi, rate=INTEREST_RATE, months=120):
    r = rate / 12

    if emi <= 0 or months <= 0:
        return 0

    if r == 0:
        return emi * months

    loan_amount = emi * (1 - (1 + r) ** (-months)) / r

    return loan_amount


def calculate_repayment_months(loan_amount, emi, rate=INTEREST_RATE):
    r = rate / 12

    if loan_amount <= 0 or emi <= 0:
        return None

    monthly_interest = loan_amount * r

    if emi <= monthly_interest:
        return None

    months = -math.log(1 - (loan_amount * r / emi)) / math.log(1 + r)

    return math.ceil(months)


def calculate_remaining_work_years(emp_length):
    remaining_years = AVERAGE_CAREER_YEARS - emp_length

    if remaining_years < 1:
        remaining_years = 1

    return remaining_years


def average_projected_annual_income(annual_income, years):
    years = int(math.ceil(years))

    if years <= 0:
        return annual_income

    total_income = 0

    for year in range(years):
        total_income += annual_income * ((1 + ANNUAL_HIKE_RATE) ** year)

    avg_income = total_income / years

    return avg_income


def risk_category(risk):
    if risk < 0.30:
        return "Low"

    elif risk < 0.60:
        return "Medium"

    else:
        return "High"


def create_explanation(
    full_loan_possible,
    tenure_needed_for_requested_loan,
    max_allowed_tenure,
    total_emi_ratio,
    risk,
    data
):
    explanation = []

    if full_loan_possible:
        explanation.append(
            "Requested loan can be repaid within the remaining working period"
        )
    else:
        if tenure_needed_for_requested_loan is None:
            explanation.append(
                "Entered EMI is not enough to repay the requested loan"
            )
        else:
            explanation.append(
                "Requested loan tenure exceeds the remaining working period"
            )

    if total_emi_ratio > 0.60:
        explanation.append(
            "Total EMI burden is very high compared to monthly income"
        )

    elif total_emi_ratio > 0.50:
        explanation.append(
            "Total EMI burden is moderately high because of current and existing EMI commitments"
        )

    if risk > 0.60:
        explanation.append(
            "Risk score is high, so the application requires manual review"
        )

    elif risk > 0.45:
        explanation.append(
            "Risk score is medium, so the system applies cautious approval logic"
        )

    if data["delinq_2yrs"] == 0 and data["pub_rec"] == 0:
        explanation.append(
            "Good credit history helped reduce risk"
        )
    else:
        explanation.append(
            "Credit record contains public record or past credit issue"
        )

    if data["has_existing_loan"] == 1:
        explanation.append(
            "Existing loan EMI was considered while calculating repayment capacity"
        )

    explanation.append(
        "Annual income was projected with 8% yearly hike for EMI capacity calculation"
    )

    return explanation


# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return render_template(
        "index.html",
        customer=None
    )


@app.route("/fetch_pan", methods=["POST"])
def fetch_pan():
    pan = request.form["pan"].upper().strip()

    customer = customer_db[
        customer_db["pan"] == pan
    ]

    if customer.empty:
        return render_template(
            "index.html",
            error="PAN not found",
            customer=None
        )

    customer = customer.iloc[0].to_dict()

    return render_template(
        "index.html",
        customer=customer
    )


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # =========================
        # EXISTING LOAN INPUT
        # =========================

        has_existing_loan = request.form.get("has_existing_loan", "no")

        if has_existing_loan == "yes":
            existing_loan_amount = get_float_value("existing_loan_amount")
            existing_loan_remaining = get_float_value("existing_loan_remaining")
            existing_loan_emi = get_float_value("existing_loan_emi")
            existing_loan_tenure = get_float_value("existing_loan_tenure")
        else:
            existing_loan_amount = 0
            existing_loan_remaining = 0
            existing_loan_emi = 0
            existing_loan_tenure = 0

        # =========================
        # MAIN INPUT DATA
        # =========================

        data = {
            "loan_amnt": get_float_value("loan_amnt"),
            "annual_inc": get_float_value("annual_inc"),
            "monthly_debt": get_float_value("monthly_debt"),
            "emp_length": get_float_value("emp_length"),

            "revol_util": 35,
            "total_acc": 15,

            "installment": get_float_value("installment"),
            "savings": get_float_value("savings"),
            "collateral": int(request.form.get("collateral", 0)),

            "delinq_2yrs": get_float_value("delinq_2yrs"),
            "pub_rec": get_float_value("pub_rec"),

            "has_existing_loan": 1 if has_existing_loan == "yes" else 0,
            "existing_loan_amount": existing_loan_amount,
            "existing_loan_remaining": existing_loan_remaining,
            "existing_loan_emi": existing_loan_emi,
            "existing_loan_tenure": existing_loan_tenure
        }

    except ValueError:
        return render_template(
            "index.html",
            error="Please enter valid numeric values",
            customer=None
        )

    # =========================
    # BASIC VALIDATION
    # =========================

    if data["annual_inc"] <= 0:
        return render_template(
            "index.html",
            error="Annual income must be greater than 0",
            customer=None
        )

    monthly_income = data["annual_inc"] / 12

    if data["monthly_debt"] < 0:
        return render_template(
            "index.html",
            error="Monthly debt cannot be negative",
            customer=None
        )

    if data["monthly_debt"] > monthly_income:
        return render_template(
            "index.html",
            error="Monthly debt exceeds monthly income",
            customer=None
        )

    if data["loan_amnt"] <= 0:
        return render_template(
            "index.html",
            error="Loan amount must be greater than 0",
            customer=None
        )

    if data["installment"] <= 0:
        return render_template(
            "index.html",
            error="EMI must be greater than 0",
            customer=None
        )

    if data["emp_length"] < 0 or data["emp_length"] > AVERAGE_CAREER_YEARS:
        return render_template(
            "index.html",
            error="Employment experience must be between 0 and 33 years",
            customer=None
        )

    if data["delinq_2yrs"] < 0 or data["pub_rec"] < 0:
        return render_template(
            "index.html",
            error="Credit history values cannot be negative",
            customer=None
        )

    if data["savings"] < 0:
        return render_template(
            "index.html",
            error="Savings cannot be negative",
            customer=None
        )

    # =========================
    # EXISTING LOAN VALIDATION
    # =========================

    if has_existing_loan == "yes":

        if data["existing_loan_amount"] <= 0:
            return render_template(
                "index.html",
                error="Existing loan amount must be greater than 0",
                customer=None
            )

        if data["existing_loan_remaining"] <= 0:
            return render_template(
                "index.html",
                error="Existing loan remaining amount must be greater than 0",
                customer=None
            )

        if data["existing_loan_remaining"] > data["existing_loan_amount"]:
            return render_template(
                "index.html",
                error="Existing loan remaining amount cannot exceed existing loan amount",
                customer=None
            )

        if data["existing_loan_emi"] <= 0:
            return render_template(
                "index.html",
                error="Existing loan EMI must be greater than 0",
                customer=None
            )

        if data["existing_loan_tenure"] <= 0:
            return render_template(
                "index.html",
                error="Existing loan tenure must be greater than 0",
                customer=None
            )

    # =========================
    # CAREER AND TENURE LOGIC
    # =========================

    remaining_work_years = calculate_remaining_work_years(
        data["emp_length"]
    )

    max_allowed_tenure = int(remaining_work_years * 12)

    if max_allowed_tenure < MIN_TENURE_MONTHS:
        max_allowed_tenure = MIN_TENURE_MONTHS

    avg_projected_annual_income = average_projected_annual_income(
        data["annual_inc"],
        remaining_work_years
    )

    avg_projected_monthly_income = avg_projected_annual_income / 12

    # =========================
    # EMI CAPACITY
    # =========================

    safe_emi_capacity = (
        SAFE_EMI_RATIO * avg_projected_monthly_income
    ) - data["existing_loan_emi"]

    safe_emi_capacity = max(
        0,
        safe_emi_capacity
    )

    emi_considered = min(
        data["installment"],
        safe_emi_capacity
    )

    # =========================
    # FEATURE ENGINEERING
    # =========================

    dti = (
        data["monthly_debt"] /
        (monthly_income + 1e-6)
    ) * 100

    current_emi_ratio = (
        data["installment"] /
        (monthly_income + 1e-6)
    )

    total_emi_ratio = (
        (data["installment"] + data["existing_loan_emi"]) /
        (monthly_income + 1e-6)
    )

    projected_total_emi_ratio = (
        (data["installment"] + data["existing_loan_emi"]) /
        (avg_projected_monthly_income + 1e-6)
    )

    existing_loan_burden = (
        data["existing_loan_remaining"] /
        (data["annual_inc"] + 1e-6)
    )

    df = pd.DataFrame([data])

    df["monthly_income"] = monthly_income
    df["dti"] = dti
    df["dti_normalized"] = dti / 100
    df["credit_utilization"] = df["revol_util"] / 100
    df["emi_to_income"] = current_emi_ratio
    df["savings_ratio"] = df["savings"] / (df["loan_amnt"] + 1e-6)

    df["total_emi_ratio"] = total_emi_ratio
    df["existing_loan_burden"] = existing_loan_burden
    df["avg_projected_income"] = avg_projected_annual_income
    df["remaining_work_years"] = remaining_work_years

    for col in features:
        if col not in df.columns:
            df[col] = 0

    df = df[features]

    # =========================
    # ML PREDICTION
    # =========================

    risk = model.predict_proba(df)[0][1]

    emi_log = reg_model.predict(df)[0]
    predicted_emi = np.expm1(emi_log)

    # =========================
    # RISK ADJUSTMENT
    # =========================

    if total_emi_ratio > 0.75:
        risk += 0.20

    elif total_emi_ratio > 0.60:
        risk += 0.12

    elif total_emi_ratio > 0.50:
        risk += 0.06

    if has_existing_loan == "yes":

        if existing_loan_burden > 1.00:
            risk += 0.10

        elif existing_loan_burden > 0.50:
            risk += 0.05

    if projected_total_emi_ratio < current_emi_ratio:
        risk -= 0.03

    risk = max(
        0,
        min(risk, 1.0)
    )

    # =========================
    # LOAN AND TENURE CALCULATION
    # =========================

    required_emi_for_requested_loan = calculate_required_emi(
        data["loan_amnt"],
        months=max_allowed_tenure
    )

    emi_shortfall = max(
        0,
        required_emi_for_requested_loan - data["installment"]
    )

    tenure_needed_for_requested_loan = calculate_repayment_months(
        data["loan_amnt"],
        emi_considered
    )

    full_loan_possible = (
        tenure_needed_for_requested_loan is not None and
        tenure_needed_for_requested_loan <= max_allowed_tenure
    )

    if full_loan_possible:
        recommended_loan = data["loan_amnt"]
    else:
        recommended_loan = calculate_loan_amount(
            emi_considered,
            months=max_allowed_tenure
        )

        recommended_loan = min(
            recommended_loan,
            data["loan_amnt"]
        )

    final_tenure = calculate_repayment_months(
        recommended_loan,
        emi_considered
    )

    if final_tenure is None:
        final_tenure = max_allowed_tenure

    # =========================
    # SAFE RECOMMENDED AMOUNT
    # =========================

    safe_recommended_loan = calculate_loan_amount(
        emi_considered,
        months=max_allowed_tenure
    )

    safe_recommended_loan = min(
        safe_recommended_loan,
        data["loan_amnt"]
    )

    safe_loan_lower = safe_recommended_loan * 0.90

    safe_loan_upper = min(
        safe_recommended_loan * 1.10,
        data["loan_amnt"]
    )

    safe_percentage = (
        safe_recommended_loan /
        data["loan_amnt"]
    ) * 100

    # =========================
    # RANGE CALCULATION
    # =========================

    approval_percentage = (
        recommended_loan /
        data["loan_amnt"]
    ) * 100

    lower_range = recommended_loan * 0.90

    upper_range = min(
        recommended_loan * 1.10,
        data["loan_amnt"]
    )

    # =========================
    # FINAL DECISION LOGIC
    # =========================

    if risk > 0.75 or emi_considered <= 0:
        decision = "Rejected"

    elif full_loan_possible:

        if risk <= 0.45 and total_emi_ratio <= 0.50:
            decision = "Approved"

        elif risk <= 0.60 and total_emi_ratio <= 0.60:

            if data["collateral"] == 1:
                decision = "Secured Approval"

            else:
                decision = "Approved with Conditions"

        else:

            if data["collateral"] == 1:
                decision = "Secured Manual Review"

            else:
                decision = "Manual Review Required"

    else:

        if approval_percentage >= 50:
            decision = "Partially Approved"

        elif approval_percentage >= 25:
            decision = "Manual Review Required"

        else:
            decision = "Rejected"

    category = risk_category(risk)

    # =========================
    # EXPLANATION
    # =========================

    explanation = create_explanation(
        full_loan_possible,
        tenure_needed_for_requested_loan,
        max_allowed_tenure,
        total_emi_ratio,
        risk,
        data
    )

    # =========================
    # RESULT PAGE
    # =========================

    return render_template(
        "result.html",

        decision=decision,
        risk=round(risk, 3),
        category=category,

        requested_loan=round(data["loan_amnt"], 2),
        recommended_loan=round(recommended_loan, 2),
        loan_lower=round(lower_range, 2),
        loan_upper=round(upper_range, 2),
        approval_percentage=round(approval_percentage, 2),

        safe_recommended_loan=round(safe_recommended_loan, 2),
        safe_loan_lower=round(safe_loan_lower, 2),
        safe_loan_upper=round(safe_loan_upper, 2),
        safe_percentage=round(safe_percentage, 2),

        user_emi=round(data["installment"], 2),
        emi_considered=round(emi_considered, 2),
        predicted_emi=round(predicted_emi, 2),
        required_emi=round(required_emi_for_requested_loan, 2),
        emi_shortfall=round(emi_shortfall, 2),

        emp_length=round(data["emp_length"], 1),
        remaining_work_years=round(remaining_work_years, 1),
        max_allowed_tenure=max_allowed_tenure,
        tenure_needed=tenure_needed_for_requested_loan,
        final_tenure=final_tenure,

        annual_hike_rate=ANNUAL_HIKE_RATE * 100,
        avg_projected_annual_income=round(avg_projected_annual_income, 2),
        avg_projected_monthly_income=round(avg_projected_monthly_income, 2),

        has_existing_loan=has_existing_loan,
        existing_loan_amount=round(existing_loan_amount, 2),
        existing_loan_remaining=round(existing_loan_remaining, 2),
        existing_loan_emi=round(existing_loan_emi, 2),
        existing_loan_tenure=round(existing_loan_tenure, 2),
        total_emi_ratio=round(total_emi_ratio * 100, 2),

        explanation=explanation
    )


if __name__ == "__main__":
    app.run(debug=True)