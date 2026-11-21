import time
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from backend.inference_engine import MultiModalScreeningEngine

app = FastAPI(title="Diabetic Microvascular Screening Portal", version="2.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

engine = MultiModalScreeningEngine()


@app.get("/")
def serve_dashboard():
    return FileResponse("static/index.html")


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "device": engine.device.type.upper(),
        "dr_weights_loaded": engine.dr_loaded,
        "dfu_weights_loaded": engine.dfu_loaded,
    }


@app.post("/screen")
async def screen_patient(
    retinal_scan: Optional[UploadFile] = File(None),
    foot_scan: Optional[UploadFile] = File(None),
):
    if retinal_scan is None and foot_scan is None:
        raise HTTPException(status_code=400, detail="Please upload a retinal scan, a foot scan, or both.")

    start_time = time.time()
    response = {}
    risks = []

    if retinal_scan is not None:
        dr_result = engine.analyze_retina(await retinal_scan.read())
        response["retina"] = dr_result
        risks.append(dr_result["risk_score"])

    if foot_scan is not None:
        dfu_result = engine.analyze_foot(await foot_scan.read())
        response["foot"] = dfu_result
        risks.append(dfu_result["risk_score"])

    combined_risk = round(max(risks) * 0.7 + (sum(risks) / len(risks)) * 0.3, 1)

    if combined_risk >= 65:
        triage_level = "HIGH RISK"
        recommendation = "Urgent referral to Ophthalmology / Diabetic Foot Clinic within 48 hours."
    elif combined_risk >= 30:
        triage_level = "MODERATE RISK"
        recommendation = "Schedule specialist follow-up within 2–4 weeks; reinforce glycemic & offloading care."
    else:
        triage_level = "LOW RISK"
        recommendation = "No acute microvascular lesions detected. Continue routine annual diabetic screening."

    response["overall_risk_score"] = combined_risk
    response["triage_level"] = triage_level
    response["recommendation"] = recommendation
    response["inference_ms"] = round((time.time() - start_time) * 1000)
    return response