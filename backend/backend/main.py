"""Lens — FastAPI server: data APIs, tool analysis, chat SSE, static serving."""

import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Depends, Query, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from database import get_db, engine, Base
from models import (
    Patient, VitalSign, LabResult, ClinicalNote, MedicationLog,
    PatientOut, VitalSignOut, LabResultOut, ClinicalNoteOut, MedicationLogOut,
    ToolAnalysis, ChatRequest,
)
from agent import analyze_tool, stream_chat, stream_website

app = FastAPI(title="Lens — AI Healthcare Platform", version="0.1.0")

GENERATED_DIR = Path(__file__).parent / "generated" / "site"
STATIC_DIR = Path(__file__).parent / "static"

# Ensure directories exist
GENERATED_DIR.mkdir(parents=True, exist_ok=True)


# ── Data APIs ───────────────────────────────────────────────────────────────

@app.get("/api/patient/{patient_id}", response_model=PatientOut)
def get_patient(patient_id: str, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(404, "Patient not found")
    return patient


@app.get("/api/patient/{patient_id}/vitals", response_model=list[VitalSignOut])
def get_vitals(patient_id: str, vital_type: Optional[str] = None, limit: int = Query(500, le=2000), db: Session = Depends(get_db)):
    q = db.query(VitalSign).filter(VitalSign.patient_id == patient_id)
    if vital_type:
        q = q.filter(VitalSign.vital_type == vital_type)
    return q.order_by(VitalSign.recorded_at.desc()).limit(limit).all()


@app.get("/api/patient/{patient_id}/labs", response_model=list[LabResultOut])
def get_labs(patient_id: str, test_name: Optional[str] = None, limit: int = Query(500, le=2000), db: Session = Depends(get_db)):
    q = db.query(LabResult).filter(LabResult.patient_id == patient_id)
    if test_name:
        q = q.filter(LabResult.test_name == test_name)
    return q.order_by(LabResult.recorded_at.desc()).limit(limit).all()


@app.get("/api/patient/{patient_id}/notes", response_model=list[ClinicalNoteOut])
def get_notes(patient_id: str, limit: int = Query(100, le=500), db: Session = Depends(get_db)):
    return (
        db.query(ClinicalNote)
        .filter(ClinicalNote.patient_id == patient_id)
        .order_by(ClinicalNote.encounter_date.desc())
        .limit(limit)
        .all()
    )


@app.get("/api/patient/{patient_id}/medications", response_model=list[MedicationLogOut])
def get_medications(patient_id: str, db: Session = Depends(get_db)):
    return (
        db.query(MedicationLog)
        .filter(MedicationLog.patient_id == patient_id)
        .order_by(MedicationLog.start_date.desc())
        .all()
    )


@app.get("/api/patient/{patient_id}/weights", response_model=list[VitalSignOut])
def get_weights(patient_id: str, limit: int = Query(200, le=1000), db: Session = Depends(get_db)):
    return (
        db.query(VitalSign)
        .filter(VitalSign.patient_id == patient_id, VitalSign.vital_type == "weight")
        .order_by(VitalSign.recorded_at.desc())
        .limit(limit)
        .all()
    )


# ── Tool Analysis ───────────────────────────────────────────────────────────

@app.post("/api/analyze-tool")
async def analyze_tool_endpoint(body: dict):
    description = body.get("description", "")
    if not description.strip():
        raise HTTPException(400, "Description is required")
    result = await analyze_tool(description)
    return result


# ── Helpers ─────────────────────────────────────────────────────────────────

def _build_context(req_patient_id: str, req_tools: list, db: Session):
    """Build patient_context and tools_context strings from request data."""
    patient = db.query(Patient).filter(Patient.id == req_patient_id).first()
    if not patient:
        raise HTTPException(404, "Patient not found")

    vitals_count = db.query(VitalSign).filter(VitalSign.patient_id == req_patient_id).count()
    labs_count = db.query(LabResult).filter(LabResult.patient_id == req_patient_id).count()
    notes_count = db.query(ClinicalNote).filter(ClinicalNote.patient_id == req_patient_id).count()
    meds_count = db.query(MedicationLog).filter(MedicationLog.patient_id == req_patient_id).count()

    lab_tests = [
        r[0] for r in
        db.query(LabResult.test_name).filter(LabResult.patient_id == req_patient_id).distinct().all()
    ]
    vital_types = [
        r[0] for r in
        db.query(VitalSign.vital_type).filter(VitalSign.patient_id == req_patient_id).distinct().all()
    ]

    patient_context = (
        f"Patient: {patient.first_name} {patient.last_name}, "
        f"{patient.gender}, born {patient.birth_date}, "
        f"from {patient.city}, {patient.state}. "
        f"Primary diagnosis: {patient.primary_diagnosis or 'N/A'}.\n"
        f"Available data: {vitals_count} vital signs ({', '.join(vital_types)}), "
        f"{labs_count} lab results ({', '.join(lab_tests)}), "
        f"{notes_count} clinical notes, {meds_count} medications."
    )

    tools_context = ""
    if req_tools:
        tools_lines = []
        for t in req_tools:
            cats = ", ".join(t.get("data_categories", []))
            tools_lines.append(f"- {t.get('name', 'Unknown')}: {cats}")
        tools_context = "The user has configured the following tools:\n" + "\n".join(tools_lines)

    return patient_context, tools_context


# ── Chat (Conversational Planning) ─────────────────────────────────────────

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest, db: Session = Depends(get_db)):
    patient_context, tools_context = _build_context(req.patient_id, req.tools, db)

    async def event_generator():
        async for chunk in stream_chat(
            message=req.message,
            tools_context=tools_context,
            patient_context=patient_context,
            history=req.history,
        ):
            yield {"event": "chunk", "data": json.dumps({"text": chunk})}
        yield {"event": "done", "data": json.dumps({"message": "done"})}

    return EventSourceResponse(event_generator())


# ── Generate Website (after plan approval) ──────────────────────────────────

@app.post("/api/generate")
async def generate_endpoint(req: ChatRequest, db: Session = Depends(get_db)):
    patient_context, tools_context = _build_context(req.patient_id, req.tools, db)

    # Build plan context from conversation history
    plan_parts = []
    for h in req.history:
        plan_parts.append(f"{h['role'].upper()}: {h['content']}")
    plan_context = "\n\n".join(plan_parts[-10:])

    # Gather sample data so the generator knows the exact JSON shapes
    pid = req.patient_id
    patient_obj = db.query(Patient).filter(Patient.id == pid).first()
    sample_patient = {
        "id": patient_obj.id, "first_name": patient_obj.first_name,
        "last_name": patient_obj.last_name, "birth_date": str(patient_obj.birth_date),
        "gender": patient_obj.gender, "race": patient_obj.race,
        "city": patient_obj.city, "state": patient_obj.state,
        "primary_diagnosis": patient_obj.primary_diagnosis,
    } if patient_obj else {}

    vital = db.query(VitalSign).filter(VitalSign.patient_id == pid).first()
    sample_vital = {"recorded_at": str(vital.recorded_at), "vital_type": vital.vital_type,
                    "value": vital.value, "unit": vital.unit} if vital else {}

    lab = db.query(LabResult).filter(LabResult.patient_id == pid).first()
    sample_lab = {"recorded_at": str(lab.recorded_at), "test_name": lab.test_name,
                  "value": lab.value, "unit": lab.unit, "loinc_code": lab.loinc_code} if lab else {}

    note = db.query(ClinicalNote).filter(ClinicalNote.patient_id == pid).first()
    sample_note = {"encounter_date": str(note.encounter_date), "encounter_class": note.encounter_class,
                   "description": note.description, "reason": note.reason} if note else {}

    med = db.query(MedicationLog).filter(MedicationLog.patient_id == pid).first()
    sample_med = {"start_date": str(med.start_date), "stop_date": str(med.stop_date) if med.stop_date else None,
                  "description": med.description, "reason": med.reason} if med else {}

    weight = db.query(VitalSign).filter(VitalSign.patient_id == pid, VitalSign.vital_type == "weight").first()
    sample_weight = {"recorded_at": str(weight.recorded_at), "vital_type": "weight",
                     "value": weight.value, "unit": weight.unit} if weight else {}

    sample_data = {
        "patient": sample_patient, "vital": sample_vital, "lab": sample_lab,
        "note": sample_note, "med": sample_med, "weight": sample_weight,
    }

    async def event_generator():
        html_parts = []
        async for chunk in stream_website(
            tools_context=tools_context,
            patient_context=patient_context,
            patient_id=req.patient_id,
            plan_context=plan_context,
            sample_data=sample_data,
        ):
            html_parts.append(chunk)
            yield {"event": "chunk", "data": json.dumps({"text": chunk})}

        full_html = "".join(html_parts)
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        (GENERATED_DIR / "index.html").write_text(full_html, encoding="utf-8")

        yield {"event": "done", "data": json.dumps({"message": "Website generated successfully"})}

    return EventSourceResponse(event_generator())


# ── Patient List (for UI) ──────────────────────────────────────────────────

@app.get("/api/patients")
def list_patients(db: Session = Depends(get_db)):
    patients = db.query(Patient).all()
    return [{"id": p.id, "name": f"{p.first_name} {p.last_name}", "diagnosis": p.primary_diagnosis} for p in patients]


# ── Static Serving ──────────────────────────────────────────────────────────

@app.get("/site/{path:path}")
async def serve_generated(path: str = ""):
    if not path or path == "/":
        path = "index.html"
    file_path = GENERATED_DIR / path
    if file_path.exists():
        return FileResponse(file_path, media_type="text/html")
    return HTMLResponse(
        "<html><body style='background:#0F172A;color:#94A3B8;display:flex;align-items:center;justify-content:center;height:100vh;font-family:Inter,sans-serif;'>"
        "<div style='text-align:center'><h2 style='color:#3B82F6'>Lens</h2><p>No website generated yet.<br>Configure your tools and ask me to build a dashboard.</p></div>"
        "</body></html>"
    )


@app.get("/site/")
async def serve_generated_root():
    return await serve_generated("")


@app.get("/")
async def serve_ui():
    return FileResponse(STATIC_DIR / "chat.html", media_type="text/html")
