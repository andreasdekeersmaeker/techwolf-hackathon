from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Text
from database import Base
from pydantic import BaseModel
from datetime import date, datetime
from typing import Optional


# ── ORM Models ──────────────────────────────────────────────────────────────

class Patient(Base):
    __tablename__ = "patients"
    id = Column(String, primary_key=True)
    first_name = Column(String)
    last_name = Column(String)
    birth_date = Column(Date)
    gender = Column(String)
    race = Column(String)
    city = Column(String)
    state = Column(String)
    primary_diagnosis = Column(String)


class VitalSign(Base):
    __tablename__ = "vital_signs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String, index=True)
    recorded_at = Column(DateTime)
    vital_type = Column(String)  # heart_rate, systolic_bp, diastolic_bp, weight, temperature, respiratory_rate, o2_sat
    value = Column(Float)
    unit = Column(String)


class LabResult(Base):
    __tablename__ = "lab_results"
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String, index=True)
    recorded_at = Column(DateTime)
    test_name = Column(String)  # creatinine, sodium, potassium, troponin, ejection_fraction, glucose, hemoglobin_a1c, etc.
    value = Column(Float)
    unit = Column(String)
    loinc_code = Column(String)


class ClinicalNote(Base):
    __tablename__ = "clinical_notes"
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String, index=True)
    encounter_date = Column(DateTime)
    encounter_class = Column(String)
    description = Column(Text)
    reason = Column(Text)


class MedicationLog(Base):
    __tablename__ = "medication_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String, index=True)
    start_date = Column(DateTime)
    stop_date = Column(DateTime, nullable=True)
    description = Column(String)
    reason = Column(String, nullable=True)


# ── Pydantic Schemas ────────────────────────────────────────────────────────

class PatientOut(BaseModel):
    id: str
    first_name: str
    last_name: str
    birth_date: date
    gender: str
    race: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    primary_diagnosis: Optional[str] = None

    class Config:
        from_attributes = True


class VitalSignOut(BaseModel):
    recorded_at: datetime
    vital_type: str
    value: float
    unit: str

    class Config:
        from_attributes = True


class LabResultOut(BaseModel):
    recorded_at: datetime
    test_name: str
    value: float
    unit: str
    loinc_code: Optional[str] = None

    class Config:
        from_attributes = True


class ClinicalNoteOut(BaseModel):
    encounter_date: datetime
    encounter_class: str
    description: str
    reason: Optional[str] = None

    class Config:
        from_attributes = True


class MedicationLogOut(BaseModel):
    start_date: datetime
    stop_date: Optional[datetime] = None
    description: str
    reason: Optional[str] = None

    class Config:
        from_attributes = True


class ToolAnalysis(BaseModel):
    name: str
    icon: str
    data_categories: list[str]
    mapped_endpoints: list[dict]
    summary: str


class ChatRequest(BaseModel):
    patient_id: str
    message: str
    tools: list[dict] = []
    history: list[dict] = []
