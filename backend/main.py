# backend/app/main.py
import os
import re
import json
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import httpx

# ---------- Load environment ----------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "")
if not HF_TOKEN:
    print("WARNING: HF_TOKEN not set. Hugging Face API may fail.")

# ---------- Configuration ----------
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
REQUEST_TIMEOUT = 35.0
MAX_RETRIES = 2
RATE_LIMIT_WAIT_BASE = 5  # seconds

# Simple in‑memory cache (for identical user inputs)
cache: Dict[str, Dict] = {}

# ---------- Pydantic models ----------
class GenerateRequest(BaseModel):
    user_input: str = Field(..., max_length=500, description="App description")

class ArchitectureComponent(BaseModel):
    name: str
    type: str  # frontend|backend|database|cache|api_gateway|other
    description: str

class Threat(BaseModel):
    stride_category: str  # Spoofing|Tampering|Repudiation|InformationDisclosure|DenialOfService|ElevationOfPrivilege
    threat: str
    mitigation: str

class SecurityControl(BaseModel):
    category: str  # Authentication|Authorization|DataProtection|Logging|Other
    control: str

class ArchitectureResponse(BaseModel):
    architecture_components: List[ArchitectureComponent]
    data_flow_description: str
    threat_model: List[Threat]
    security_controls: List[SecurityControl]
    mermaid_diagram_code: str
    safety_score: int = Field(ge=0, le=100)
    safety_warnings: List[str]

# ---------- LLM interaction ----------
async def call_llm(prompt: str) -> str:
    """Call Hugging Face inference API with retries and rate limit handling."""
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 1500,
            "temperature": 0.3,
            "return_full_text": False
        }
    }

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await client.post(HF_API_URL, headers=HF_HEADERS, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    # Hugging Face returns a list with one dict containing 'generated_text'
                    if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
                        return data[0]["generated_text"].strip()
                    else:
                        raise ValueError("Unexpected response format from HF API")
                elif response.status_code == 429:  # Rate limited
                    wait = RATE_LIMIT_WAIT_BASE * (attempt + 1)
                    print(f"Rate limited. Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                else:
                    raise Exception(f"HF API error {response.status_code}: {response.text}")
            except (httpx.TimeoutException, httpx.RequestError) as e:
                if attempt == MAX_RETRIES:
                    raise Exception(f"Network error after retries: {str(e)}")
                await asyncio.sleep(2)
                continue
            except Exception as e:
                if attempt == MAX_RETRIES:
                    raise
                await asyncio.sleep(2)
                continue
    raise Exception("Max retries exceeded")

def build_prompt(user_input: str) -> str:
    """Construct the system + user prompt for the LLM."""
    system = (
        "You are an AI Security Architect. Output ONLY valid JSON. "
        "Do not include any other text, explanations, or markdown formatting. "
        "Use exactly this schema:\n"
        "{\n"
        '  "architecture_components": [\n'
        '    {"name": "string", "type": "frontend|backend|database|cache|api_gateway|other", "description": "string"}\n'
        "  ],\n"
        '  "data_flow_description": "string",\n'
        '  "threat_model": [\n'
        '    {"stride_category": "Spoofing|Tampering|Repudiation|InformationDisclosure|DenialOfService|ElevationOfPrivilege",\n'
        '     "threat": "string", "mitigation": "string"}\n'
        "  ],\n"
        '  "security_controls": [\n'
        '    {"category": "Authentication|Authorization|DataProtection|Logging|Other", "control": "string"}\n'
        "  ],\n"
        '  "mermaid_diagram_code": "string"\n'
        "}\n"
        "Make sure the Mermaid code is valid (graph TD syntax)."
    )
    return f"{system}\n\nUser request: {user_input}\n\nJSON:"

def extract_json(raw: str) -> Dict[str, Any]:
    """Extract JSON from LLM output that may contain extra text."""
    # Remove markdown code fences
    raw = re.sub(r'```json\s*|\s*```', '', raw, flags=re.IGNORECASE)
    # Find first { and last }
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM output")
    json_str = match.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e.msg} near {json_str[e.pos-20:e.pos+20]}")

# ---------- Safety evaluation (keyword based) ----------
def evaluate_safety(data: Dict[str, Any]) -> tuple[int, List[str]]:
    """Return (score, warnings). Score 0-100, higher is safer."""
    score = 100
    warnings = []
    # Combine all text fields for keyword search
    all_text = json.dumps(data).lower()
    controls = json.dumps(data.get("security_controls", [])).lower()
    data_flow = data.get("data_flow_description", "").lower()

    # Authentication check
    if not any(k in controls for k in ["authentication", "auth", "login", "mfa", "jwt", "oauth"]):
        score -= 15
        warnings.append("Missing authentication controls")
    # Encryption check
    if not any(k in controls for k in ["encrypt", "tls", "https", "aes", "kms"]):
        score -= 15
        warnings.append("Missing encryption controls (at rest or in transit)")
    # Logging / monitoring
    if not any(k in controls for k in ["log", "audit", "monitor", "cloudtrail", "siem"]):
        score -= 10
        warnings.append("Missing logging / monitoring controls")
    # STRIDE completeness (at least 4 distinct categories)
    threats = data.get("threat_model", [])
    categories = {t.get("stride_category") for t in threats if isinstance(t, dict)}
    missing_categories = 0
    expected = {"Spoofing", "Tampering", "Repudiation", "InformationDisclosure", "DenialOfService", "ElevationOfPrivilege"}
    for cat in expected:
        if cat not in categories:
            missing_categories += 1
    score -= min(missing_categories * 5, 20)
    if missing_categories >= 3:
        warnings.append(f"Only {len(categories)} of 6 STRIDE categories covered")
    # Data flow description length
    if len(data_flow) < 20:
        score -= 10
        warnings.append("Data flow description too short")
    # Mermaid validation (simple check for graph TD and arrows)
    mermaid = data.get("mermaid_diagram_code", "")
    if "graph TD" not in mermaid or "-->" not in mermaid:
        score -= 20
        warnings.append("Mermaid diagram invalid or missing syntax")
    # Ensure score is within bounds
    score = max(0, min(100, score))
    return score, warnings

# ---------- FastAPI app ----------
app = FastAPI(title="AI Security Architect Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For MVP; restrict to your Vercel domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok", "model": "Mistral-7B-Instruct-v0.3"}

@app.post("/generate", response_model=ArchitectureResponse)
async def generate(request: GenerateRequest):
    # Check cache
    cache_key = request.user_input.strip().lower()
    if cache_key in cache:
        cached = cache[cache_key].copy()
        return cached

    try:
        prompt = build_prompt(request.user_input)
        raw_output = await call_llm(prompt)
        parsed = extract_json(raw_output)

        # Validate required top-level keys
        required_keys = ["architecture_components", "data_flow_description", "threat_model", "security_controls", "mermaid_diagram_code"]
        for key in required_keys:
            if key not in parsed:
                raise ValueError(f"Missing required key: {key}")

        # Convert dicts to Pydantic models (will validate types)
        components = [ArchitectureComponent(**c) for c in parsed["architecture_components"]]
        threats = [Threat(**t) for t in parsed["threat_model"]]
        controls = [SecurityControl(**c) for c in parsed["security_controls"]]

        # Compute safety
        safety_score, safety_warnings = evaluate_safety(parsed)

        response_obj = ArchitectureResponse(
            architecture_components=components,
            data_flow_description=parsed["data_flow_description"],
            threat_model=threats,
            security_controls=controls,
            mermaid_diagram_code=parsed["mermaid_diagram_code"],
            safety_score=safety_score,
            safety_warnings=safety_warnings
        )

        # Cache (limit size to avoid memory bloat)
        if len(cache) > 100:
            # Remove oldest (simple FIFO)
            oldest_key = next(iter(cache))
            del cache[oldest_key]
        cache[cache_key] = response_obj.dict()

        return response_obj

    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid AI output: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# Optional: startup/shutdown events for logging
@app.on_event("startup")
async def startup():
    print("AI Security Architect backend started.")

@app.on_event("shutdown")
async def shutdown():
    print("Backend shutting down.")
