"""Claude agents for Lens: tool analyzer + website planner + website generator."""

import json
import os
from typing import AsyncGenerator
from anthropic import Anthropic, AsyncAnthropic

client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
sync_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

AVAILABLE_ENDPOINTS = [
    {"endpoint": "/api/patient/{id}", "description": "Patient demographics, summary, and primary diagnosis", "category": "demographics"},
    {"endpoint": "/api/patient/{id}/vitals", "description": "Vital signs: heart rate, blood pressure, weight, temperature, respiratory rate, O2 saturation", "category": "vitals"},
    {"endpoint": "/api/patient/{id}/labs", "description": "Lab results: creatinine, sodium, potassium, troponin, ejection fraction, glucose, HbA1c, cholesterol, CBC, liver function, etc. Filter with ?test_name=", "category": "labs"},
    {"endpoint": "/api/patient/{id}/notes", "description": "Clinical notes from encounters: office visits, ER visits, hospitalizations, procedures", "category": "clinical_notes"},
    {"endpoint": "/api/patient/{id}/medications", "description": "Medication history: current and past medications with start/stop dates and reasons", "category": "medications"},
    {"endpoint": "/api/patient/{id}/weights", "description": "Weight history over time for trend tracking", "category": "vitals"},
]

TOOL_ANALYZER_PROMPT = """You analyze healthcare tool/product descriptions. Given a description, extract:
1. The tool name (short, e.g. "Epic EHR")
2. An icon keyword (one of: hospital, flask, heartbeat, pill, clipboard, monitor, stethoscope, microscope, chart)
3. What data categories this tool provides
4. Which of the available API endpoints can serve this data

Available API endpoints:
{endpoints}

Return ONLY valid JSON in this exact format:
{{
  "name": "Tool Name",
  "icon": "icon_keyword",
  "data_categories": ["demographics", "vitals", "labs", "clinical_notes", "medications"],
  "mapped_endpoints": [
    {{"category": "demographics", "endpoint": "/api/patient/{{id}}", "description": "Patient demographics and summary"}}
  ],
  "summary": "One sentence summary of what this tool provides."
}}"""

PLANNER_PROMPT = """You are Lens, an AI healthcare dashboard planner. You help doctors design the perfect monitoring dashboard through conversation.

CONFIGURED TOOLS:
{tools_context}

PATIENT CONTEXT:
{patient_context}

AVAILABLE DATA ENDPOINTS:
{endpoints_list}

YOUR ROLE:
You are a collaborative planner. When the user describes what they want, you should:

1. **Propose a dashboard plan** — describe the sections you'd build, what data each section would show, and how it would be laid out. Use clear headers and bullet points.

2. **Map data to their tools** — explain which of their configured tools provides the data for each section. If a section would use data from a specific tool, say so.

3. **Identify missing data** — if there's data that would make the dashboard better but isn't available from their configured tools, mention it as a suggestion. For example: "If you had access to echocardiogram reports, I could add a cardiac imaging section."

4. **Ask for confirmation** — end your response by asking if the user wants to proceed with this plan, or if they'd like to adjust anything.

FORMAT YOUR RESPONSE as readable text with these sections:
- A short title for the dashboard
- **Sections** — numbered list of dashboard sections with what each shows
- **Data sources** — which configured tools feed each section
- **Nice to have** — additional data that would enhance it (if any)
- A closing question asking if they'd like to proceed or adjust

Keep it concise but informative. Be enthusiastic but professional. Do NOT output any HTML or code — just the plan as readable text.

If the user is asking follow-up questions or making adjustments to a previous plan, incorporate their feedback and present the updated plan.

If the user says something like "yes", "looks good", "go ahead", "build it", "generate", or otherwise approves the plan, respond with EXACTLY this on its own line at the very end:
[READY_TO_GENERATE]

This signals that the plan is approved and the dashboard should be built."""

WEBSITE_GENERATOR_PROMPT = """You are Lens, an AI that generates complete healthcare dashboard websites.

CONFIGURED TOOLS:
{tools_context}

PATIENT CONTEXT:
{patient_context}

CONVERSATION CONTEXT (the plan that was discussed and approved):
{plan_context}

CRITICAL: The patient ID is EXACTLY: {patient_id}
You MUST hardcode this ID in all fetch() URLs. The API is on the same origin (relative URLs).

AVAILABLE ENDPOINTS AND THEIR EXACT JSON RESPONSE SHAPES:

1. GET /api/patient/{patient_id}
Returns: {sample_patient}

2. GET /api/patient/{patient_id}/vitals  (returns array, most recent first)
Each item: {sample_vital}
vital_type values: heart_rate, systolic_bp, diastolic_bp, weight, temperature, respiratory_rate, o2_saturation

3. GET /api/patient/{patient_id}/labs  (returns array, most recent first)
Each item: {sample_lab}
test_name values: creatinine, sodium, potassium, troponin, ejection_fraction, glucose, hemoglobin_a1c, cholesterol_total, cholesterol_hdl, cholesterol_ldl, bun, hemoglobin, hematocrit, platelets, wbc, rbc, alt, ast, albumin, bilirubin, gfr, triglycerides
Filter by test: /api/patient/{patient_id}/labs?test_name=troponin

4. GET /api/patient/{patient_id}/notes  (returns array, most recent first)
Each item: {sample_note}

5. GET /api/patient/{patient_id}/medications  (returns array)
Each item: {sample_med}

6. GET /api/patient/{patient_id}/weights  (returns array, most recent first)
Each item: {sample_weight}

TECHNICAL REQUIREMENTS:
- Output a COMPLETE, self-contained HTML document
- Use Tailwind CSS via CDN: <script src="https://cdn.tailwindcss.com"></script>
- Use Chart.js via CDN: <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
- Use Inter font via Google Fonts
- All data MUST be fetched live from the API using fetch() — do NOT use placeholder/hardcoded data
- Use async/await and handle the responses correctly based on the JSON shapes shown above
- The page will be displayed inside an iframe

REQUIRED DATA FETCHING PATTERN — use this exact pattern:
```javascript
document.addEventListener('DOMContentLoaded', async () => {{
  try {{
    const pid = '{patient_id}';
    const [patient, vitals, labs, notes, medications, weights] = await Promise.all([
      fetch('/api/patient/' + pid).then(r => r.json()),
      fetch('/api/patient/' + pid + '/vitals').then(r => r.json()),
      fetch('/api/patient/' + pid + '/labs').then(r => r.json()),
      fetch('/api/patient/' + pid + '/notes').then(r => r.json()),
      fetch('/api/patient/' + pid + '/medications').then(r => r.json()),
      fetch('/api/patient/' + pid + '/weights').then(r => r.json()),
    ]);
    // patient is an object with: id, first_name, last_name, birth_date, gender, race, city, state, primary_diagnosis
    // vitals is an array of: {{recorded_at, vital_type, value, unit}}
    // labs is an array of: {{recorded_at, test_name, value, unit, loinc_code}}
    // notes is an array of: {{encounter_date, encounter_class, description, reason}}
    // medications is an array of: {{start_date, stop_date, description, reason}}
    // weights is an array of: {{recorded_at, vital_type, value, unit}}

    // To filter vitals by type: vitals.filter(v => v.vital_type === 'heart_rate')
    // To filter labs by test: labs.filter(l => l.test_name === 'troponin')

    buildDashboard(patient, vitals, labs, notes, medications, weights);
  }} catch(e) {{
    document.body.innerHTML = '<div style="color:red;padding:40px">Error loading data: ' + e.message + '</div>';
  }}
}});
```

DESIGN SYSTEM:
- Dark medical theme: bg-slate-900 (#0F172A), cards bg-slate-800 (#1E293B)
- Accent blue: #3B82F6 for primary actions and highlights
- Warning amber: #F59E0B for alerts and warnings
- Danger red: #EF4444 for critical values
- Success green: #10B981 for normal/good values
- Card-based layout with subtle shadows, rounded-xl corners, generous padding
- Charts: smooth lines, clear legends, labeled axes, semi-transparent fills
- Tables: zebra striping, hover highlights
- Header with dashboard title + patient name + key demographics
- Footer: "Generated by Lens · AI Healthcare Platform · Powered by TechWolf"
- Responsive design that works well in an iframe

SELF-DESCRIPTION (for future skills profile analysis):
- Add <meta name="description" content="..."> summarizing the page purpose
- Add <meta name="required-skills" content="..."> listing clinical skills needed
- Add data-purpose="..." attribute on each <section>
- Add HTML comments: <!-- Section: [Name] - [Purpose] - Skills: [...] -->

Build the dashboard following the approved plan. Make it comprehensive with real charts and tables using the LIVE API data.

OUTPUT: ONLY the complete HTML document. No markdown fences, no explanation. Start with <!DOCTYPE html> and end with </html>."""


async def analyze_tool(description: str) -> dict:
    """Analyze a tool description and return structured data about its capabilities."""
    endpoints_text = "\n".join(
        f"- {e['endpoint']}: {e['description']} (category: {e['category']})"
        for e in AVAILABLE_ENDPOINTS
    )

    response = await client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=[{"role": "user", "content": description}],
        system=TOOL_ANALYZER_PROMPT.format(endpoints=endpoints_text),
    )

    text = response.content[0].text.strip()
    # Try to parse JSON from the response
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


async def stream_chat(
    message: str,
    tools_context: str,
    patient_context: str,
    history: list[dict],
) -> AsyncGenerator[str, None]:
    """Stream a conversational planning response from Claude."""
    endpoints_list = "\n".join(
        f"- {e['endpoint']}: {e['description']}"
        for e in AVAILABLE_ENDPOINTS
    )

    system = PLANNER_PROMPT.format(
        tools_context=tools_context or "No tools configured yet.",
        patient_context=patient_context,
        endpoints_list=endpoints_list,
    )

    messages = []
    for h in history[-10:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})

    async with client.messages.stream(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2048,
        system=system,
        messages=messages,
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def stream_website(
    tools_context: str,
    patient_context: str,
    patient_id: str,
    plan_context: str,
    sample_data: dict | None = None,
) -> AsyncGenerator[str, None]:
    """Stream a generated website HTML from Claude, based on the approved plan."""
    sd = sample_data or {}

    system = WEBSITE_GENERATOR_PROMPT.format(
        tools_context=tools_context or "No tools configured yet. Use all available endpoints.",
        patient_context=patient_context,
        patient_id=patient_id,
        plan_context=plan_context,
        sample_patient=json.dumps(sd.get("patient", {}), default=str),
        sample_vital=json.dumps(sd.get("vital", {}), default=str),
        sample_lab=json.dumps(sd.get("lab", {}), default=str),
        sample_note=json.dumps(sd.get("note", {}), default=str),
        sample_med=json.dumps(sd.get("med", {}), default=str),
        sample_weight=json.dumps(sd.get("weight", {}), default=str),
    )

    messages = [{"role": "user", "content": "Generate the dashboard now based on the approved plan."}]

    async with client.messages.stream(
        model="claude-sonnet-4-5-20250929",
        max_tokens=16000,
        system=system,
        messages=messages,
    ) as stream:
        async for text in stream.text_stream:
            yield text
