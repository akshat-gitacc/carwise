const NUMERIC = [
    ["age_years", "Age (years)", 0, 40, 1],
    ["mileage_km", "Mileage (km)", 0, 500000, 1000],
    ["previous_owners", "Previous owners", 1, 10, 1],
    ["service_completeness", "Service history completeness (0-1)", 0, 1, 0.05],
    ["months_since_service", "Months since last service", 0, 120, 1],
    ["accident_count", "Accidents reported", 0, 10, 1],
    ["engine_cc", "Engine capacity (cc, 0 if electric)", 0, 6000, 1],
    ["safety_rating", "Safety rating (1-5)", 1, 5, 1],
    ["seats", "Seats", 2, 9, 1],
    ["doors", "Doors", 2, 5, 1],
    ["airbags", "Airbags", 0, 10, 1],
    ["warranty_remaining_months", "Warranty remaining (months)", 0, 60, 1],
];

const SELECTS = [
    ["fuel_type", "Fuel type", ["petrol", "diesel", "electric", "hybrid"]],
    ["body_material", "Body material", ["steel", "aluminium", "composite"]],
    ["country_of_manufacture", "Country of manufacture", ["Japan", "Germany", "India", "USA", "Korea", "China", "Other"]],
    ["transmission", "Transmission", ["manual", "automatic"]],
    ["drive_type", "Drive type", ["FWD", "RWD", "AWD"]],
];

const form = document.getElementById("form");
const fields = document.getElementById("fields");
const statusEl = document.getElementById("status");
const btn = document.getElementById("go");
const resultEl = document.getElementById("result");

for (const [name, label, min, max, step] of NUMERIC) {
    fields.insertAdjacentHTML("beforeend",
        `<label>${label}<input name="${name}" type="number" min="${min}" max="${max}" step="${step}"></label>`);
}
for (const [name, label, opts] of SELECTS) {
    fields.insertAdjacentHTML("beforeend",
        `<label>${label}<select name="${name}">${opts.map(o => `<option>${o}</option>`).join("")}</select></label>`);
}

const esc = s => String(s).replace(/[&<>"']/g,
    c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = p => (p * 100).toFixed(0) + "%";
const pretty = s => s.replace(/_/g, " ");

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    btn.disabled = true;
    statusEl.className = "";
    statusEl.textContent = "Scoring car... (first run can take a while while the image model loads)";

    const car = {};
    for (const [name] of NUMERIC) {
        const v = form[name].value;
        car[name] = v === "" ? null : Number(v);
    }
    for (const [name] of SELECTS) car[name] = form[name].value;

    const fd = new FormData();
    fd.append("car", JSON.stringify(car));
    for (const f of form.exterior.files) fd.append("exterior", f);
    for (const f of form.engine.files) fd.append("engine", f);

    try {
        const res = await fetch("/api/assess", { method: "POST", body: fd });
        if (!res.ok) throw new Error(`Server returned ${res.status}`);
        const data = await res.json();
        statusEl.textContent = "";
        render(data);
    } catch (err) {
        statusEl.className = "error";
        statusEl.textContent = "Something went wrong: " + err.message;
    } finally {
        btn.disabled = false;
    }
});

function driverList(items) {
    if (!items.length) return "<p>None</p>";
    return "<ul>" + items.map(d =>
        `<li>${esc(pretty(d.feature))} = ${esc(d.value ?? "missing")} (${d.contribution > 0 ? "+" : ""}${d.contribution.toFixed(2)})</li>`
    ).join("") + "</ul>";
}

function render(r) {
    const probs = Object.entries(r.probabilities).map(([c, p]) =>
        `<div class="row"><span>${esc(c)}</span><div class="bar"><div style="width:${p * 100}%"></div></div><span>${pct(p)}</span></div>`
    ).join("");

    const visuals = Object.entries(r.visual_scores).map(([k, v]) =>
        `<li>${esc(pretty(k))}: ${v.toFixed(0)} / 100</li>`
    ).join("");

    const notes = r.photo_notes.length
        ? "<ul>" + r.photo_notes.map(n => `<li>${esc(n)}</li>`).join("") + "</ul>"
        : "";

    resultEl.innerHTML = `
    <h2>Predicted quality: ${esc(r.label)}</h2>
    <h3>Class probabilities</h3>${probs}
    <h3>Photo-based scores</h3><ul>${visuals}</ul>${notes}
    <h3>Explanation</h3>
    <p class="explain" id="explain"></p>
    <h3>Factors helping this rating</h3>${driverList(r.positives)}
    <h3>Factors holding it back</h3>${driverList(r.negatives)}
    <p class="caveat">Trained on synthetic demonstration data. This is a model estimate, not a
    verdict on the car's real condition. Get a professional inspection before buying.</p>
  `;
    document.getElementById("explain").textContent = r.explanation;
    resultEl.hidden = false;
}