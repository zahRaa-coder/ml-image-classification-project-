const $ = (id) => document.getElementById(id);
const drop = $("drop"), fileIn = $("file"), preview = $("preview"), go = $("go");
let file = null, last = null, camMode = "cam";

try { const t = localStorage.getItem("theme"); if (t) document.documentElement.dataset.theme = t; } catch {}
$("themeBtn").onclick = () => {
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const cur = document.documentElement.dataset.theme || (dark ? "dark" : "light");
  const next = cur === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch {}
};

function setFile(f) {
  if (!f || !f.type.startsWith("image/")) return showError("Please choose an image file.");
  if (f.size > 10 * 1024 * 1024) return showError("Image is larger than 10 MB.");
  file = f; showError("");
  preview.src = URL.createObjectURL(f); preview.hidden = false; $("dropInner").hidden = true;
  go.disabled = false;
}
function showError(msg) { $("error").textContent = msg; $("error").hidden = !msg; }

fileIn.onchange = () => setFile(fileIn.files[0]);
["dragenter", "dragover"].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", ev => setFile(ev.dataTransfer.files[0]));
document.addEventListener("paste", ev => { const it = [...ev.clipboardData.items].find(i => i.type.startsWith("image/")); if (it) setFile(it.getAsFile()); });

go.onclick = async () => {
  if (!file) return;
  go.classList.add("loading"); go.disabled = true; showError("");
  const fd = new FormData(); fd.append("image", file); fd.append("model", $("model").value);
  try {
    const r = await fetch("/predict", { method: "POST", body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || "Prediction failed");
    render(d);
  } catch (e) { showError(e.message); }
  finally { go.classList.remove("loading"); go.disabled = false; }
};

function render(d) {
  last = d;
  $("results").hidden = false;
  $("meta").textContent = `${d.model} · ${d.inference_ms} ms`;
  $("unknown").hidden = !d.unknown;
  $("score").textContent = d.score; $("thr").textContent = d.threshold;
  $("preds").classList.toggle("dim", d.unknown);
  $("binCard").hidden = d.unknown;
  $("preds").innerHTML = d.predictions.map((p, i) => `
    <li data-slug="${p.slug}" class="${i === 0 ? "sel" : ""}">
      <div class="row"><span>${i + 1}. ${p.name}</span><span>${(p.confidence * 100).toFixed(1)}%</span></div>
      <div class="bar"><i data-w="${(p.confidence * 100).toFixed(1)}"></i></div>
    </li>`).join("");
  requestAnimationFrame(() => document.querySelectorAll(".bar i").forEach(b => b.style.width = b.dataset.w + "%"));
  document.querySelectorAll("#preds li").forEach(li => li.onclick = () => {
    document.querySelectorAll("#preds li").forEach(x => x.classList.remove("sel"));
    li.classList.add("sel"); $("binCard").hidden = false; showBin(d.all_info[li.dataset.slug], li.dataset.slug);
  });
  if (d.unknown) document.querySelectorAll(".bin-slot").forEach(el => el.classList.remove("on"));
  else showBin(d.info, d.predictions[0].slug);
  showCam();
  $("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

function showBin(b, slug) {
  $("bName").textContent = b.name;
  $("bRec").textContent = b.recyclable ? "Recyclable" : "Not recyclable";
  $("bBin").textContent = b.bin;
  $("bTips").innerHTML = b.tips.map(t => `<li>${t}</li>`).join("");
  $("bFact").textContent = b.fact;
  document.querySelectorAll(".bin-slot").forEach(el => el.classList.toggle("on", el.dataset.slug === slug));
}
function showCam() { if (last) $("cam").src = camMode === "cam" ? last.gradcam : last.original; }
document.querySelectorAll(".seg button").forEach(b => b.onclick = () => {
  document.querySelectorAll(".seg button").forEach(x => x.classList.remove("on"));
  b.classList.add("on"); camMode = b.dataset.v; showCam();
});
