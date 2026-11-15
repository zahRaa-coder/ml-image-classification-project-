"""Step 2c: image review page - click images to mark them wrong, then save.

Run:  python src/review.py   ->  http://127.0.0.1:5001
Marked images are appended to data/rejected.txt (files are moved to data/rejected/,
never permanently deleted). Re-run split.py afterwards.
"""
import shutil
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from config import ROOT, CLEAN_DIR, SLUGS, DISPLAY, REJECTED_TXT, DATA

app = Flask(__name__)
TRASH = DATA / "rejected"

PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>EcoScan · Image Review</title><style>
body{font-family:system-ui,sans-serif;margin:0;background:#0f172a;color:#e2e8f0}
header{position:sticky;top:0;background:#111827ee;padding:12px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:2}
select,button{font-size:15px;padding:8px 12px;border-radius:8px;border:0}
button{background:#ef4444;color:#fff;cursor:pointer;font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px;padding:12px}
.c{position:relative;cursor:pointer;border-radius:8px;overflow:hidden;border:3px solid transparent}
.c img{width:100%;aspect-ratio:1;object-fit:cover;display:block}
.c.x{border-color:#ef4444}.c.x img{opacity:.35}
.c.x:after{content:"✕";position:absolute;inset:0;display:grid;place-items:center;font-size:48px;color:#ef4444}
small{opacity:.7}</style></head><body>
<header><b>Review</b>
<select id=cls onchange="location='?c='+this.value">{% for s in slugs %}
<option value="{{s}}" {{'selected' if s==cur}}>{{disp[s]}} ({{counts[s]}})</option>{% endfor %}</select>
<small>Click wrong images, then</small><button onclick=save()>Delete selected (<span id=n>0</span>)</button></header>
<div class=grid>{% for f in files %}<div class=c data-f="{{f}}" onclick="this.classList.toggle('x');upd()">
<img loading=lazy src="/img/{{cur}}/{{f}}"></div>{% endfor %}</div>
<script>
function upd(){n.textContent=document.querySelectorAll('.c.x').length}
async function save(){const f=[...document.querySelectorAll('.c.x')].map(e=>e.dataset.f);
if(!f.length||!confirm('Remove '+f.length+' images?'))return;
await fetch('/reject',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({c:'{{cur}}',files:f})});
location.reload()}
</script></body></html>"""


@app.route("/")
def index():
    cur = request.args.get("c", SLUGS[0])
    files = sorted(p.name for p in (CLEAN_DIR / cur).glob("*.jpg"))
    counts = {s: len(list((CLEAN_DIR / s).glob("*.jpg"))) for s in SLUGS}
    return render_template_string(PAGE, slugs=SLUGS, disp=DISPLAY, cur=cur, files=files, counts=counts)


@app.route("/img/<c>/<f>")
def img(c, f):
    return send_from_directory(CLEAN_DIR / c, f)


@app.route("/reject", methods=["POST"])
def reject():
    d = request.get_json(); c = d["c"]
    assert c in SLUGS
    (TRASH / c).mkdir(parents=True, exist_ok=True)
    with open(REJECTED_TXT, "a") as fh:
        for f in d["files"]:
            src = CLEAN_DIR / c / f
            if src.exists() and src.parent == CLEAN_DIR / c:
                fh.write(str(src.relative_to(ROOT)) + "\n")
                shutil.move(src, TRASH / c / f)
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(port=5001, debug=False)
