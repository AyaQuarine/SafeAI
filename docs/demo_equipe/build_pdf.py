"""Génère le PDF de démonstration équipe (captures locales)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent
SCREENS = ROOT / "screens"
OUT = ROOT / "SafeAI_Common_V1_Demo_Equipe.pdf"

PAGE_W, PAGE_H = A4
MARGIN = 14 * mm
INK = HexColor("#101820")
MUTED = HexColor("#4a5560")
ACCENT = HexColor("#1f6f5a")
LINE = HexColor("#d5dbe3")
BG = HexColor("#f4f6f8")


def draw_header(c: canvas.Canvas, title: str, subtitle: str = "") -> float:
    y = PAGE_H - MARGIN
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(MARGIN, y, title)
    y -= 6 * mm
    if subtitle:
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 9)
        c.drawString(MARGIN, y, subtitle)
        y -= 5 * mm
    c.setStrokeColor(LINE)
    c.setLineWidth(0.6)
    c.line(MARGIN, y, PAGE_W - MARGIN, y)
    return y - 6 * mm


def bullets(c: canvas.Canvas, y: float, lines: list[str], size: int = 9) -> float:
    c.setFont("Helvetica", size)
    c.setFillColor(INK)
    for line in lines:
        c.drawString(MARGIN, y, "•  " + line)
        y -= 4.6 * mm
    return y


def section(c: canvas.Canvas, y: float, title: str) -> float:
    c.setFillColor(ACCENT)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(MARGIN, y, title)
    return y - 5.5 * mm


def fit_image(path: Path, max_w: float, max_h: float) -> tuple[float, float]:
    with Image.open(path) as im:
        w, h = im.size
    scale = min(max_w / w, max_h / h)
    return w * scale, h * scale


def draw_shot(c: canvas.Canvas, y: float, path: Path, caption: str, max_h: float = 95 * mm) -> float:
    max_w = PAGE_W - 2 * MARGIN
    iw, ih = fit_image(path, max_w, max_h)
    y -= ih
    c.setFillColor(BG)
    c.rect(MARGIN - 1 * mm, y - 1 * mm, iw + 2 * mm, ih + 2 * mm, fill=1, stroke=0)
    c.drawImage(str(path), MARGIN, y, width=iw, height=ih, preserveAspectRatio=True, mask="auto")
    y -= 4.5 * mm
    c.setFillColor(MUTED)
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(MARGIN, y, caption)
    return y - 6 * mm


def page_footer(c: canvas.Canvas, n: int, total: int) -> None:
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 8)
    c.drawString(MARGIN, 8 * mm, "SafeAI Common V1.0 — pédagogique, non médical, non certifié")
    c.drawRightString(PAGE_W - MARGIN, 8 * mm, f"{n}/{total}")


def build() -> Path:
    c = canvas.Canvas(str(OUT), pagesize=A4)
    total = 5

    # --- Page 1 : essentiel ---
    y = draw_header(
        c,
        "SafeAI Common V1.0 — Brief équipe",
        "Chaîne hors ligne validée · visualiseur JSON · pas d’analyses métier encore",
    )
    y = section(c, y, "Ce qui est livré")
    y = bullets(
        c,
        y,
        [
            "WAV FR → Acquisition (Silero + qualité) → Whisper → Community-1 → AlignmentEngine",
            "Deux JSON même session_id : aligned_session + *_diarization.json",
            "UI native : http://127.0.0.1:8765/  et  /common",
            "UI Docker démo : http://127.0.0.1:8766/common  (sans Silero / sans micro)",
            "Tests : 174 (backend gelé) · 177 (backend + UI /common)",
        ],
    )
    y -= 2 * mm
    y = section(c, y, "Ce qui n’est PAS inclus")
    y = bullets(
        c,
        y,
        [
            "Pas d’analyse de risque / émotions / fusion finale",
            "Pas de WER·DER·latence continue validés",
            "Le navigateur ne lance aucun modèle — CLI d’abord, UI ensuite",
        ],
    )
    y -= 2 * mm
    y = section(c, y, "Recette réelle (dernière campagne)")
    # compact table
    rows = [
        ("R01", "completed", "2 énoncés · 1 locuteur"),
        ("R02", "empty", "0 énoncé · hallucination Amara en diagnostics"),
        ("R06", "partial", "10 énoncés · 2 locuteurs · 1 intersection (~321 ms)"),
    ]
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(MUTED)
    c.drawString(MARGIN, y, "Scénario")
    c.drawString(MARGIN + 28 * mm, y, "Statut")
    c.drawString(MARGIN + 55 * mm, y, "À l’écran")
    y -= 4 * mm
    c.setStrokeColor(LINE)
    c.line(MARGIN, y + 2 * mm, PAGE_W - MARGIN, y + 2 * mm)
    c.setFont("Helvetica", 9)
    c.setFillColor(INK)
    for a, b, d in rows:
        c.drawString(MARGIN, y, a)
        c.drawString(MARGIN + 28 * mm, y, b)
        c.drawString(MARGIN + 55 * mm, y, d)
        y -= 5 * mm
    y -= 2 * mm
    y = section(c, y, "R06 — pourquoi partial")
    y = bullets(
        c,
        y,
        [
            "Hallucinations Whisper écartées (ex. Sous-titres, « ? » trop long)",
            "Parole Community-1 sans texte lexical → diagnostics « non transcrite »",
            "Chevauchement conservé dans diarization.json, pas inventé en énoncé texte",
        ],
    )
    y -= 2 * mm
    y = section(c, y, "Pour les deux sous-équipes")
    y = bullets(
        c,
        y,
        [
            "Équipe texte → AlignedSession (texte, temps, attribution, diagnostics)",
            "Équipe voix → DiarizationResult (tours, locuteurs anonymes, overlaps)",
            "Ne pas présenter les exemples /static/examples comme des résultats réels",
        ],
    )
    page_footer(c, 1, total)
    c.showPage()

    # --- Page 2 : acquisition ---
    y = draw_header(c, "1 · Diagnostic acquisition (natif)", "http://127.0.0.1:8765/  · démo synthétique")
    y = bullets(
        c,
        y,
        [
            "Signal + plages parole/silence",
            "Métriques qualité · segments vers la suite · incidents",
            "Docker léger ≠ Silero complet → acquisition réelle en natif",
        ],
    )
    y = draw_shot(
        c,
        y,
        SCREENS / "01_acquisition_demo.png",
        "Capture : démonstration synthétique — 11 blocs · 3 segments de parole",
        max_h=145 * mm,
    )
    page_footer(c, 2, total)
    c.showPage()

    # --- Page 3 : R01 + R02 ---
    y = draw_header(c, "2 · Common V1 — R01 et R02", "http://127.0.0.1:8765/common  · JSON réels")
    y = bullets(c, y, ["R01 completed — 2 énoncés, speaker_00", "R02 empty — silence confirmé, hallucination journalisée"])
    y = draw_shot(c, y, SCREENS / "02_common_r01.png", "R01 — session terminée", max_h=70 * mm)
    y = draw_shot(c, y, SCREENS / "03_common_r02.png", "R02 — silence confirmé", max_h=70 * mm)
    page_footer(c, 3, total)
    c.showPage()

    # --- Page 4 : R06 ---
    y = draw_header(c, "3 · Common V1 — R06 (deux locuteurs)", "aligned + diarization · même session_id")
    y = bullets(
        c,
        y,
        [
            "Statut Partielle · 10 énoncés · 2 locuteurs",
            "9 tours · 1 intersection (~321 ms) · 2 tours marqués overlap",
            "Diagnostics : texte suspect écarté + parole non transcrite",
        ],
    )
    y = draw_shot(c, y, SCREENS / "04_common_r06.png", "R06 — résultat partiel documenté (pas un échec silencieux)", max_h=140 * mm)
    page_footer(c, 4, total)
    c.showPage()

    # --- Page 5 : Docker + suite ---
    y = draw_header(c, "4 · Docker démo + prochaines étapes", "http://127.0.0.1:8766/common")
    y = bullets(
        c,
        y,
        [
            "docker compose up -d demo → port 8766 (natif reste sur 8765)",
            "SAFEAI_DISABLE_MICROPHONE=1 · pas de HF token pour la démo",
            "Profil models / offline : optionnel, non validé comme jalon V1",
        ],
    )
    y = draw_shot(c, y, SCREENS / "05_docker_common.png", "Même visualiseur depuis le conteneur démo", max_h=95 * mm)
    y = section(c, y, "État & suite")
    y = bullets(
        c,
        y,
        [
            "Gel backend Common V1.0 : OK · Frontend démo : OK · Docker démo : OK",
            "Ouvert : WER/DER/latence · continu · analyses texte/voix métier · 1er push Git",
        ],
    )
    page_footer(c, 5, total)
    c.showPage()
    c.save()
    return OUT


if __name__ == "__main__":
    path = build()
    print(path)
