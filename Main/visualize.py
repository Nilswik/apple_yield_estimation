"""Bilder och videor för äppelpipelinen. Påverkar inga siffror.

Modulen importerar inget från pipeline_3d.py; inställningar som avståndsgränsen skickas in som parametrar.
Text ritas med Pillow (put_text), eftersom OpenCV:s typsnitt saknar å, ä och ö.
"""

from functools import cache

import cv2
import matplotlib

# ! måste stå före pyplot-importen: ritar till fil utan att öppna något fönster
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

FPS = 15
HOLD_SECONDS = 5   # hur länge sammanfattningen visas i slutet av videon

GREEN = (0, 255, 0)
CYAN = (255, 255, 0)
RED = (0, 0, 255)
GRAY = (190, 190, 190)


def num(x, decimals=1):
    """Tal med decimalkomma: 11.6 -> "11,6"."""
    return f"{x:.{decimals}f}".replace(".", ",")


# Text och bilder

@cache
def _font(size, bold):
    """Arial (DejaVu Sans på Linux), annars Pillows inbyggda typsnitt."""
    names = ("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def put_text(img, text, org, scale, color=(255, 255, 255), thick=1):
    """Skriv text med svart kontur. org = textens baslinje (som cv2.putText), scale 1 = ungefär 30 px,
    thick >= 2 = fet stil. Bilden ändras på plats."""
    # * OpenCV:s typsnitt saknar å, ä, ö, därför ritas texten med Pillow och klistras in i bilden

    # 1. typsnitt och textens ruta
    font = _font(max(8, round(30 * scale)), thick >= 2)
    stroke = 2 if scale < 1 else 3
    left, top, right, bottom = font.getbbox(text, anchor="ls", stroke_width=stroke)

    # 2. klipp ut området runt texten
    # ? bara det lilla området konverteras, mycket snabbare än hela bilden (spelar roll i videorna)
    x0, y0 = max(0, org[0] + left), max(0, org[1] + top)
    x1, y1 = min(img.shape[1], org[0] + right + 1), min(img.shape[0], org[1] + bottom + 1)
    if x1 <= x0 or y1 <= y0:
        return

    # 3. rita texten med svart kontur och lägg tillbaka området
    # ! färgen är BGR (OpenCV-ordning); det fungerar eftersom Pillow aldrig byter kanalordning här
    region = Image.fromarray(np.ascontiguousarray(img[y0:y1, x0:x1]))
    ImageDraw.Draw(region).text((org[0] - x0, org[1] - y0), text, font=font, fill=tuple(color), anchor="ls",
                                stroke_width=stroke, stroke_fill=(0, 0, 0))
    img[y0:y1, x0:x1] = np.asarray(region)


def draw_box(img, box, color, thickness=1, pad=0):
    cv2.rectangle(img, (int(box.x1) - pad, int(box.y1) - pad), (int(box.x2) + pad, int(box.y2) + pad), color, thickness)


def top_bar(img, text, scale=0.6):
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    put_text(img, text, (10, 24), scale)


def tile_grid(tiles, cols):
    """Lägg lika stora bilder i ett rutnät; sista raden fylls ut med svart."""
    tiles = list(tiles)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def colorize_depth(filled, w, h, max_depth_mm):
    """Djupbilden i färg (nära = rött, långt bort = blått, för långt = grått, okänt = svart), skalad till w x h."""
    z = cv2.resize(filled, (w, h), interpolation=cv2.INTER_NEAREST)
    z0 = np.nan_to_num(z, nan=0)

    # ? skalan går från 0,5 m (rött) till avståndsgränsen (blått)
    norm = np.clip((z0 - 500) / (max_depth_mm - 500), 0, 1)
    img = cv2.applyColorMap((255 - norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[np.isnan(z)] = (0, 0, 0)
    img[z0 > max_depth_mm] = (80, 80, 80)
    return img


def crop_sheet(result, apples, label, path, cols=8, tile=120):
    """Spara ett utklipp per äpple (ur vyn med högst confidence), med label(äpple) som text."""
    views = result["views"]
    tiles = []

    for apple in apples.itertuples():
        own = views[(views.apple == apple.apple) & (views.z_source == "apple")]
        if not len(own):
            continue
        best = own.loc[own.conf.idxmax()]

        # klipp ut en kvadrat runt äpplet; kanten speglas om äpplet sitter nära bildkanten
        img = cv2.imread(str(result["frames"][int(best.frame)]))
        half = int(max(24, 1.4 * best["size"]))
        padded = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_REFLECT)
        cx, cy = int(best.cx) + half, int(best.cy) + half
        crop = cv2.resize(padded[cy - half: cy + half, cx - half: cx + half], (tile, tile),
                          interpolation=cv2.INTER_CUBIC)

        box_half = best["size"] / 2 * tile / (2 * half)
        c0, c1 = int(tile / 2 - box_half), int(tile / 2 + box_half)
        cv2.rectangle(crop, (c0, c0), (c1, c1), CYAN, 1)
        put_text(crop, label(apple), (3, tile - 6), 0.4)
        tiles.append(crop)

    if tiles:
        cv2.imwrite(str(path), tile_grid(tiles, cols))


def depth_check_plot(measured, scale, path, ref_diameter_mm):
    """Diagram för kalibreringen: i pixlar krymper äpplena med avståndet, i mm (med djup) gör de det inte."""
    z = measured.z_mm.to_numpy() / 1000
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    ax[0].scatter(z, measured.size_px, s=25)
    ax[0].set_title("I pixlar: äpplen långt bort ser mindre ut", loc="left")
    ax[0].set_ylabel("äpplets storlek i bilden (px)")

    ax[1].scatter(z, measured.diameter_raw_mm * scale, s=25)
    ax[1].axhline(ref_diameter_mm, color="gray", linewidth=1, label=f"referensäpple {ref_diameter_mm:.0f} mm")
    ax[1].set_title("I mm, med djup: storleken beror inte på avståndet", loc="left")
    ax[1].set_ylabel("äpplets diameter (mm)")
    ax[1].legend()

    for a in ax:
        a.set_xlabel("avstånd till äpplet (m)")
        a.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# Videor

def summary_panel(img, s):
    """Gör bilden mörkare och skriv sammanfattningen på den."""
    H, W = img.shape[:2]
    out = (img * 0.3).astype(np.uint8)

    x0, y0, x1, y1 = W // 2 - 450, H // 2 - 200, W // 2 + 450, H // 2 + 200
    cv2.rectangle(out, (x0, y0), (x1, y1), (20, 20, 20), -1)
    cv2.rectangle(out, (x0, y0), (x1, y1), (200, 200, 200), 2)

    x, y = x0 + 40, y0 + 50
    put_text(out, f"{s['recording']}  ({s['tree']})", (x, y), 0.75, GRAY)
    put_text(out, f"{s['apples_counted']} räknade äpplen", (x, y + 65), 1.5, (120, 255, 120), 3)
    put_text(out, f"rimligt intervall {s['count_low']} - {s['count_high']}", (x, y + 100), 0.75, GRAY)
    put_text(out, f"Uppskattad skörd: {num(s['estimate_kg'])} kg", (x, y + 155), 1.25, (255, 255, 255), 2)
    put_text(out, f"rimligt intervall {num(s['kg_low'])} - {num(s['kg_high'])} kg", (x, y + 190), 0.75, GRAY)
    put_text(out, f"medelvikt per äpple {s['mean_weight_g']:.0f} g, {s['weight_source']}", (x, y + 235), 0.6, GRAY)
    put_text(out, "Bara synliga äpplen, ena sidan av trädet", (x, y + 290), 0.6, (150, 150, 150))
    return out


def render(result, summary, visible, out, depth_frames, conf_keep, max_depth_mm):
    """annotated_3d.mp4 och final.png: räknade äpplen i grönt, uppmätta i cyan, bakgrund i rött,
    djupbilden i hörnet och sammanfattningen på slutet. depth_frames = fylld djupbild per bildruta."""
    frame_paths, detections = result["frames"], result["det"]
    H, W = cv2.imread(str(frame_paths[0])).shape[:2]
    video = cv2.VideoWriter(str(out / "annotated_3d.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    shown = {f: g for f, g in detections[detections.conf >= conf_keep].groupby("frame")}
    counted = {f: g for f, g in result["counted"].groupby("frame")}
    diameter_of = visible.set_index("apple").diameter_mm.to_dict()
    views = result["views"]
    measured = {f: g for f, g in views[views.apple.isin(diameter_of)].groupby("frame")} if len(views) else {}
    none = pd.DataFrame()

    inset_w, inset_h = 320, 180
    running, img = 0, None

    for frame, (frame_path, filled) in enumerate(zip(frame_paths, depth_frames)):
        img = cv2.imread(str(frame_path))

        # 1. alla säkra detektioner: grå, eller röd med avstånd om de är bakgrund
        for box in shown.get(frame, none).itertuples():
            if box.far:
                draw_box(img, box, RED, 2)
                put_text(img, f"{num(box.z_mm / 1000)} m", (int(box.x1), int(box.y1) - 4), 0.45, (120, 120, 255))
            else:
                draw_box(img, box, (170, 170, 170))

        # 2. äpplen som räknas i denna bildruta: gröna
        counted_here = counted.get(frame, none)
        running += len(counted_here)
        for box in counted_here.itertuples():
            draw_box(img, box, GREEN, 2)

        # 3. uppmätta äpplen: cyan med diameter
        for box in measured.get(frame, none).itertuples():
            draw_box(img, box, CYAN, 2, pad=2)
            put_text(img, f"{diameter_of[box.apple]:.0f} mm", (int(box.x1), int(box.y1) - 6), 0.5, CYAN)

        # 4. textrad, teckenförklaring och djupbild
        top_bar(img, f"bildruta {frame}   räknade hittills: {running}")

        lx, ly = 10, H - 70
        cv2.rectangle(img, (lx - 6, ly - 20), (lx + 330, H - 8), (0, 0, 0), -1)
        legend = [(GREEN, "räknas i denna bildruta"), (CYAN, "helt synligt: mätt i mm"),
                  (RED, "för långt bort (bakgrund): räknas inte")]
        for j, (color, text) in enumerate(legend):
            cv2.rectangle(img, (lx, ly - 10 + 20 * j), (lx + 14, ly + 2 + 20 * j), color, 2)
            put_text(img, text, (lx + 24, ly + 2 + 20 * j), 0.45)

        x0, y0 = W - inset_w - 10, H - inset_h - 10
        img[y0:y0 + inset_h, x0:x0 + inset_w] = colorize_depth(filled, inset_w, inset_h, max_depth_mm)
        cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + inset_w, y0 + inset_h), (255, 255, 255), 1)
        put_text(img, f"djup: rött nära, blått {max_depth_mm / 1000:.0f} m, grått för långt", (x0, y0 - 6), 0.42)

        video.write(img)

    # 5. sammanfattningen på sista bildrutan, visad i HOLD_SECONDS
    final = summary_panel(img, summary)
    cv2.imwrite(str(out / "final.png"), final)
    for _ in range(HOLD_SECONDS * FPS):
        video.write(final)
    video.release()


def render_audit(result, out, hold_seconds=2):
    """Långsam video för att kontrollera räkningen med ögat: varje räknat äpple får ett nummer och bildrutan
    hålls stilla. Skriver audit_numbered.mp4 och audit_apples.csv."""
    frame_paths = result["frames"]
    counted = {f: g for f, g in result["counted"].sort_values(["frame", "cx"]).groupby("frame")}
    H, W = cv2.imread(str(frame_paths[0])).shape[:2]
    video = cv2.VideoWriter(str(out / "audit_numbered.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    numbers, n = [], 0
    for frame, frame_path in enumerate(frame_paths):
        img = cv2.imread(str(frame_path))
        counted_here = counted.get(frame)

        if counted_here is None:
            top_bar(img, f"bildruta {frame}   räknade äpplen hittills: {n}", 0.65)
            video.write(img)
            continue

        # ? numren följer samma ordning som audit_apples.csv, så ett äpple i videon kan slås upp i tabellen
        for _, box in counted_here.iterrows():
            n += 1
            numbers.append(dict(number=n, frame=frame, x=round(box.cx), y=round(box.cy), conf=round(box.conf, 2)))
            draw_box(img, box, GREEN, 2)
            put_text(img, str(n), (int(box.x2) + 2, int(box.y1) + 10), 0.5, GREEN)

        top_bar(img, f"bildruta {frame}   räknade äpplen hittills: {n}", 0.65)
        for _ in range(hold_seconds * FPS):
            video.write(img)

    video.release()
    pd.DataFrame(numbers).to_csv(out / "audit_apples.csv", index=False)
