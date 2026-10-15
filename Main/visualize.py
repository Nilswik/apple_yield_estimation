"""Ritning för äppelpipelinen: videor, sammanfattningsbild, djupbild i färg, utklipp av äpplen och kalibreringsdiagram.

Påverkar inga siffror: pipeline_3d.py räknar ut allt och anropar sedan funktionerna här för att visa resultatet.
Modulen importerar inget från pipeline_3d.py; inställningar som avståndsgränsen skickas in som parametrar.

All text som kunden ser är på svenska och ritas med Pillow (put_text), eftersom OpenCV:s typsnitt saknar å, ä och ö.
"""

from functools import lru_cache

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

FPS = 15           # videornas bildfrekvens
HOLD_SECONDS = 5   # hur länge sammanfattningen visas i slutet av videon


def num(x, decimals=1):
    """Tal med decimalkomma, för text som kunden ser (11.6 -> "11,6")."""
    return f"{x:.{decimals}f}".replace(".", ",")


# ---------------------------------------------------------------- text och bilder

@lru_cache(maxsize=None)
def _font(size, bold):
    """Arial (eller DejaVu Sans på Linux) i given storlek; Pillows inbyggda typsnitt om ingen av dem finns."""
    for name in (("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def put_text(img, text, org, scale, color=(255, 255, 255), thick=1):
    """Skriv text med svart kontur så att den syns mot alla bakgrunder.

    Parametrar:
        img: BGR-bild, ändras på plats.
        text: texten.
        org: (x, y) för textens baslinje, vänstra änden (som i cv2.putText).
        scale: storlek som OpenCV:s fontScale (1 = typsnitt på ungefär 30 px).
        color: färg i BGR.
        thick: 1 = normal, 2 eller mer = fet.
    """
    font = _font(max(8, round(30 * scale)), thick >= 2)
    stroke = 2 if scale < 1 else 3
    left, top, right, bottom = font.getbbox(text, anchor="ls", stroke_width=stroke)
    # rita bara i området runt texten (snabbare än att konvertera hela bilden)
    x0, y0 = max(0, org[0] + left), max(0, org[1] + top)
    x1, y1 = min(img.shape[1], org[0] + right + 1), min(img.shape[0], org[1] + bottom + 1)
    if x1 <= x0 or y1 <= y0:
        return
    region = Image.fromarray(np.ascontiguousarray(img[y0:y1, x0:x1]))
    ImageDraw.Draw(region).text((org[0] - x0, org[1] - y0), text, font=font, fill=tuple(color), anchor="ls",
                                stroke_width=stroke, stroke_fill=(0, 0, 0))
    img[y0:y1, x0:x1] = np.asarray(region)


def tile_grid(tiles, cols):
    """Lägg lika stora bilder i ett rutnät; sista raden fylls ut med svart."""
    tiles = list(tiles)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def colorize_depth(filled, w, h, max_depth_mm):
    """Djupbilden i färg (nära = rött, långt bort = blått, bortom max_depth_mm grått, okänt svart), skalad till w x h."""
    z = cv2.resize(filled, (w, h), interpolation=cv2.INTER_NEAREST)
    norm = np.clip((np.nan_to_num(z, nan=0) - 500) / (max_depth_mm - 500), 0, 1)
    img = cv2.applyColorMap((255 - norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[np.isnan(z)] = (0, 0, 0)
    img[np.nan_to_num(z, nan=0) > max_depth_mm] = (80, 80, 80)
    return img


def crop_sheet(result, apples, label, path, cols=8, tile=120):
    """Spara ett utklipp per äpple, ur vyn med högst confidence.

    Parametrar:
        result: från pipeline_3d.analyse().
        apples: äpplena som ska visas (rader i result["apples"]).
        label: funktion äppelrad -> text som skrivs på utklippet.
        path: bildfil att skriva.
        cols, tile: antal kolumner och utklippens storlek i px.
    """
    views = result["views"]
    tiles = []
    for apple in apples.itertuples():
        own = views[(views.apple == apple.apple) & (views.z_source == "apple")]
        if not len(own):
            continue
        best = own.loc[own.conf.idxmax()]
        img = cv2.imread(str(result["frames"][int(best.frame)]))
        half = int(max(24, 1.4 * best["size"]))
        padded = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_REFLECT)
        cx, cy = int(best.cx) + half, int(best.cy) + half
        crop = cv2.resize(padded[cy - half: cy + half, cx - half: cx + half], (tile, tile), interpolation=cv2.INTER_CUBIC)
        box_half = best["size"] / 2 * tile / (2 * half)
        cv2.rectangle(crop, (int(tile / 2 - box_half), int(tile / 2 - box_half)),
                      (int(tile / 2 + box_half), int(tile / 2 + box_half)), (255, 255, 0), 1)
        put_text(crop, label(apple), (3, tile - 6), 0.4)
        tiles.append(crop)
    if tiles:
        cv2.imwrite(str(path), tile_grid(tiles, cols))


# ---------------------------------------------------------------- diagram

def depth_check_plot(measured, scale, path, ref_diameter_mm):
    """Diagram för kalibreringen: äpplets storlek i pixlar krymper med avståndet; i mm (pixlar x avstånd) ska den inte
    göra det.

    Parametrar:
        measured: de helt synliga kalibreringsäpplena (kolumner z_mm, size_px, diameter_raw_mm).
        scale: den anpassade storleksskalan.
        path: bildfil att skriva.
        ref_diameter_mm: referensäpplets diameter (ritas som en linje).
    """
    z = measured.z_mm.to_numpy() / 1000
    surface, ink, ink2, grid, series = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3", "#2a78d6"
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=surface)
    for a in ax:
        a.set_facecolor(surface)
        a.spines[["top", "right"]].set_visible(False)
        a.spines[["left", "bottom"]].set_color(grid)
        a.grid(axis="y", color=grid, linewidth=0.8)
        a.set_axisbelow(True)
        a.tick_params(colors=ink2, length=0)
        a.set_xlabel("avstånd till äpplet (m)", color=ink2)
    ax[0].scatter(z, measured.size_px, s=30, color=series, edgecolor=surface, linewidth=1.2)
    ax[0].set_ylabel("äpplets storlek i bilden (px)", color=ink2)
    ax[0].set_ylim(0, float(measured.size_px.max()) * 1.15)
    ax[0].set_title("I pixlar: äpplen långt bort ser mindre ut", loc="left", color=ink, fontsize=11)
    diameter_mm = measured.diameter_raw_mm * scale
    ax[1].axhline(ref_diameter_mm, color=ink2, linewidth=1)
    ax[1].scatter(z, diameter_mm, s=30, color=series, edgecolor=surface, linewidth=1.2, zorder=3)
    ax[1].text(1.01, ref_diameter_mm, f"referens-\näpple\n{ref_diameter_mm:.0f} mm", color=ink2, fontsize=9, va="center",
               transform=ax[1].get_yaxis_transform())
    ax[1].set_ylabel("äpplets diameter från djupet (mm)", color=ink2)
    ax[1].set_ylim(0, max(100, float(diameter_mm.max()) + 10))
    ax[1].set_title("I mm, med djup: storleken beror inte längre på avståndet", loc="left", color=ink, fontsize=11)
    fig.suptitle(f"Helt synliga äpplen från träd 1 som användes för kalibreringen (n = {len(measured)})", x=0.01, ha="left", color=ink, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=surface)
    plt.close(fig)


# ---------------------------------------------------------------- videor

def summary_panel(img, s):
    """Gör bilden mörkare och skriv sammanfattningen (summary-dict från pipeline_3d.summarize) på den."""
    H, W = img.shape[:2]
    out = (img * 0.3).astype(np.uint8)
    x0, y0, x1, y1 = W // 2 - 450, H // 2 - 235, W // 2 + 450, H // 2 + 235
    cv2.rectangle(out, (x0, y0), (x1, y1), (20, 20, 20), -1)
    cv2.rectangle(out, (x0, y0), (x1, y1), (200, 200, 200), 2)
    x, y = x0 + 40, y0 + 50
    gray, light = (190, 190, 190), (150, 150, 150)
    put_text(out, f"{s['recording']}  ({s['tree']})  - 3D", (x, y), 0.75, gray)
    put_text(out, f"{s['apples_counted']} räknade äpplen", (x, y + 65), 1.5, (120, 255, 120), 3)
    put_text(out, f"rimligt intervall {s['count_low']} - {s['count_high']}  (medel av 12 räkningar)", (x, y + 100), 0.75, gray)
    put_text(out, f"Uppskattad skörd: {num(s['estimate_kg'])} kg", (x, y + 155), 1.25, (255, 255, 255), 2)
    put_text(out, f"rimligt intervall {num(s['kg_low'])} - {num(s['kg_high'])} kg", (x, y + 190), 0.75, gray)
    put_text(out, f"medelvikt per äpple {s['mean_weight_g']:.0f} g ({s['weight_low_g']:.0f}-{s['weight_high_g']:.0f}), "
                  f"mediandiameter {s['median_diameter_mm'] or 0:.0f} mm", (x, y + 235), 0.7, gray)
    put_text(out, s["weight_source"], (x, y + 265), 0.6, gray)
    put_text(out, "storleksskala från referensäpplet på 109 g (träd 1)", (x, y + 293), 0.6, gray)
    put_text(out, f"avståndsgräns {num(s['max_depth_m'])} m: {s['background_removed']:.0f} föremål i bakgrunden räknas inte",
             (x, y + 321), 0.6, gray)
    put_text(out, "Bara synliga äpplen, ena sidan av trädet", (x, y + 360), 0.6, light)
    return out


def render(result, summary, visible, out, depth_frames, conf_keep, max_depth_mm):
    """Skriv annotated_3d.mp4 och final.png: räknade äpplen i grönt, uppmätta äpplen i cyan med sin storlek, bakgrund i
    rött, en djupbild i hörnet, och sist sammanfattningen.

    Parametrar:
        result: från pipeline_3d.analyse().
        summary, visible: från pipeline_3d.summarize().
        out: mapp för resultaten.
        depth_frames: djupbild (fylld, i mm) för varje bildruta, i ordning.
        conf_keep: detektioner med minst denna confidence ritas.
        max_depth_mm: avståndsgränsen (detektioner bortom den ritas röda).
    """
    frame_paths, detections, counted, views = result["frames"], result["det"], result["counted"], result["views"]
    H, W = cv2.imread(str(frame_paths[0])).shape[:2]
    video = cv2.VideoWriter(str(out / "annotated_3d.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    shown = {f: g for f, g in detections[detections.conf >= conf_keep].groupby("frame")}
    counted_by_frame = {f: g for f, g in counted.groupby("frame")}
    diameter_of = visible.set_index("apple").diameter_mm.to_dict()
    measured_by_frame = {f: g for f, g in views[views.apple.isin(diameter_of)].groupby("frame")} if len(views) else {}
    inset_w, inset_h = 320, 180
    running, img = 0, None
    for frame, (frame_path, filled) in enumerate(zip(frame_paths, depth_frames)):
        img = cv2.imread(str(frame_path))
        for box in shown.get(frame, pd.DataFrame()).itertuples():
            p1, p2 = (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2))
            if box.far:
                cv2.rectangle(img, p1, p2, (0, 0, 255), 2)
                put_text(img, f"{num(box.z_mm / 1000)} m", (p1[0], p1[1] - 4), 0.45, (120, 120, 255))
            else:
                cv2.rectangle(img, p1, p2, (170, 170, 170), 1)
        counted_here = counted_by_frame.get(frame, pd.DataFrame())
        running += len(counted_here)
        for box in counted_here.itertuples():
            cv2.rectangle(img, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), (0, 255, 0), 2)
        for box in measured_by_frame.get(frame, pd.DataFrame()).itertuples():
            cv2.rectangle(img, (int(box.x1) - 2, int(box.y1) - 2), (int(box.x2) + 2, int(box.y2) + 2), (255, 255, 0), 2)
            put_text(img, f"{diameter_of[box.apple]:.0f} mm", (int(box.x1), int(box.y1) - 6), 0.5, (255, 255, 0))
        n_far = int(shown[frame].far.sum()) if frame in shown else 0
        cv2.rectangle(img, (0, 0), (W, 34), (0, 0, 0), -1)
        put_text(img, f"bildruta {frame}   räknade hittills (denna räkning): {running}   avståndsgräns {num(max_depth_mm / 1000)} m"
                      f"   bortom gränsen i denna bildruta: {n_far}", (10, 24), 0.6)
        # teckenförklaring
        lx, ly = 10, H - 70
        cv2.rectangle(img, (lx - 6, ly - 20), (lx + 330, H - 8), (0, 0, 0), -1)
        for j, (color, text) in enumerate((((0, 255, 0), "räknas i denna bildruta"), ((255, 255, 0), "helt synligt: mätt i mm"),
                                           ((0, 0, 255), "för långt bort (bakgrund): räknas inte"))):
            cv2.rectangle(img, (lx, ly - 10 + 20 * j), (lx + 14, ly + 2 + 20 * j), color, 2)
            put_text(img, text, (lx + 24, ly + 2 + 20 * j), 0.45)
        # djupbild i hörnet
        x0, y0 = W - inset_w - 10, H - inset_h - 10
        img[y0:y0 + inset_h, x0:x0 + inset_w] = colorize_depth(filled, inset_w, inset_h, max_depth_mm)
        cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + inset_w, y0 + inset_h), (255, 255, 255), 1)
        put_text(img, f"djup: rött 0,5 m - blått {max_depth_mm / 1000:.0f} m, grått = för långt", (x0, y0 - 6), 0.42)
        video.write(img)
    final = summary_panel(img, summary)
    cv2.imwrite(str(out / "final.png"), final)
    for _ in range(HOLD_SECONDS * FPS):
        video.write(final)
    video.release()


def render_audit(result, out, hold_seconds=2):
    """Långsam video för att kontrollera räkningen med ögat: varje räknat äpple får ett nummer i den bildruta där det
    räknas, och den bildrutan hålls stilla. Skriver audit_numbered.mp4 och audit_apples.csv (de numrerade äpplena)."""
    frame_paths, counted = result["frames"], result["counted"].sort_values(["frame", "cx"])
    H, W = cv2.imread(str(frame_paths[0])).shape[:2]
    video = cv2.VideoWriter(str(out / "audit_numbered.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    numbers, n = [], 0
    counted_by_frame = {f: g for f, g in counted.groupby("frame")}
    for frame, frame_path in enumerate(frame_paths):
        img = cv2.imread(str(frame_path))
        counted_here = counted_by_frame.get(frame)
        if counted_here is None:
            cv2.rectangle(img, (0, 0), (W, 34), (0, 0, 0), -1)
            put_text(img, f"bildruta {frame}   räknade äpplen hittills: {n}", (10, 24), 0.65)
            video.write(img)
            continue
        for _, box in counted_here.iterrows():
            n += 1
            numbers.append(dict(number=n, frame=frame, x=round(box.cx), y=round(box.cy), size_px=round(box["size"], 1),
                                conf=round(box.conf, 2), distance_m=round(box.z_mm / 1000, 2) if np.isfinite(box.z_mm) else None))
            cv2.rectangle(img, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), (0, 255, 0), 2)
            put_text(img, str(n), (int(box.x2) + 2, int(box.y1) + 10), 0.5, (0, 255, 0))
        cv2.rectangle(img, (0, 0), (W, 34), (0, 0, 0), -1)
        put_text(img, f"bildruta {frame}   räknade äpplen hittills: {n}   (nya nummer i denna bildruta: {n - len(counted_here) + 1}-{n})", (10, 24), 0.65)
        for _ in range(hold_seconds * FPS):
            video.write(img)
    video.release()
    pd.DataFrame(numbers).to_csv(out / "audit_apples.csv", index=False)
