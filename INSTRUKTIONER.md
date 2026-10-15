# Så kör du äppelräknaren (steg för steg)

Följ stegen i ordning. Allt görs i **PowerShell** på Windows. Skriv eller klistra in kommandona exakt som de står och tryck Enter efter varje rad.

Du behöver:
- mappen **Archive** (inspelningarna), som du redan har
- internet första gången (för att ladda ner program)
- ungefär 10 GB ledigt utrymme

---

## Steg 1: Installera Python 3.12 (bara första gången)

1. Gå till https://www.python.org/downloads/ och ladda ner **Python 3.12** (inte 3.13 eller nyare).
2. Kör installationen. **Kryssa i "Add python.exe to PATH"** längst ner i första rutan innan du klickar Install.
3. Kontrollera att det fungerar. Öppna PowerShell och skriv:
   ```
   py -3.12 --version
   ```
   Det ska stå något i stil med `Python 3.12.10`. Står det ett fel: installera om och glöm inte krysset.

## Steg 2: Hämta koden

Lägg koden i en mapp **utan å, ä eller ö** i sökvägen, till exempel `C:\apple`. (Programmet kan inte läsa bilder från mappar med å, ä, ö.)

```
cd C:\
git clone https://github.com/Nilswik/-pple.git apple
cd C:\apple
```

Har du inte git? Ladda ner koden som zip från GitHub (grön knapp "Code" → "Download ZIP") och packa upp den så att det blir `C:\apple`.

## Steg 3: Lägg in Archive-mappen

Kopiera hela mappen **Archive** in i `C:\apple`, bredvid mappen `Main`. Det ska se ut så här:

```
C:\apple\
    Archive\        <- inspelningarna
    Main\
    README.md
    requirements.txt
```

Kontrollera: i `C:\apple\Archive` ska det finnas mappar som heter bland annat `test_images` och `Tree2`.

## Steg 4: Installera programmen (bara första gången)

Kör i PowerShell, en rad i taget:

```
cd C:\apple
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
```

**Har datorn ett NVIDIA-grafikkort?** Kör då den här raden (gör allt mycket snabbare):
```
.venv\Scripts\python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```
Inget NVIDIA-kort eller osäker? Hoppa över raden ovan.

Kör sedan:
```
.venv\Scripts\python -m pip install -r requirements.txt
```

Det tar några minuter. Det är klart när du får tillbaka prompten utan röd feltext.

## Steg 5: Kör programmet

```
cd C:\apple\Main
..\.venv\Scripts\python pipeline_3d.py ../Archive/test_images ../Archive/Tree2 ../Archive/1dceb275-6e92-4368-9df5-1526972df094
```

Det tar några minuter (längre utan grafikkort). När det är klart skrivs en sammanfattning ut för varje träd.

**Rätt resultat ser ut så här:**

| Inspelning | Äpplen | Skörd |
|---|---|---|
| test_images | 100 | 11,6 kg |
| Tree2 | 50 | 7,1 kg |
| 1dceb275... | 43 | 5,5 kg |

## Steg 6: Titta på resultaten

Öppna mappen `C:\apple\Results\pipeline_3d`. Där finns en mapp per inspelning med:

- **annotated_3d.mp4**: videon med alla äpplen markerade och sammanfattningen på slutet
- **audit_numbered.mp4**: långsam video där varje räknat äpple har ett nummer, för att kontrollera räkningen
- **final.png**: bara sammanfattningen
- **summary.json** och **comparison.csv**: siffrorna

Färgerna i videon: grön = räknat äpple, cyan = äpple som mätts i mm, röd = för långt bort (bakgrund, räknas inte).

## Steg 7 (valfritt): Kör egna OAK-D-bilder

Har du bilder från OAK-D-kameran (en mapp med `_metadata.json`, färgbild och djupbild per bild, som kameran exporterar)? Lägg mappen på en plats **utan å, ä eller ö** i sökvägen, till exempel `C:\oakd`, och kör:

```
cd C:\apple\Main
..\.venv\Scripts\python oak_d_snapshots.py C:/oakd
```

Resultaten hamnar i `C:\apple\Results\oak_d`:
- en bild per foto med äpplena markerade och djupbilden bredvid
- **overview.png**: alla foton i en bild
- **summary.csv**: antal äpplen och storlek per foto

Färgerna: grön = tydligt synligt äpple, orange = delvis dolt, cyan = uppmätt i mm, röd = för långt bort.

Obs: räkningen gäller per foto. Fotona överlappar, så siffrorna ska inte läggas ihop till ett träd.

---

## Om något går fel

| Felmeddelande | Gör så här |
|---|---|
| `py` känns inte igen | Python är inte installerat med "Add to PATH". Gör om steg 1. |
| `git` känns inte igen | Använd zip-nedladdningen i steg 2. |
| `No such file or directory` / hittar inte `Archive` | Archive ligger fel. Kontrollera steg 3 och att du står i `C:\apple\Main` när du kör steg 5. |
| Bilder kan inte läsas / `!_src.empty()` | Sökvägen innehåller å, ä eller ö. Flytta allt till `C:\apple`. |
| `ModuleNotFoundError` | Steg 4 blev inte klart. Kör om det. |
| Mycket långsamt | Normalt utan NVIDIA-kort. Låt det jobba klart. |
| Steg 7 hittar inga foton | Mappen ska innehålla filer som slutar på `_metadata.json`. Kontrollera sökvägen. |

**Kör aldrig** `pipeline_3d.py --calibrate`. Det skriver över kalibreringen och ändrar resultaten.
