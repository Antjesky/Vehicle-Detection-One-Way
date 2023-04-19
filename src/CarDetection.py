
# Fahrzeug-Objekterkennungsmodell mit Detektionstransformator (DETR)
# Imports
!{sys.executable} - m pip install IProgress opencv-python ipywidgets transformers timm torch == 1.10.1+cu111 torchvision == 0.11.2+cu111 torchaudio == 0.10.1 - f https: // download.pytorch.org/whl/cu111/torch_stable.html

!jupyter nbextension enable - -py widgetsnbextension


import sys
import os
import json
import cv2
import torch
from tqdm.auto import tqdm
import IProgress
import requests
from collections import defaultdict
from PIL import Image, ImageDraw
from transformers import pipeline, DetrImageProcessor, DetrForObjectDetection

# CUDA-fähige CPU
# Prüfen einer CUDA-fähigen CPU - Festlegung Ausführungsort des Scriptes
if torch.cuda.is_available():
    print("Running on CUDA device")
    device = torch.device("cuda")
else:
    print("WARNING: Running on CPU")
    device = torch.device("cpu")

# Funktion, die Frames aus einem Video extrahiert


def video_to_frames(
        in_file: str,
        out_dir: str,
        fps: int = 30
):
    # Überprüfen, ob das Ausgabe-Verzeichnis 'out_dir' bereits existiert und nicht leer ist.
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) != 0:
        # Überspringe Extraktion der Frames, da sie bereits existieren
        print("Skipping frame extraction since they already exist")
        return

    # Wenn das Ausgabe-Verzeichnis 'out_dir' nicht existiert, eines erstellen.
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Standard-Opencv-Videoerfassung 'VideoCapture' Objekt aus der OpenCV Bibliothek erzeugen.
    vid = cv2.VideoCapture(in_file)

    count = 0
    while True:
        ret, frame = vid.read()

        if count % fps == 0:
            cv2.imwrite(os.path.join(out_dir, f"frames_{count}.png"), frame)

        if not ret:
            break
        count += 1

    vid.release()


# Extrahieren der Bilder 'extract frames'
# 'fps'=30 bedeutet, dass jeder 30. Frame extrahiert wird
video_to_frames(
    "Vehicle Detection Junktion 1.mp4",
    out_dir="extracted_frames",
    fps=30
)

# Objeterkennung in den extrahierten Bildern
# Festlegung Speicherpfad Analysebilder, Ergebnisbilder und Nutzung des Modells "facebook/detr-resnet-101"


def detect_objects_in_frames(
    in_dir: str,
    out_dir: str = "detected_frames",
    model_name: str = "facebook/detr-resnet-101",
    confidence=0.9  # Wahrscheinlichkeitsschwelle 90%
):

    # Laden aller Bilder in den Arbeitsspeicher
    frames = {}

    # Jedes Bild wird geladen und dessen Dimensionen (Breite, Höhe) gespeichert
    for image_name in os.listdir(in_dir):
        image = Image.open(os.path.join(in_dir, image_name))
        frames[image_name] = image

    # Laden ImageProcessor-Objekt, das für das Modell benötigt wird
    processor = DetrImageProcessor.from_pretrained(model_name)
    # Laden des Modells
    # .to('cuda') platziert das Modell auf der GPU (schlägt fehl, wenn die GPU nicht erkannt wird)
    model = DetrForObjectDetection.from_pretrained(model_name).to(device)

    # Initialisierung eines defaultdict-Objekts zum Zählen der erkannten Objekte
    label_count = defaultdict(int)

    # Iterations-Schleife über alle Frames
    for frame_name in tqdm(frames.keys()):
        frame = frames[frame_name]
        # Vorverarbeitung des Frames entsprechend den Anforderungen des Modells
        # z.B. Skalierung der Pixelwerte auf einen bestimmten Bereich
        # .to(device) verschiebt das verarbeitete Frame auf die GPU, falls vorhanden
        frame_processed = processor(
            images=frame, return_tensors="pt").to(device)

        # Modellvorhersage auf dem Frame ausführen
        detections = model(**frame_processed)

        # Ergebnisse aufbereiten und Bounding Boxes zeichnen
        # invertiert die Bildabmessungen, z.B. (Breite, Höhe) -> (Höhe, Breite)
        target_sizes = torch.tensor([frame.size[::-1]])

        # post-processing für das Modell
        # Schwelle bedeutet Modellvertrauen -> ggf. anpassen
        results = processor.post_process_object_detection(
            detections, target_sizes=target_sizes, threshold=confidence)[0]

        # PIL ImageDraw zum Zeichnen von Bounding Boxes / Text in das Bild
        draw = ImageDraw.Draw(frame)
        # für jede Erkennung (bestehend aus Score, Label und Bounding Box)
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            # das Modell arbeitet mit numerischen Labels
            # model.config.id2label enthält ein Dictionary, das beispielsweise 0 -> Bus, 1 -> Auto usw. zuordnet
            print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )
            x, y, x2, y2 = tuple(box)
            # Zeichne ein Rechteck um die Erkennung mit einer roten Umrandung und einer Breite von 1px
            draw.rectangle((x, y, x2, y2), outline='red', width=1)
            # Text in Schwarz zeichnen
            draw.text(
                (x, y), model.config.id2label[label.item()], fill='black')

            # den Zähler um 1 erhöhen
            label_count[model.config.id2label[label.item()]] += 1

        # Frame-Out-Verzeichnis erstellen, wenn es nicht vorhanden ist
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # Bild speichern (enthält jetzt die Begrenzungsrahmen - bounding boxes)
        frame.save(os.path.join(out_dir, f"{frame_name}.png"))

    # Enthält alle erkannten Objekte und deren Häufigkeit - zurück, um anzuzeigen.
    return label_count


# Ausgabe der Objekterkennung über den Wahrscheinlichkeitsschellenwert 0,9
"""
Apply the model to the extracted frames in 'in_dir' and save to 'out_dir'. Use the model 'model_name', e.g.
one of the below models from Facebook.
The model returns only detections above the 'confidence' threshold.
"""

# model_name = "facebook/detr-resnet-101"
model_name = "facebook/detr-resnet-50"

# Anzahl der Erkennungen zählen
label_count = detect_objects_in_frames(
    in_dir="extracted_frames",
    out_dir=f"detections-{model_name}",
    model_name=model_name,
    confidence=0.9
)

# Ausgabe Anzahl der erkannten Fahrzeuge
for (label, num) in label_count.items():
    print(f"Detected {num}\t{label}s")

# Ausgabe in eine Json-Datei
# Erkennungen in Datei ausgeben
with open(f"detections-{model_name}.json", "w") as f:
    json.dump(label_count, f, indent=4)
