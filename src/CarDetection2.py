# Fahrzeug-Instanz-Segmentierungsmodell mit Maskformer
# Import
import os
import json
import cv2
import torch
import numpy as np
from tqdm.auto import tqdm
import IProgress
import requests
from collections import defaultdict
from PIL import Image, ImageDraw
from transformers import pipeline, DetrImageProcessor, DetrForObjectDetection, MaskFormerImageProcessor, MaskFormerForInstanceSegmentation
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches

# Prüfung CUDA-fähige Grafikkarte
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


# Extrahieren der Bilder aus dem Video-Stream
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
    confidence=0.9
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
# print detections to file
with open(f"detections-{model_name}.json", "w") as f:
    json.dump(label_count, f, indent=4)


 def ade_palette():
        """ADE20K palette that maps each class to RGB values."""
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]


### Instanz-Segmentierung in Einzelbildern
def instance_segmentation_in_frames(
    in_dir: str,
    out_dir: str = "detected_frames",
    model_name: str = "facebook/detr-resnet-101",
    confidence = 0.9
):

    # Lade alle Bilder in den Arbeitsspeicher
    frames = {}

    # Lade jedes Bild und dessen Abmessungen (Breite, Höhe)
    for image_name in os.listdir(in_dir):
        image = Image.open(os.path.join(in_dir, image_name))
        frames[image_name] = image

    # Lade den benötigten Bildprozessor für das spezifische Modell
    processor = MaskFormerImageProcessor.from_pretrained(model_name)
    # Lade das spezifische Modell
# .to('cuda') platziert das Modell auf der GPU. Falls keine GPU verfügbar ist, schlägt dies fehl.
    model = MaskFormerForInstanceSegmentation.from_pretrained(model_name).to(device)

    # Initialisierung eines defaultdict-Objekts zum Zählen der erkannten Objekte
    label_count = defaultdict(int)

    color_palette = np.array(ade_palette())

    # Iteriere über alle Bilder
    for frame_name in tqdm(frames.keys()):
        frame = frames[frame_name]
        # Führe die benötigte Vorverarbeitung für das Modell durch
        # z.B. skalieren der Pixelwerte auf einen bestimmten Bereich
        # Setze das verarbeitete Bild erneut auf die GPU, falls verfügbar
        frame_processed = processor(images=frame, return_tensors="pt").to(device)

        with torch.no_grad():
            # das Modell auf einem bestimmten Frame laufen lassen
            detections = model(**frame_processed)

        # Postprocessing für das spezifische Modell
        # Schwellenwert bedeutet Vertrauen in das Modell -> ggf. anpassen
        results = processor.post_process_panoptic_segmentation(detections, target_sizes=[frame.size[::-1]])[0]

        segmentation = results["segmentation"].to('cpu')
        segments = results["segments_info"]

        color_seg = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)

        # für jedes Segment die Legende zeichnen
        for segment in segments:
            segment_id = segment['id']
            segment_label_id = segment['label_id']
            segment_label = model.config.id2label[segment_label_id]

            label_count[segment_label] += 1

            visual_mask = segmentation == segment_id

            color_seg[visual_mask] = color_palette[segment_label_id]

        color_seg = color_seg / 255.
        frame_np = np.asarray(frame.convert("RGB"))
        frame_np = frame_np / 255.

        frame_and_color_seg = color_seg*0.5 + frame_np*0.5

        out_image = Image.fromarray((frame_and_color_seg*255).astype(np.uint8))

        # Frame-Out-Verzeichnis erstellen, wenn es nicht existiert
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # Frame speichern (er enthält jetzt die Bounding Box)
        out_image.save(os.path.join(out_dir, f"{frame_name}.png"))


    return label_count


### Laden des vortrainiertes Modells für die Instanzsegmentierung von Facebook AI Research
# Definiere das zu verwendende Modell für die Instanzsegmentierung
instance_segmentation_model = "facebook/maskformer-swin-base-coco"
# Alternativ kann auch das größere Modell "facebook/maskformer-swin-large-coco" verwendet werden
instance_segmentation_model = "facebook/maskformer-swin-large-coco"

# Führe die Instanzsegmentierung auf allen Frames des input-Verzeichnisses "extracted_frames" aus und speichere die Ergebnisse im output-Verzeichnis
# Die Variable "label_count" enthält ein defaultdict mit der Anzahl der Instanzen pro Klassenlabel.
label_count = instance_segmentation_in_frames(
    in_dir="extracted_frames",
    out_dir=f"detections-{instance_segmentation_model}",
    model_name=instance_segmentation_model,
    confidence=0.9
)

### Ausgabe Anzahl der erkannten Fahrzeuge
for (label, num) in label_count.items():
    print(f"Detected {num}\t{label}s")

### Ausgabe in eine Json-Datei
# Erkennungen in Datei ausgeben
with open(f"detections-{model_name}.json", "w") as f:
    json.dump(label_count, f, indent=4)
