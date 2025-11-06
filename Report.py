"""
===========================
Project Report
===========================

Authors: Gössl Marcel, Marek Simon, Schrenk Dominik, Unger Miriam
Date: 06.11.2025
Course: Computer Vision
Github Repo: https://github.com/Dom4i/CV_GarbageProject

----------------------------------------
1. Dataset Description
----------------------------------------
• What dataset did you use?
    Bei dem Datensatz handelt es sich um Bilder von verschiedenen Abfalltypen. Die Daten stammen von Kaggle: https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset/data
    Die Bilder des Datensatzes haben die Dimension 512x384 (pixel)
• How many images/classes does it contain?
    Gesamtanzahl der Bilder: 2527
    Klassen:
        Cardboard: 403
        Glass: 501
        Metal: 410
        Paper: 594
        Plastic: 482
        Trash: 137

• Was the dataset balanced or imbalanced?
    Mit Ausnahme der Klasse "Trash" beeinhaltet jede Klasse zwischen 403 und 594 Bilder. Diese Klassen haben daher keine optimale balance, zeigen aber auch keine großen Unterschiede.
    Die Klasse "Trash" hat allerdings nur 137 Bilder und damit deutlich weniger Daten.
• What was the imaging source or environment?
    Die Bilder wurden mit einer optischen Kamera aufgenommen
    Es handelt sich um Abfall, der auf einem einfärbigen Hintergrund aufgenommen wurde (meist weiß oder grau). Bei der Klasse "Cardboard" aber auch tweilweise "Paper" sind hingegen sind einige Objekte nah an der Kamera, sodass nur das Objekte ohne Hintergrund sichbar ist.
        Daher wird vermutet, dass (mit Ausnahme der "Cardboard" und der "Paper" Bilder ohne Hintergrund) der Hintergrund keinen Bias erzeugt.
• How did you preprocess or split the data?
    Für das Training wurden die Bilder auf 224×224 Pixel skaliert (aus Performance-Gründen), in Tensors konvertiert und mittels ImageNet-Statistiken normalisiert (Mean = [0.485, 0.456, 0.406], Std = [0.229, 0.224, 0.225]). Die Aufteilung in Trainings- und Validierungsmenge erfolgte stratifiziert (80/20) mit fixer Zufallsinitialisierung, um die Klassenverteilung in beiden Splits konsistent zu halten und Reproduzierbarkeit sicherzustellen.

    Datenaugmentation: Zur Robustheit gegenüber leichten Lage- und Helligkeitsänderungen kamen moderat dosierte, zufällige Transformationen zum Einsatz (u. a. RandomResizedCrop mit scale≈0.9–1.0, RandomHorizontalFlip, Rotation ca. ±8–10°, leichtes ColorJitter). Zusätzlich wurden Klassengewichte verwendet, um die Unterrepräsentation der Klasse „Trash“ teilweise auszugleichen. Eine Variante mit originalem Seitenverhältnis (384×512) wurde evaluiert, wegen längerer Laufzeiten jedoch zunächst zurückgestellt
----------------------------------------
2. Relation to Real Applications
----------------------------------------
• How does this project relate to real-world use cases?
• In what kind of system could your model be applied?
• What practical benefits could it have?

    Die Mülltrennung ist essentiell für die Verarbeitung von Abfall. Da große Abfallmengen (wie etwa in Abfallanlagenm Mülldeponien etc.) kaum durch Menschen getrennt werden können, kann diese Technologie helfen (so wie sie es teilweise jetzt auch schon tut).
    Eine Beispielanwendung für ein Klassifizierungsmodell von Müllarten kann die Trennung durch Maschinen sein, wo sich Müllobjekte auf einem Förderband bewegen, automatisch erkannt und dadurch auch getrennt werden.
        Z.B. Überprüfung von schon getrenntem Müll (der von Haushalten bereits in Glas, Plastik, Restmüll etc. getrennt wurde)
        Z.B. Trennung von Mischungen aus verschiedenen Müllarten (z.B. öffentliche Mülleimer, bei denen keine Trennung möglich ist)

----------------------------------------
3. Problems and Solutions
----------------------------------------
• Which problems or challenges did you encounter?
    Während der Entwicklung traten mehrere Herausforderungen auf. Besonders problematisch war die Klasse „Trash“, da sie sehr unscharf definiert ist und Objekte unterschiedlichster Art umfasst –
    von kaputten Gegenständen bis zu Essensresten. Diese fehlende visuelle Einheitlichkeit, kombiniert mit der geringen Bildanzahl (nur 137 Beispiele), erschwerte dem Modell das Lernen konsistenter Merkmale und führte zu unzuverlässigen Vorhersagen.

    Darüber hinaus kam es regelmäßig zu Verwechslungen zwischen den Klassen „Metal“, „Plastic“ und „Glass“. Vor allem bei Flaschen oder Dosen mit Etiketten hatte das Modell Schwierigkeiten, klare Grenzen zu ziehen, da sich
    die Oberflächen und Formen dieser Materialien stark ähneln und teilweise reflektieren. In solchen Fällen wurde beispielsweise eine Plastikflasche mit glänzender Oberfläche häufig als Glas erkannt oder umgekehrt.

• How did you solve or work around them?
    Um diese Probleme abzumildern, wurde eine Reihe von Optimierungen und Experimenten durchgeführt:
    Durch Datenaugmentation (z. B. Rotation, Flip, leichte Farbvariationen) wurde versucht, die Robustheit gegenüber Beleuchtung und Perspektive zu erhöhen.
    Mit Klassengewichten wurde der Einfluss der seltenen Trash-Bilder im Training verstärkt.
    Zusätzlich wurden verschiedene Hyperparameter wie Lernrate, Batchgröße, Epochenzahl und Dropout getestet, um ein stabiles Gleichgewicht zwischen Over- und Underfitting zu erreichen.

    In einem weiteren Versuch wurde die Klasse „Trash“ vollständig entfernt, wodurch sich die Gesamtgenauigkeit deutlich verbesserte. Dies zeigte, dass der Datensatz insbesondere an dieser Stelle strukturelle Schwächen aufweist.
    Dennoch blieb die Klasse im finalen Modell enthalten, um dem ursprünglichen Ziel einer vollständigen Müllklassifizierung gerecht zu werden.

----------------------------------------
4. Implementation Details
----------------------------------------
    Das Projekt wurde vollständig in Python umgesetzt, unter Verwendung der Bibliothek PyTorch für Modellaufbau und Training.
    Zusätzlich kamen torchvision (für Datenvorverarbeitung und Augmentation) sowie scikit-learn (für Metriken und den stratifizierten Datensplit) zum Einsatz.

    Zunächst wurden die Bilder mit passenden Transforms (Resize, Normalisierung, optionale Augmentation) vorbereitet und in Trainings- und Validierungsdaten aufgeteilt.
    Anschließend wurde ein eigenes Convolutional Neural Network (SmallCNN) entworfen, das mehrere Faltungs- und Pooling-Schichten enthält und mit einem linearen Klassifikator abschließt.
    Dieses Modell wurde bewusst kompakt gehalten, um auch auf Geräten ohne High-End-GPU effizient trainiert werden zu können.

    Für das Training wurde der AdamW-Optimizer mit einer lernratenbasierten Anpassung (ReduceLROnPlateau) verwendet. Zur Verbesserung der Robustheit wurden Klassengewichte berücksichtigt und ein moderates Label Smoothing eingesetzt.

----------------------------------------
5. Results and Evaluation
----------------------------------------
    Das Modell wurde in mehreren Trainingsdurchläufen mit unterschiedlichen Parametern und Augmentationseinstellungen evaluiert. Nach einigen Tests mit Lernrate, Batchgröße und Epochenzahl erreichte das finale Modell
    – mit moderater Datenaugmentation und optimaler Epochenwahl – eine Accuracy zwischen 0.66 und 0.75 auf dem Validierungsdatensatz.

    In einer Zwischenversion, in der die Klasse „Trash“ vollständig entfernt wurde, konnte die Genauigkeit sogar auf knapp über 0.8 gesteigert werden.
    Dieses Ergebnis bestätigte, dass die stark variierende und unterrepräsentierte Trash-Klasse einen deutlichen Einfluss auf die Gesamtleistung hatte.

    Zur Auswertung wurden eine Confusion Matrix sowie ein Classification Report sowohl in der Konsole als auch visuell dargestellt, um die Fehlklassifikationen pro Klasse besser analysieren zu können.
    Zusätzlich wurden mehrere Beispielbilder und Grad-CAM-Visualisierungen erstellt, die zeigen, welche Bildbereiche das Modell bei der Klassifikation besonders stark gewichtet.
    Diese Visualisierungen werden auch in der Präsentation verwendet, um das Verhalten des Modells anschaulich zu erklären.

----------------------------------------
6. Discussion and Learnings
----------------------------------------
    Insgesamt zeigte das Projekt sehr deutlich, wie stark die Ergebnisse bei Deep-Learning-Modellen von den gewählten Parametern und Trainingsbedingungen abhängen.
    Schon kleine Änderungen an Lernrate, Batchgröße oder Augmentation führten zu teils relativ spürbaren Unterschieden in der Accuracy. Wir haben gelernt, dass es in der Praxis kein „optimales Rezept“ gibt,
    sondern dass viel Ausprobieren, Beobachten und Anpassen (und Warten) notwendig ist, um ein stabiles Modell zu erhalten.

    Ein weiteres wichtiges Learning war der Einfluss der Hardware-Performance. Während ein Teammitglied mit einer leistungsstarken GPU (RTX 3070 Ti) ein Training in etwa 2–3 Minuten pro Durchlauf durchführen konnte,
    mussten andere Teammitglieder ohne dedizierte GPU oft 10–15 Minuten pro Training warten und konnten daher nur wenige Epochen testen. Diese Einschränkungen waren auch der Grund, warum wir schon früh im Projekt beschlossen,
    die Bildgröße auf 224×224 Pixel zu reduzieren, um die Rechenzeit zu verkürzen und Experimente überhaupt praktikabel zu machen.

    Trotz dieser Hürden war das Projekt lehrreich und zeigte, dass praktische Erfahrung mit Modellen, Datensätzen und Parametern oft wichtiger ist als theoretisches Wissen allein.
    Wir konnten ein grundsätzlich funktionierendes Modell entwickeln, verstehen, wo seine Grenzen liegen, und wertvolle Erfahrung im Umgang mit Trainingsstrategien, Augmentation und Performance-Optimierung sammeln.
"""