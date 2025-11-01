"""
===========================
Project Report
===========================

Authors:
Date:
Course: Computer Vision

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
    Es handelt sich um Abfall, der auf einem einfärbigen Hintergrund aufgenommen wurde (meist weiß oder grau). Bei der Klasse "Cardboard" sind hingegen sind einige Objekte nah an der Kamera, sodass nur das Objekte ohne Hintergrund sichbar ist.
        Daher wird vermutet, dass (mit Ausnahme der "Cardboard" Bilder ohne Hintergrund) der Hintergrund keinen Bias erzeugt.
• How did you preprocess or split the data?

----------------------------------------
2. Relation to Real Applications
----------------------------------------
• How does this project relate to real-world use cases?
• In what kind of system could your model be applied?
• What practical benefits could it have?

    Die Mülltrennung ist essentiell für die Verarbeitung von Abfall. Da große Abfallmengen (wie etwa in Abfallanlagenm Mülldeponien etc.) kaum durch Menschen getrennt werden können, kann technologische Unterstützung helfen.
    Eine Beispielanwendung für ein Klassifizierungsmodell von Müllarten kann die Trennung durch Maschinen sein, wo sich Müllobjekte auf einem Förderband bewegen, automatisch erkannt und dadurch auch getrennt werden.
        Z.B. Überprüfung von schon getrenntem Müll (der von Haushalten bereits in Glas, Plastik, Restmüll etc. getrennt wurde)
        Z.B. Trennung von Mischungen aus verschiedenen Müllarten (z.B. öffentliche Mülleimer, bei denen keine Trennung möglich ist)

----------------------------------------
3. Problems and Solutions
----------------------------------------
• Which problems or challenges did you encounter?
• How did you solve or work around them?
• What changes did you make to improve results?

----------------------------------------
4. Implementation Details
----------------------------------------
• What did you do, step by step?
• Which libraries or methods did you use?
• Why did you choose your specific model or approach?

----------------------------------------
5. Results and Evaluation
----------------------------------------
• How well did your model perform?
• What metrics did you use to measure performance?
• Did you visualize or compare results?

----------------------------------------
6. Discussion and Learnings
----------------------------------------
• What worked well and what didn’t?
• What did you learn from this project?
• How could the project be improved in the future?
"""

# (Optional) Example section for visualizations or statistics
# import matplotlib.pyplot as plt
# import cv2
# import os
# # Add code here if you want to show example images or model outputs
