# Kugellager Experiment 


## Verusch 1 (Shot in the dark)
Ich hab ein gefettetes und ein trockenes Lager. 
Die Lager sind im Skateboard eingebaut und ich hab das Handy in der mitte liegen während ich alle Aufnahmen mache.
Beide sind schon stark verschliessen, neue Lager sind leider zu leise. 


### Audio Vorbereitung 
Erstmal muss ich aus den langen Audioaufnahmen kleine Sampels machen.
Dafür habe ich sie mir in Audacity angeschaut um erstmal selber zu sehen was ich aufgenommen habe. 
Positiv überrascht hat mich direkt das man mit bloßen Auge am Spektogramm die beiden Aufnahemen unterscheiden könnte. 
Ich habe mir ein Python Skript gennerieren lassen das immer wenn der Ton unter -45db fällt einen Cut setzt. 
Ich hab 14 sampels on den gefetteten und 26 von den trockenen. 

### Daten erkunden und Features 
Ich habe meine sampels die von unterschiedlicher länge wahren 2-5sec in 1sec Segmente geteilt. So hab ich aus meinen 40 Samples, 104 Segmente machen können. 
Ich habe erstmal ein Feature gewhählt und zwar Mel-Spektogramm der druchschnitt pro Zeile. 

### Model 
Ein Desision Tree mit einer Schicht hat es geschafft jedes Sample perfekt hervor zu sagen. 
Dafür hatte er nur eine Regel (duh), "mel_35 <= -59.8", sonst nichts. 
Stellt sich heraus, die schöne Eigenschafft die wir vorher gesehen haben mit der Form der Spektogramme hat zur Folge das es reicht den durchschnittswert einer der Obersten Zeilen zu vergleichen um die Vorhersagen zu treffen. 
Es kommt dazu das ich für diesen ersten Versuch sehr wenige Samples hatte, aber das ist nicht das eizige Problem. Auch mit mehr Samples denke ich wäre das Ergebniss nicht viel anders gewesen. Was ich zusätzlich zu mehr Samples brauche, ist mehr Variation innerhalb der einzellnen Klassen. 
Fett und Trocken waren jeweils ein Kugellager Paar aus einer Position aufgenommen. 
Ich muss mir einen neuen Versuchsaufbau überlegen und auch überlegen welche Klassen ich haben will. Alte Lager sind schön laut und haben ein eindeutiges Roll, geräusch. Neue Lager sind sehr leise, und haben nicht viel was sie hergeben. 

## Versuch 2 (Ait Ben Ahmed, Dataset)
Anstatt meine Energie in einen Versuchsaufbau zu stecken möchte ich mich lieber auf die Daten Seite konzentrieren. Zum Glück habe ich Online dieses Dataset gefunden: 
**Ait Ben Ahmed, Abdelbaset (2023), “Sound Datasets of a Rolling Element Bearing under Various Operating Conditions”, Mendeley Data, V1, doi: 10.17632/n9y9c7xrz3.1**

Es gibt 2 Datasets zur Auswahl, eines das mit einem stethoscope direkt am Lagergehäuse aufgenommen wurde und eines das mit einem Microfone im Raum aufgenommen wurde. Ich wähle jetzt den ersten Datensatz weil ich mir erhoffe das die Ergebisse des Stethoscopes klarere Muster mit weniger Rauschen liefern. 