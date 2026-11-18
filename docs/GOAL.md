# Ziel und Idee

Stand: 2026-11-10. Dieses Dokument beschreibt, was das Projekt herausfinden soll. Was dafür konkret läuft, steht in
[PLAN.md](PLAN.md). Die älteren Dokumente (EXPERIMENTS.md, METHOD.md, ICL.md) beschreiben den gesamten Methodenkatalog
im Code; der dort skizzierte Umfang ist durch PLAN.md ersetzt.

## Die Idee

Ein World-Action-Model (WAM, hier LingBot-VA) scheitert unter einer Verschiebung, zum Beispiel weil der Tisch tiefer
steht und der Greifer nicht tief genug geht. Das Modell soll aus diesem eigenen Fehlversuch lernen, **ohne dass
jemand die Korrektur vorgibt**:

1. Das WAM führt eine oder mehrere Episoden aus. Die Fehlversuche werden gespeichert (zunächst nur Fehlversuche).
2. Ein **VLM schaut sich den echten Rollout an** und beschreibt, was falsch war und was anders gemacht werden muss
   („der Greifer hat 3 cm über dem Objekt geschlossen, er muss tiefer“).
3. Zusätzlich wird der **Vorhersagefehler** berechnet: der Verlauf der Abweichung zwischen dem, was das WAM
   vorhergesagt hat, und dem, was wirklich passiert ist.
4. Beides bekommt das WAM beim nächsten Versuch **als Kontext**: den Verlauf des Vorhersagefehlers und die
   VLM-Analyse, dazu je nach Variante den Rollout selbst, nur das Fenster um den Fehler oder nur den Text.
5. Das WAM soll daraus ableiten: „An dieser Stelle habe ich angenommen, dass die Welt so funktioniert. Das stimmte
   nicht, ich muss tiefer greifen.“

## Was herausgefunden werden soll

**Frage 1 (Hauptfrage): Funktioniert die Idee überhaupt?** Gibt es Lebenszeichen, also wird die Policy bei −5 cm und
−10 cm Tischhöhe im zweiten Versuch messbar besser, wenn sie den Fehlerkontext bekommt?

Teilfragen:
- Liefert das VLM eine brauchbare Analyse (Fehlertyp, Richtung, Größe der Korrektur)?
- Nutzt das WAM den Kontext? Dafür wird es trainiert, Kontext zu nutzen (siehe unten), aber **nie auf der Aufgabe,
  auf der getestet wird**.
- Was muss im Kontext stehen: Rollout (vorgestellt + echt), Fehlerkarte, nur das Fenster um den Fehler, nur der Text?
- Wie kommt der Kontext am besten ins Modell: als Tokens im Kontextfenster, über AdaLN/FiLM-Modulation, als
  Prompt-Tokens, als Fast-Weights?

**Frage 2: Funktioniert der Dual-Site-Adapter auf einem WAM?** Das ist die Übertragung von *Few-Shot Residual
Adaptation of Pretrained Latent World Models* (JEPA-Weltmodell + CEM-Planer) auf ein WAM: Adapter innen im
Transformer und außen auf dem vorhergesagten Latent, trainiert **nur mit dem Vorhersagefehler** aus wenigen eigenen
Episoden, ohne VLM und ohne Korrekturvorgabe.

## Wird das WAM dafür trainiert?

Ja, aber nur über kleine Zusatzmodule. Die Gewichte von LingBot-VA bleiben eingefroren.

- **Für Frage 1 (Kontext nutzen, „Meta-Training“):** Ohne Training nutzt LingBot-VA fremden Kontext nicht. Deshalb
  werden Zusatzmodule (Adapter+, LoRA, AdaLN, FiLM, …) auf Sequenzen `[Fehlerkontext | erfolgreicher Versuch auf
  demselben Seed]` trainiert, mit dem Loss nur auf dem erfolgreichen Teil. Das Modell lernt so: „Wenn im Kontext
  dieser Fehler steht, mach es anders.“ Zur Testzeit gibt es dann keinen Gradientenschritt mehr, nur Kontext.
  Trainiert wird auf 7 Aufgaben mit zufälliger Tischhöhe, getestet auf einer 8. Aufgabe, die im Training nie vorkam.
- **Für Frage 2 (Dual-Site):** Der Adapter wird zur Testzeit auf 3 eigenen Episoden der Zielumgebung trainiert, wie
  im Paper. Das ist Test-Time-Adaption, kein Kontext.

## Testaufbau

- **Eine Testaufgabe:** `move_playingcard_away`. In der Baseline gelingt sie bei normaler Tischhöhe in 72 von 72
  Episoden, bei −5 cm in 7 von 72 und bei −10 cm in 0 von 72. Der Fehler ist immer derselbe: Der Greifer schließt zu
  hoch (`missed_grasp`).
- **Zwei Verschiebungen:** Tisch −5 cm und −10 cm (`table_h_m05`, `table_h_m10`).
- **Messgröße:** Anteil der Fehlversuche, die im zweiten Versuch auf demselben Seed gelingen, verglichen mit einem
  zweiten Versuch ohne Lernen (`retry_plain`). Alle Methoden laufen auf denselben Seeds.

## Was als Lebenszeichen zählt

1. Offline: Die VLM-Analyse stimmt mit dem Simulator-Orakel überein (Fehlertyp, Vorzeichen und Größe von Δz).
2. Im Meta-Training: Der Validierungs-Loss ist mit Fehlerkontext niedriger als ohne (auf zurückgehaltenen Seeds).
3. In der Simulation: Die Variante mit Kontext ist besser als (a) der Retry ohne Lernen, (b) derselbe Kontext ohne
   Meta-Training und (c) **derselbe Meta-Adapter ohne Kontext**. Vergleich (c) ist der entscheidende: Er trennt
   „nutzt den Kontext“ von „wurde durch das Training einfach besser bei verschobenen Tischen“.

## Was bisher bekannt ist

- Mit einer **Orakel-Korrektur** und Adapter-Update werden auf der Testaufgabe bei −5 cm 14 von 16 Fehlversuchen im
  zweiten Anlauf gelöst (Retry ohne Lernen: 2 von 17), bei −10 cm 7 von 10 (0 von 20). Das Modell kann eine richtige
  Korrektur also umsetzen. Offen ist, ob das VLM sie liefert und ob Kontext statt Gewichts-Update reicht.
- Ein Adapter-Update **nur aus dem Vorhersagefehler** einer Fehl-Episode liegt bisher auf dem Niveau eines Retries mit
  Exploration (5 von 15 gegenüber 8 von 18). Das eigentliche Dual-Site-Protokoll läuft noch.
- Ohne Meta-Training hat das WAM den Fehlerkontext im ersten Test nicht genutzt (0 von 2, nur ein Einzelfall).

## Einordnung in die Literatur

Stand der Suche vom 2026-11-10, nicht erschöpfend; viele Arbeiten sind nur aus Abstracts bekannt. Keine gefundene
Arbeit kombiniert alle Teile (WAM + eigener Vorhersagefehler + VLM-Analyse + beides als Kontext).

| Arbeit | Was sie macht | Unterschied zur Idee |
|---|---|---|
| [RoboRetry](https://jedward225.github.io/RoboRetry/) | eigener Fehlversuch im Kontext einer VLA; Gewinn vor allem durch gute Korrektur-Sprache | kein Weltmodell, kein Vorhersagefehler; Korrekturen von Menschen annotiert |
| [FAR](https://arxiv.org/abs/2607.01111) | Gewichts-Update aus Fehlversuchen + Exploration | kein Kontext, kein VLM |
| [WAM-TTT](https://arxiv.org/abs/2607.06988) | Test-Time-Training eines WAM über Video-Vorhersage, meta-trainiert | menschliche Videos statt eigener Fehler, kein VLM |
| [World-Coherent Decoding](https://arxiv.org/pdf/2609.02159) | nutzt die Abweichung Vorstellung/Realität | nur zum Ranken von Plänen |
| [Reflective VLA](https://arxiv.org/pdf/2606.25215) | Beobachtung-Aktion-Folge im Kontext | kein Fehlerbezug, kein VLM |
| [FARE](https://arxiv.org/pdf/2609.18016) | Kontext-Reset bei Fehlern in WAMs | keine Analyse, kein Lernen |
| [Dream2Fix](https://arxiv.org/pdf/2603.13528), [FailSafe](https://arxiv.org/pdf/2510.01642) | VLM liefert Fehlertyp und Korrektur | kein Vorhersagefehler, kein WAM-Kontext |
| [AdaJEPA](https://arxiv.org/pdf/2606.32026) | Test-Time-Adaption eines latenten Weltmodells | Gewichte statt Adapter, kein WAM |

RoboRetry nennt als offene Grenze, dass Freitext für Kontakt-Präzision nicht reicht. Genau dort setzt die Idee an:
Das WAM hat eine eigene Vorstellung, die sich mit der Realität vergleichen lässt.
