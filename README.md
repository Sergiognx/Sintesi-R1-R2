# Sintesi R1–R2

Applicazione Python con interfaccia grafica (Tkinter) per la sintesi di regolatori R1–R2 nel dominio della frequenza.

L’applicazione guida passo passo nella progettazione del regolatore a partire dalle specifiche statiche e dinamiche, mostrando tutti i passaggi di calcolo e i risultati grafici.

---

## Funzionalità principali

- Inserimento della funzione di trasferimento G(s)
- Progetto di R1 per il rispetto dell’errore a regime su rampa unitaria
- Calcolo del coefficiente di smorzamento ζ a partire dall’overshoot S%
- Calcolo del margine di fase φm
- Calcolo delle pulsazioni ωn e ωc
- Sintesi di R2 come:
  - rete anticipatrice (lead)
  - rete attenuatrice (lag)
- Diagrammi di Bode di:
  - G(jω)
  - R1(jω)G(jω)
  - L(jω) = R(jω)G(jω)
- Risposta allo scalino unitario:
  - del sistema G(s)
  - del sistema in anello chiuso

---

## Requisiti

- Python 3.x
- numpy
- matplotlib
- tkinter

Nota: tkinter è incluso di default in Python su Windows e macOS. Su alcune distribuzioni Linux potrebbe essere necessario installarlo separatamente.

---

## Installazione

1. Clonare il repository oppure scaricare i file del progetto
2. Installare le dipendenze Python eseguendo:

pip install -r requirements.txt

Se tkinter non è presente su Linux:

sudo apt install python3-tk

---

## Avvio dell’applicazione

Eseguire il file principale con:

python sintesi_r1_r2_gui.py

Si aprirà l’interfaccia grafica per l’inserimento dei dati e la visualizzazione dei risultati.

---

## Struttura del progetto

Sintesi-R1-R2/
├── sintesi_r1_r2_gui.py
├── README.md
└── requirements.txt

---

## Contesto didattico

Il progetto è pensato come supporto didattico per corsi universitari di Controlli Automatici, esercitazioni sulla sintesi nel dominio della frequenza e comprensione del legame tra specifiche temporali e frequenziali.

---

## Autore

Sergio Ginex
