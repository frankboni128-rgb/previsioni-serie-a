# Previsioni Serie A

Modello statistico per la Serie A (Elo, Poisson/Dixon-Coles, expected goals, valori di mercato delle rose).
GitHub Actions lo esegue ogni mattina e aggiorna la pagina in `docs/` (GitHub Pages).

- `modello.py` — scarica i dati, rifà i test sul passato, calcola le previsioni della prossima giornata
- `docs/index.html` — la pagina consultabile
- `docs/dati.json`, `docs/previsioni_giornata.csv`, `docs/storico.csv` — generati automaticamente

Partite da aggiungere a mano (se il calendario automatico manca): variabile `PARTITE_MANUALI` in `modello.py`.
