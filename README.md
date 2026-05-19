# Robotski vid - Izziv 9HPT

Projekt za analizo gibanja roke pri testu devetih zatičev.

Cilj projekta je iz videoposnetka določiti kinematične parametre gibanja roke:
- trajektorijo gibanja,
- dolžino poti,
- hitrost,
- pospešek.

## Datoteke

- `analyze_motion.py`: analiza gibanja iz videa
- `rezultati.py`: čiščenje in obdelava rezultatov
- `summary_results.py`: priprava skupnega povzetka rezultatov
- `track_hand.py`: sledenje roki
- `Video.py`: pomoč pri delu z videom
- `Dockerfile`: okolje za zagon

## Docker

Zgradimo Docker sliko:

```bash
docker build -t rv-izziv .