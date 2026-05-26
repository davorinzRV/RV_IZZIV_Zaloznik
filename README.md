# Robotski vid - Izziv 9HPT

Projekt analizira gibanje roke pri testu devetih zatičev oziroma 9HPT.

Cilj je iz video posnetkov določiti prijemno točko med palcem in kazalcem, ji slediti skozi čas ter izračunati kinematične parametre:

- trajektorijo,
- dolžino poti,
- hitrost,
- pospešek,
- delež uspešnih zaznav,
- kandidate za prijem oziroma odlaganje zatiča.

## Zagon

Celoten projekt se zažene z ukazom:

```bash
make run (SAMO TO JE TREBA POGNATI)
```

Ukaz samodejno:

1. poišče video posnetke v `/data/patient_001`,
2. za vsak video izvede analizo gibanja,
3. zazna prijemno točko roke,
4. izračuna pot, hitrost in pospešek,
5. očisti koordinate z odstranjevanjem skokov, interpolacijo in glajenjem,
6. pripravi skupno tabelo rezultatov,
7. pretvori rezultate iz pikslov v milimetre,
8. zazna kandidate za prijem oziroma odlaganje zatiča.

## Glavni rezultati

Po zagonu `make run` nastanejo naslednji glavni izhodi:

```text
results/summary_metrics.csv
tables/tabela_porocilo.csv
tables/tabela_porocilo.md
tables/tabela_porocilo_mm.csv
event_results/summary_events_candidates.csv
```

Za vsak video nastaneta tudi mapi:

```text
results/*_v3/
results/*_v3_clean/
```

V njih so shranjeni:

```text
tracking.csv
kinematics.csv
kinematics_clean.csv
metrics.json
metrics_clean.json
annotated.mp4
plot_trajectory_clean.png
plot_path_clean.png
plot_speed_clean.png
plot_acceleration_clean.png
```

## Kandidati za prijem/odlaganje

Dodatna nadgradnja zazna kandidate za prijem oziroma odlaganje zatiča.

Rezultati so v:

```text
event_results/summary_events_candidates.csv
event_results/*_plot_events_clean.png
```

Kandidati so določeni kot lokalni minimumi hitrosti prijemne točke. To niso dokončno potrjeni dogodki, ampak časovne točke, ki so primerne za nadaljnji pregled na označenem videu.

## Docker

Če Docker slika še ni zgrajena, se najprej požene:

```bash
make build
```

Nato:

```bash
make run
```


## Povzetek

Projekt iz video posnetkov 9HPT določi prijemno točko roke, izračuna pot, hitrost in pospešek, rezultate pretvori v milimetre ter dodatno označi kandidate za prijem oziroma odlaganje zatiča.
