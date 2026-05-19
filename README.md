# Robotski vid - Izziv 9HPT

Projekt za analizo gibanja roke pri testu devetih zatičev.

## Datoteke
- analyze_motion.py: analiza gibanja iz videa
- rezultati.py: obdelava rezultatov
- summary_results.py: povzetek rezultatov
- track_hand.py: sledenje roki
- Video.py: pomoč pri delu z videom
- Dockerfile: okolje za zagon

## Zagon
```bash
python3 analyze_motion.py --video pot/do/videa.mp4 --out results/video_v3
python3 rezultati.py --input results/video_v3 --output results/video_v3_clean
