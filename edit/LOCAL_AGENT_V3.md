# Задание: перерендер v1/v2 с новой логикой склеек и эффектами референса (ПК пользователя)

Репозиторий `C:\Users\andre\max-verstappen-edit`, ветка `main-edit-pipeline`. Медиа уже лежат
в `media/` с прошлого раза (клипы, музыка, `media/title`). Границы те же, что в
`edit/LOCAL_AGENT_PROMPT.md`: работай только в папке репо, видео не коммить, код и таймлайны
коммить и пушь.

## 1. Обновить код
```
git fetch origin
git switch main-edit-pipeline
git pull
python -m pip install opencv-python-headless numpy numba librosa soundfile
```

## 2. Пересобрать таймлайны (планировщик смотрит на настоящий футаж)
```
python edit/build_video_timeline.py edit/cuts_video/v1.json edit/timeline_v1_16x9.json
python edit/build_video_timeline.py edit/cuts_video/v2.json edit/timeline_v2_16x9.json
```
Сборщик печатает таблицу решений: для каждой склейки класс по музыке (drop / phrase / bar /
beat / off / build / silence), силу баса, движение в конце уходящего и в начале входящего
плана (скорость@направление, z = наезд/зум), яркость и выбранный переход с причиной.
Кэш измерений: `media/work/edges/`. Проверь таблицу глазами:
- на дропе (v1 b15, v2 b69) — вспышка/зум (ручные), в тишине v2 b64-b68 — ручной тёмный план;
- whip должен стоять там, где оба плана реально движутся в одну сторону; если направление
  явно не то (например, источник с вшитой графикой даёт ложный поток) — поставь переход руками
  в `fx` этого плана (`"whip"`, `"dip"`, `"cut"`, ...), это отключает автомат на этой склейке;
- `PROBLEM:` строк быть не должно (keep-out, выход за клип, пересечение склейки источника).

## 3. Превью, проверка, финал
```
python edit/render.py edit/timeline_v1_16x9.json media/work/preview/v1.mp4 --preview
python edit/render.py edit/timeline_v2_16x9.json media/work/preview/v2.mp4 --preview
```
Посмотри контакт-листы склеек (`edit/verify_render.py ... sheet.jpg`), особенно:
двухкадровые переходы (bandmix / strobecut / lumamix — в кадре два плана сразу), whiteout
(сквозь засвет должна читаться картинка, не плоский белый), scanline (распад на полосы в
чёрное), бирюзово-розовую вставку `xpro` (v1 b30, v2 b89), финал с радиальными пульсами,
мерцанием, зеркальными панелями и рябью. Потом финальный рендер:
```
python edit/render.py edit/timeline_v1_16x9.json media/output/verstappen_edit_v1_16x9.mp4
python edit/render.py edit/timeline_v2_16x9.json media/output/verstappen_edit_v2_16x9.mp4
python edit/verify_render.py edit/timeline_v1_16x9.json media/output/verstappen_edit_v1_16x9.mp4 media/work/v1_qc.jpg
python edit/verify_render.py edit/timeline_v2_16x9.json media/output/verstappen_edit_v2_16x9.mp4 media/work/v2_qc.jpg
python edit/style_metrics.py media/output/verstappen_edit_v1_16x9.mp4 edit/timeline_v1_16x9.json
python edit/style_metrics.py media/output/verstappen_edit_v2_16x9.mp4 edit/timeline_v2_16x9.json
```
Цели метрик — таблица в `LOCAL_AGENT_PROMPT.md` (вспышки/провалы 3.5-6 в секунду и т.д.).
Копии MP4 — в `C:\Users\andre\Desktop\verstappen_edits\` (имена с суффиксом `_v3`).

## 4. Итог
Закоммить пересобранные `edit/timeline_v*_16x9.json` (и правки cut-листов, если были) и
запушь. Напиши пользователю: пути к MP4, таблицы решений по склейкам (кратко), метрики
«референс / v1 / v2», что пришлось поправить руками и почему.
