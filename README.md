# 1D FDTD: вакуум, диэлектрик, дисперсия Лоренца

- Одномерный FDTD-солвер уравнений Максвелла: поля $E_z$, $H_y$, среда немагнитная ($\mu_r = 1$).
- Среды: вакуум, диэлектрик, линейная среда Лоренца.
- Основа — линейная часть статьи Goorjian & Taflove (Optics Letters 17(3), 1992).
- Нелинейности (χ³, Керр, Раман) нет.

## Модель

$$
\frac{\partial^2 P}{\partial t^2} + \delta \frac{\partial P}{\partial t} + \omega_0^2 P
= \varepsilon_0 \omega_0^2 (\varepsilon_s - \varepsilon_\infty) E,
\qquad
D = \varepsilon_0 \varepsilon_\infty E + P
$$

- При $\varepsilon_s = \varepsilon_\infty$ получается $P \equiv 0$: так задаются вакуум и обычный диэлектрик.

## Сборка и запуск

```
cmake -S . -B build && cmake --build build -j
./build/fdtd1d --scenario=lorentz
```

- Нужны CMake 3.24+ и C++17.
- Сценарии: `vacuum`, `dielectric`, `lorentz`.
- Флаги: `--outdir`, `--eps_s`, `--nsteps`.
- Результат: `output/<сценарий>/snapshot_NN.csv` и `index.csv`.

## Графики и проверки

```
uv sync
uv run scripts/plot.py --scenario=lorentz
uv run scripts/verify.py free-space --outdir=output/vacuum
uv run scripts/verify.py fresnel --outdir=output/dielectric
uv run scripts/verify.py dispersion --outdir=output/lorentz
```

- Проверки сравнивают результат с аналитикой; при провале — код выхода 1.
- Скорость в вакууме: расхождение 0.0017%.
- Френель ($n = 1.5$): $r$ и $t$ — расхождение меньше $10^{-4}$ от амплитуды падающей волны.
- Групповая скорость в среде Лоренца: 0.06%.

## Тесты

```
ctest --test-dir build --output-on-failure
uv run pytest tests/
```

- C++ — GoogleTest; CMake скачивает его при первой сборке (нужен интернет).
- Python — pytest.

## Ограничения

- Мягкий источник: импульс идёт в обе стороны, левая копия поглощается границей.
- Граница Мура в среде Лоренца подстроена под несущую; другие частоты слегка отражаются.
- Граница раздела сдвинута на $\Delta x / 2$ (10 нм) влево.
