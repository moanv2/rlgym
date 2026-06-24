# RLGYM — Drive-the-Arena Presentation

An interactive 3D Rocket League arena that doubles as our Reinforcement-Learning
final-project presentation. Drive the neon arena, reach a glowing pole, press **E** —
that phase's slide deck opens in a browser tab.

## Run it
Open **`index.html`** in a recent browser (Chrome / Edge / Firefox). No install, no server.
Controls: **WASD / arrows** to drive, **SHIFT** to boost, **E** at a glowing pole to open its deck.

The arena and four of the five decks run **fully offline**. The Phase-2 deck
(`rlgym-PPO_loop_and_architecture.html`) is a bundled export that needs internet to render.

## The five phases
| # | Phase | Presenter | Deck |
|---|-------|-----------|------|
| 1 | Intro & Class | Marian | `01_intro_marian.html` |
| 2 | Methodology & Architecture | Diego & Marco | `rlgym-PPO_loop_and_architecture.html` |
| 3 | Our Bots in Detail | Nachi | `bots_champion_style.html` |
| 4 | The Champion | Martin | `martin_champion_deck.html` |
| 5 | Conclusion | Marco | `conclusion_marco.html` |

## What's inside
- `index.html` + `*.js` — the drivable 3D arena (three.js)
- the five `*.html` decks
- `vendor/` — local three.js, anime.js, reveal.js and fonts (so it runs offline)
- `bots_montage.mp4` — the intro montage · `fallback.html` — a no-WebGL backup

The bots are 1v1 Rocket League agents trained from scratch with PPO self-play
(rlgym / rlgym-ppo / RocketSim).
