/* =====================================================================
   RLGYM — all editable copy lives here. Loaded BEFORE app.js.
   Each section is a deck of slides: { k: kicker, slides:[ {h, body}, ... ] }.
   Add/remove slides freely. <span class="fill"> marks a value only you know.
   ===================================================================== */

// Your own song: drop an audio file you legally own next to index.html and name it
// exactly "music.mp3" — it plays and loops automatically. (Works with .mp3/.m4a/.ogg;
// change the name below to match.) Set back to null to use the built-in synth music.
const MUSIC_FILE = "music.mp3";

// Play only PART of the track. Times are in SECONDS. end:null = play to the end.
// Example: { start: 48, end: 72 } loops just the 0:48 to 1:12 section over and over.
const MUSIC_CLIP = { start: 0, end: null };

const LABELS = {
  overview:'Overview', diego:'papaya_1024', approach:'Approach', experiments:'Experiments',
  results:'Tournament', tech:'Tech & MLOps', run:'How to run', team:'Team', repo:'GitHub',
  intro:'intro & class', method:'methodology & architecture', bots:'our bots in detail',
  champion:'the author of the champion', conclusion:'the conclusion'
};

// Signposts in the arena: key, glow color, and [x, z] position.
// FIVE poles = the five presentation phases, in order:
//   1 intro       — Marian         — Intro & Class
//   2 method      — Diego & Marco   — Methodology & Architecture
//   3 bots        — Nachi           — Our Bots in Detail
//   4 champion    — Martin          — The Author of the Champion
//   5 conclusion  — Marco           — The Conclusion
// Drive to a pole + press E: embed.js opens that phase's deck (see DECKS in embed.js).
const SIGNS = [
  { key:"intro",      who:"Marian",        color:0x49e0c8, pos:[   0, -165] },
  { key:"method",     who:"Diego & Marco", color:0x3d7bff, pos:[-125,  -60] },
  { key:"bots",       who:"Nachi",         color:0xb98cff, pos:[ 125,  -60] },
  { key:"champion",   who:"Martin",        color:0xffcf4d, pos:[ -95,   85] },
  { key:"conclusion", who:"Marco",         color:0x7aa6ff, pos:[  95,  160] }
];

// Accents for ALL deck sections, independent of how many physical poles exist.
const SECTION_ACC = { overview:0x3d7bff, diego:0x35d07a, approach:0x7aa6ff, experiments:0xff9a4d, results:0xff6a2b,
  tech:0x49e0c8, run:0x8be06e, team:0xb98cff, repo:0x9fb4e0 };
const ACCENTS = {};
Object.keys(SECTION_ACC).forEach((k,i)=>{ const c=SECTION_ACC[k], r=(c>>16)&255,g=(c>>8)&255,b=c&255;
  ACCENTS[k]={css:'#'+c.toString(16).padStart(6,'0'), rgb:r+', '+g+', '+b, n:String(i+1).padStart(2,'0')}; });

const INFO = {
  overview: { k:"01 — Overview", slides:[
    { layout:"cover", h:"It learns by playing", body:`<p>RLGYM is a Rocket League bot trained with reinforcement learning. It is never told how to play. Instead it sees the game many times a second, tries actions, and earns a reward when something useful happens, like touching the ball or pushing it toward goal. Over millions of steps, Proximal Policy Optimization (PPO) turns those rewards into a policy that wins more than it loses.</p>` },
    { h:"Built like real ML", body:`<p>The agent is trained entirely in simulation, for a one versus one match. The whole project is engineered like real machine learning infrastructure: every run is reproducible, tracked in Weights and Biases, and checked by tests, so any result can be rebuilt from scratch.</p>` }
  ]},

  diego: { k:"papaya_1024 — Diego's bot", slides:[
    { layout:"cover", h:"papaya_1024", body:`<p>A 1v1 Rocket League bot trained <b>from scratch</b> — no human demos — by self-play PPO. This is the environment, the MDP, and the bot.</p><p style="margin-top:12px"><span class="chip">~3.5B self-play steps</span><span class="chip">90 discrete actions</span><span class="chip">0 human demos</span></p><a class="gh" href="diego.html" target="_blank" rel="noopener" style="margin-top:14px">&#9654; Present this part as polished slides</a>` },
    { h:"Three tools, three jobs", body:`<p class="kicker2">Part A · The toolchain</p>
      <div class="rung"><span class="c">rlgym</span><b>the environment API</b><span class="d">Defines what the agent sees, can do, is rewarded for, and where episodes start.</span></div>
      <div class="rung"><span class="c">rlgym_sim</span><b>the physics</b><span class="d">Headless C++ Rocket League (RocketSim) — millions of steps a minute, no game client.</span></div>
      <div class="rung"><span class="c">rlgym-ppo</span><b>the trainer</b><span class="d">Many parallel env workers feeding one central PPO learner.</span></div>
      <div class="rung"><span class="c">RLBot v5</span><b>deployment</b><span class="d">Separate — runs the finished bot in the real game.</span></div>` },
    { h:"Underneath, a Markov Decision Process", body:`<p class="kicker2">Part A · The MDP and the agent's world</p>
      <div class="rung"><span class="c">S · State</span><b>AdvancedObs</b><span class="d">107 numbers</span></div>
      <div class="rung"><span class="c">A · Actions</span><b>LookupAction</b><span class="d">90 discrete</span></div>
      <div class="rung"><span class="c">P · Transition</span><b>RocketSim physics</b><span class="d">the simulator steps the world</span></div>
      <div class="rung"><span class="c">R · Reward</span><b>one scalar / step</b><span class="d">the signal the policy chases</span></div>
      <div class="rung"><span class="c">&gamma; · Discount</span><b>0.99</b><span class="d">seconds of foresight</span></div>
      <p>A state setter spawns each episode (kickoff, aerial, random); the agent acts <b>15 times a second</b>, and the episode ends on a goal or a timeout. The goal: learn a policy <b>&pi;(a|s)</b> that maximizes expected discounted reward.</p>` },
    { h:"rlgym-ppo runs PPO by self-play", body:`<p class="kicker2">Part A · The learner</p>
      <p><b>Self-play:</b> the bot trains against a copy of its own current policy. <b>PPO</b> = policy gradient + actor-critic + on-policy + a clipped objective. Not DQN, SAC or BC — PPO is the stable, on-policy self-play standard.</p>
      <div class="rung"><span class="c">Actor</span><b>plays</b><span class="d">picks the action, plays the game</span></div>
      <div class="rung"><span class="c">Critic</span><b>scores</b><span class="d">scores the state for the advantage (training only)</span></div>
      <p><b>18 parallel workers &rarr; one central learner.</b></p>` },
    { h:"The algorithm is off the shelf; the craft is the reward", body:`<p class="kicker2">Part B · The bot and the reward</p>
      <p><span class="chip">AdvancedObs · 107</span><span class="chip">LookupAction · 90</span><span class="chip">1024×3 net</span><span class="chip">~3.5B steps</span></p>
      <p><b>Zero-sum wrapper · R = my − opp</b> — keeps self-play competitive, not cooperative.</p>
      <p><b>Pruned to 5 reward components across iterations v4 → v7.</b> Sparse goal events plus dense scaffolding. The lesson: more is not better — later versions removed conflict between components.</p>` },
    { layout:"leaderboard", h:"Final standings, as deployed (argmax)", body:`<p class="kicker2">Part B · Results — papaya finished 4th</p>
      <div class="rung" style="--w:.68"><span class="c">🥇 Martin</span><b>68%</b><span class="d">10B steps</span></div>
      <div class="rung" style="--w:.57"><span class="c">🥈 Nachi</span><b>57%</b><span class="d">2.9B steps</span></div>
      <div class="rung" style="--w:.46"><span class="c">🥉 Marco</span><b>46%</b><span class="d">—</span></div>
      <div class="rung" style="--w:.45"><span class="c">4 · papaya_1024</span><b>45%</b><span class="d">3.5B steps · my bot</span></div>
      <div class="rung" style="--w:.23"><span class="c">5 · Marian</span><b>23%</b><span class="d">—</span></div>
      <p>66% in sampling (50 games) became 4th at argmax (full per-pairing eval). <b>Why I lost:</b> beaten on compute (Martin spent ~2.5× my steps), beaten on commitment (Nachi won with fewer steps), and the reward was over-shaped — never sharpened, so it stayed exploitable at argmax.</p>
      <p style="font-style:italic;color:var(--acc)">"I engineered the reward instead of training the policy."</p>` },
    { layout:"media", h:"Diego's original deck", body:`<p>This is Diego's deck rebuilt natively into the arena, so it works offline. For the exact original page (loads online), open it full-screen.</p><a class="gh" href="diego-deck.html" target="_blank" rel="noopener">Open the original &#9658;</a>` }
  ]},

  approach: { k:"02 — Approach", slides:[
    { h:"The loop", body:`<p>Training is a loop. Each step the policy <b>observes</b> the world, takes an <b>action</b>, and receives a <b>reward</b>. PPO uses many of these steps to nudge the policy toward play that wins. Step through the parts with the arrows.</p><a class="gh" href="marco.html" target="_blank" rel="noopener" style="margin-top:14px">&#9654; Present this part as polished slides</a>` },
    { h:"Observation", body:`<p>The policy reads car and ball position, velocity, angular velocity and orientation, plus boost and useful relative distances, so it generalises across the whole pitch. Observation builders live in <code>src/rlbot/obs/</code>.</p>` },
    { h:"Action", body:`<p>Throttle, steer, pitch, yaw and roll, plus jump, boost and handbrake, mapped through a discrete <code>LookupAction</code> table into a compact, learnable set of choices (<code>src/rlbot/actions/</code>).</p>` },
    { h:"Reward", body:`<p>Modular components combined per run. A <code>ZeroSumReward</code> wrapper makes the duel properly competitive, while dense shaping (ball touch, ball-to-goal, boost economy) guides early learning before the sparse goal reward takes over (<code>src/rlbot/rewards/</code>).</p>` },
    { h:"Learning with PPO", body:`<p>PPO's clipped objective keeps each update close to the previous policy so training stays stable, and generalized advantage estimation balances bias against variance. A high discount factor keeps the agent focused on long-horizon outcomes like scoring, not just touching the ball, and the learning rate is tuned against the reward curves.</p>` }
  ]},

  experiments: { k:"03 — Experiments", slides:[
    { layout:"cover", h:"Blank slate to match bot", body:`<p>This is how <b>our</b> bot grows from nothing to match-ready: five stages, each one YAML in <code>configs/experiments/</code>. These are stages of one training pipeline. In the Tournament, our bot then plays four classmates' separately trained bots.</p>` },
    { h:"exp_000 · Random", body:`<p>An untrained policy. It moves without purpose and exists only as the benchmark that every trained version has to beat.</p>` },
    { h:"exp_001 · Baseline", body:`<p>The first PPO policy. It learns the basics: find the ball, make contact, and push it toward the opponent goal.</p>` },
    { h:"rewards · Shaped", body:`<p>A ZeroSum reward plus dense shaping for ball contact, goal alignment and boost economy turns random motion into intent.</p>` },
    { h:"state_setters · Curriculum", body:`<p>Hand-set kickoffs, saves and aerial chances teach the situations that actually decide matches.</p>` },
    { h:"deploy · Match bot", body:`<p>The strongest checkpoint, exported through the RLBot pipeline. This is the agent that plays the 1v1.</p>` }
  ]},

  results: { k:"04 — Tournament", slides:[
    { layout:"cover", h:"Five bots, one bracket", body:`<p>Five teammates each trained their own bot independently (not the training stages above), then ran them against each other. Every bot played a full <b>round robin</b> — many games per pairing, both sides swapped — in two modes. <b>Stochastic</b> samples actions from the policy (more varied, creative play); <b>deterministic</b> always takes the single best action (more consistent). GD is goal difference. Full charts (win-rate convergence, goals, saves, demos, goal margins) live in <code>tournament_results/figures/</code>.</p><a class="gh" href="nachi.html" target="_blank" rel="noopener" style="margin-top:14px">&#9654; Present this part as polished slides</a>` },
    { layout:"leaderboard", h:"Stochastic — round robin", body:`
      <div class="rung" style="--w:.73"><span class="c">🥇 Martin</span><b>73.0%</b><span class="d">10B steps · 876-323 · GD +553</span></div>
      <div class="rung" style="--w:.627"><span class="c">🥈 Nachi</span><b>62.7%</b><span class="d">2.9B steps · GD +307</span></div>
      <div class="rung" style="--w:.563"><span class="c">🥉 Diego</span><b>56.3%</b><span class="d">papaya v7 · GD +153</span></div>
      <div class="rung" style="--w:.35"><span class="c">4 · Marco</span><b>35.0%</b><span class="d">2.0B steps</span></div>
      <div class="rung" style="--w:.228"><span class="c">5 · Marian</span><b>22.8%</b><span class="d">1.35B steps</span></div>` },
    { layout:"leaderboard", h:"Deterministic — round robin", body:`
      <div class="rung" style="--w:.823"><span class="c">🥇 Martin</span><b>82.3%</b><span class="d">10B steps · 988-212 · GD +776</span></div>
      <div class="rung" style="--w:.692"><span class="c">🥈 Nachi</span><b>69.2%</b><span class="d">2.9B steps</span></div>
      <div class="rung" style="--w:.451"><span class="c">🥉 Marco</span><b>45.1%</b><span class="d">2.0B steps</span></div>
      <div class="rung" style="--w:.448"><span class="c">4 · Diego</span><b>44.8%</b><span class="d">papaya v7</span></div>
      <div class="rung" style="--w:.082"><span class="c">5 · Marian</span><b>8.2%</b><span class="d">1.35B steps</span></div>` },
    { layout:"media", h:"Win rate at a glance", body:`<p>Stochastic win rate across the round robin.</p>
      <svg viewBox="0 0 440 200" style="width:100%;height:auto;margin-top:8px" xmlns="http://www.w3.org/2000/svg">
        <g font-family="JetBrains Mono, monospace" font-size="13">
          <rect x="110" y="20"  width="226" height="20" rx="4" fill="#ffcf4d"/><text x="100" y="35"  text-anchor="end" fill="#c4d0ee">Martin</text><text x="344" y="35"  fill="#fff">73.0%</text>
          <rect x="110" y="56"  width="194" height="20" rx="4" fill="#7aa6ff"/><text x="100" y="71"  text-anchor="end" fill="#c4d0ee">Nachi</text><text x="312" y="71"  fill="#fff">62.7%</text>
          <rect x="110" y="92"  width="175" height="20" rx="4" fill="#ff9a4d"/><text x="100" y="107" text-anchor="end" fill="#c4d0ee">Diego</text><text x="293" y="107" fill="#fff">56.3%</text>
          <rect x="110" y="128" width="109" height="20" rx="4" fill="#ff6a2b"/><text x="100" y="143" text-anchor="end" fill="#c4d0ee">Marco</text><text x="227" y="143" fill="#fff">35.0%</text>
          <rect x="110" y="164" width="71"  height="20" rx="4" fill="#93a0c2"/><text x="100" y="179" text-anchor="end" fill="#c4d0ee">Marian</text><text x="189" y="179" fill="#fff">22.8%</text>
        </g>
      </svg>` },
    { layout:"media", h:"Watch a match", body:`<p>Add a clip at <code>clips/final.mp4</code> (or paste a YouTube embed), then swap the placeholder below for a <code>&lt;video&gt;</code> tag — there is a note next to this slide in <code>content.js</code>.</p>
      <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;height:40vh;border:1px dashed rgba(255,255,255,.18);border-radius:16px;background:rgba(10,14,28,.45)"><span style="font-size:38px;line-height:1;opacity:.45">&#9654;</span><span style="font-family:var(--mono);font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted)">clips/final.mp4 — drop your match clip here</span></div>` },
    { h:"Martin — the champion", body:`<div class="botcard"><div class="botpic" data-ph="Add bots/martin.jpg"><img src="bots/martin.jpg" alt="Martin" onerror="this.parentNode.classList.add('empty');this.style.display='none'"></div><div class="bottext"><div class="botslide"><div class="botpic"><img src="martin.png" alt="Martin" onerror="this.remove()"><svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"><rect width="200" height="200" rx="14" fill="#0b0f1c"/><circle cx="100" cy="86" r="82" fill="#ffcf4d" opacity=".14"/><g transform="translate(100 112)"><path d="M-46 6 L-30 -22 L34 -26 L52 -2 L52 14 L-40 18 Z" fill="#ffcf4d"/><path d="M-18 -20 L28 -23 L42 -4 L-22 -1 Z" fill="#ffffff" opacity=".85"/><path d="M-46 6 L52 14 L52 18 L-42 22 Z" fill="#0b1120"/><circle cx="-26" cy="20" r="11" fill="#0b1120" stroke="#ffcf4d" stroke-width="3"/><circle cx="34" cy="20" r="11" fill="#0b1120" stroke="#ffcf4d" stroke-width="3"/><path d="M52 2 q20 4 30 -3" stroke="#ffcf4d" stroke-width="6" fill="none" stroke-linecap="round" opacity=".55"/></g></svg></div><div class="botinfo"><p>Trained the longest by far at <b>10 billion steps</b>, and it shows. Martin topped both rankings: 73.0% stochastic, and a crushing 82.3% deterministic (988-212, goal difference +776). Its game is control. It holds the most possession (54-59%), is the best dribbler in stochastic play, and uses the least boost of anyone, so it wins without wasting a drop. The benchmark everyone else is chasing.</p></div></div></div></div>` },
    { h:"Nachi — the aerial threat", body:`<div class="botcard"><div class="botpic" data-ph="Add bots/nachi.jpg"><img src="bots/nachi.jpg" alt="Nachi" onerror="this.parentNode.classList.add('empty');this.style.display='none'"></div><div class="bottext"><div class="botslide"><div class="botpic"><img src="nachi.png" alt="Nachi" onerror="this.remove()"><svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"><rect width="200" height="200" rx="14" fill="#0b0f1c"/><circle cx="100" cy="86" r="82" fill="#7aa6ff" opacity=".14"/><g transform="translate(100 112)"><path d="M-46 6 L-30 -22 L34 -26 L52 -2 L52 14 L-40 18 Z" fill="#7aa6ff"/><path d="M-18 -20 L28 -23 L42 -4 L-22 -1 Z" fill="#ffffff" opacity=".85"/><path d="M-46 6 L52 14 L52 18 L-42 22 Z" fill="#0b1120"/><circle cx="-26" cy="20" r="11" fill="#0b1120" stroke="#7aa6ff" stroke-width="3"/><circle cx="34" cy="20" r="11" fill="#0b1120" stroke="#7aa6ff" stroke-width="3"/><path d="M52 2 q20 4 30 -3" stroke="#7aa6ff" stroke-width="6" fill="none" stroke-linecap="round" opacity=".55"/></g></svg></div><div class="botinfo"><p>A clear second in both modes (62.7% and 69.2%) on <b>2.9B steps</b>. Nachi lives in the air, with the highest aerial rate of the field (~59%), and is the fastest bot on the pitch. A high-tempo, airborne style that punishes anything slower.</p></div></div></div></div>` },
    { h:"Diego — papaya v7", body:`<div class="botcard"><div class="botpic" data-ph="Add bots/diego.jpg"><img src="bots/diego.jpg" alt="Diego" onerror="this.parentNode.classList.add('empty');this.style.display='none'"></div><div class="bottext"><div class="botslide"><div class="botpic"><img src="diego.png" alt="Diego" onerror="this.remove()"><svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"><rect width="200" height="200" rx="14" fill="#0b0f1c"/><circle cx="100" cy="86" r="82" fill="#ff9a4d" opacity=".14"/><g transform="translate(100 112)"><path d="M-46 6 L-30 -22 L34 -26 L52 -2 L52 14 L-40 18 Z" fill="#ff9a4d"/><path d="M-18 -20 L28 -23 L42 -4 L-22 -1 Z" fill="#ffffff" opacity=".85"/><path d="M-46 6 L52 14 L52 18 L-42 22 Z" fill="#0b1120"/><circle cx="-26" cy="20" r="11" fill="#0b1120" stroke="#ff9a4d" stroke-width="3"/><circle cx="34" cy="20" r="11" fill="#0b1120" stroke="#ff9a4d" stroke-width="3"/><path d="M52 2 q20 4 30 -3" stroke="#ff9a4d" stroke-width="6" fill="none" stroke-linecap="round" opacity=".55"/></g></svg></div><div class="botinfo"><p>A versioned bot rather than a raw step count. Diego takes third in stochastic at 56.3% (GD +153), where varied, exploratory play suits it, but slips to fourth in deterministic (44.8%). Its sampled game is stronger than its single best line.</p></div></div></div></div>` },
    { h:"Marco — the all-rounder", body:`<div class="botcard"><div class="botpic" data-ph="Add bots/marco.jpg"><img src="bots/marco.jpg" alt="Marco" onerror="this.parentNode.classList.add('empty');this.style.display='none'"></div><div class="bottext"><div class="botslide"><div class="botpic"><img src="marco.png" alt="Marco" onerror="this.remove()"><svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"><rect width="200" height="200" rx="14" fill="#0b0f1c"/><circle cx="100" cy="86" r="82" fill="#ff6a2b" opacity=".14"/><g transform="translate(100 112)"><path d="M-46 6 L-30 -22 L34 -26 L52 -2 L52 14 L-40 18 Z" fill="#ff6a2b"/><path d="M-18 -20 L28 -23 L42 -4 L-22 -1 Z" fill="#ffffff" opacity=".85"/><path d="M-46 6 L52 14 L52 18 L-42 22 Z" fill="#0b1120"/><circle cx="-26" cy="20" r="11" fill="#0b1120" stroke="#ff6a2b" stroke-width="3"/><circle cx="34" cy="20" r="11" fill="#0b1120" stroke="#ff6a2b" stroke-width="3"/><path d="M52 2 q20 4 30 -3" stroke="#ff6a2b" stroke-width="6" fill="none" stroke-linecap="round" opacity=".55"/></g></svg></div><div class="botinfo"><p>The all-rounder on <b>2.0B steps</b>. Fourth in stochastic (35.0%) but climbs to third in deterministic (45.1%), and takes best dribbler in deterministic mode. Marco does better when it commits to its best action instead of sampling.</p></div></div></div></div>` },
    { h:"Marian — the rookie", body:`<div class="botcard"><div class="botpic" data-ph="Add bots/marian.jpg"><img src="bots/marian.jpg" alt="Marian" onerror="this.parentNode.classList.add('empty');this.style.display='none'"></div><div class="bottext"><div class="botslide"><div class="botpic"><img src="marian.png" alt="Marian" onerror="this.remove()"><svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"><rect width="200" height="200" rx="14" fill="#0b0f1c"/><circle cx="100" cy="86" r="82" fill="#93a0c2" opacity=".14"/><g transform="translate(100 112)"><path d="M-46 6 L-30 -22 L34 -26 L52 -2 L52 14 L-40 18 Z" fill="#93a0c2"/><path d="M-18 -20 L28 -23 L42 -4 L-22 -1 Z" fill="#ffffff" opacity=".85"/><path d="M-46 6 L52 14 L52 18 L-42 22 Z" fill="#0b1120"/><circle cx="-26" cy="20" r="11" fill="#0b1120" stroke="#93a0c2" stroke-width="3"/><circle cx="34" cy="20" r="11" fill="#0b1120" stroke="#93a0c2" stroke-width="3"/><path d="M52 2 q20 4 30 -3" stroke="#93a0c2" stroke-width="6" fill="none" stroke-linecap="round" opacity=".55"/></g></svg></div><div class="botinfo"><p>The rookie of the group, with the fewest training steps at <b>1.35B</b>. Fifth in both modes (22.8% and 8.2%). The gap to the top is the clearest sign of how much difference billions of extra steps make.</p></div></div></div></div>` },
    { h:"Fun facts", body:`<p>🛩️ Most aerial: <b>Nachi</b> (~59%)<br>🎮 Most possession: <b>Martin</b> (54-59%)<br>💨 Fastest: <b>Nachi</b><br>⚽ Best dribbler: <b>Martin</b> (stochastic) / <b>Marco</b> (deterministic)<br>🔋 Lowest boost: <b>Martin</b></p>` }
  ]},

  tech: { k:"05 — Tech & MLOps", slides:[
    { h:"The stack", body:`<p>RLGym-PPO and rlgym_sim on top of RocketSim's headless physics, in Python, with Weights and Biases for experiment tracking and RLBot for deployment. Thousands of simulated steps run per second, far faster than real time.</p><p>The codebase is split into independent modules: <code>env</code>, <code>obs</code>, <code>actions</code>, <code>rewards</code>, <code>state_setters</code>, <code>terminal</code>, <code>models</code>, <code>training</code>, <code>evaluation</code> and <code>deployment</code>.</p><a class="gh" href="marian.html" target="_blank" rel="noopener" style="margin-top:14px">&#9654; Present this part as polished slides</a>` },
    { h:"Built like production ML", body:`
      <div class="rung"><span class="c">Reproducible</span><b>One YAML</b><span class="d">Pinned dependencies, one config file per run, deterministic seeds.</span></div>
      <div class="rung"><span class="c">Modular</span><b>Swappable</b><span class="d">Reward, observation, action and state-setter parts are independent.</span></div>
      <div class="rung"><span class="c">Versioned</span><b>Traceable</b><span class="d">Each checkpoint saved with its config, git commit and W&amp;B run id.</span></div>
      <div class="rung"><span class="c">Tested</span><b>pytest + CI</b><span class="d">GitHub Actions runs ruff, mypy and pytest on every push.</span></div>` }
  ]},

  run: { k:"06 — How to run", slides:[
    { h:"Install, train, watch, evaluate", body:`<pre><code># install (with dev extras)
pip install -e ".[dev]"

# train the baseline
python -m rlbot.training.train --config configs/experiments/exp_001_baseline.yaml

# watch a checkpoint play
python scripts/visualize.py --checkpoint checkpoints/exp_001_baseline/latest

# evaluate one bot vs another over 100 games
python scripts/evaluate.py --blue checkpoints/exp_001_baseline/latest --orange checkpoints/exp_000_random/latest --episodes 100</code></pre>` },
    { h:"Makefile shortcuts", body:`<p>The same steps, shorter:</p><pre><code>make install
make train EXP=exp_001_baseline
make eval
make test
make lint</code></pre><p>Collision meshes for rlgym_sim are dumped once before the first run; see <code>docs/setup.md</code>.</p>` }
  ]},

  team: { k:"07 — Team", slides:[
    { layout:"closing", h:"Who made this", body:`
      <p><b>Team.</b> Martin, Nachi, Diego, Marco and Marian.</p>
      <p>Reinforcement Learning, final project, IE School of Science and Technology, 2026.</p>
      <p><b>Thanks</b> to our instructors and teaching team, and to the open-source projects this builds on: RLGym, RLGym-PPO, rlgym_sim and RocketSim.</p>
      <a class="gh" href="https://github.com/moanv2/rlgym" target="_blank" rel="noopener">View the repository &#9658;</a> <a class="gh" href="martin.html" target="_blank" rel="noopener">&#9654; Present this part as polished slides</a>` }
  ]},

  repo: { k:"08 — The code", slides:[
    { layout:"closing", h:"Read the code", body:`
      <p>The simulator setup, the PPO training loop, the reward functions and the evaluation all live in the repository.</p>
      <a class="gh" href="https://github.com/moanv2/rlgym" target="_blank" rel="noopener">Open on GitHub &#9658;</a>` }
  ]}
};
