/* =====================================================================
   EMBED — open each pole's deck in the BROWSER (a new tab) when you
   reach it and press E (or via the story tour). Simpler + more robust
   than an in-app iframe overlay: every deck renders in its own full
   browser tab, so each presenter drives their own deck however they like
   (fullscreen, pointer, etc.). The arena keeps running in its own tab.
   Loaded LAST; wraps window.openPanel so the E key + tour route through it.
   Delete this one <script> line from index.html to roll back.
   ===================================================================== */
(function(){
  "use strict";

  // pole key -> the deck file that opens in a new tab (presentation order)
  var DECKS = {
    intro:      '01_intro_marian.html',                  // Phase 1 · Marian        · Intro & Class
    method:     'rlgym-PPO_loop_and_architecture.html',  // Phase 2 · Diego & Marco · Methodology & Architecture
    bots:       'bots_champion_style.html',              // Phase 3 · Nachi         · Our Bots in Detail
    champion:   'martin_champion_deck.html',             // Phase 4 · Martin        · The Author of the Champion
    conclusion: 'conclusion_marco.html'                  // Phase 5 · Marco         · The Conclusion
  };

  var last = 0;
  function openDeck(url){
    var now = Date.now();
    if (now - last < 600) return;                 // ignore key-repeat / double fire
    last = now;
    try { if (typeof Audio0 !== 'undefined' && Audio0.read) Audio0.read(true); } catch(e){}  // quiet the arena music
    var w = null;
    try { w = window.open(url, '_blank'); } catch(e){}
    // If a popup is blocked (e.g. the AUTO tour, which has no click/keypress),
    // do NOT navigate the game away — just let the presenter press E to open it.
    if (!w) { try { console.warn('[deck] popup blocked — press E at the pole to open ' + url); } catch(e){} }
  }

  /* ---------- route openPanel (E key + story tour) to a new browser tab ---------- */
  function install(){
    var orig = window.openPanel;
    window.openPanel = function(key){
      if (key && DECKS[key]) openDeck(DECKS[key]);
      else if (typeof orig === 'function') orig(key);   // anything non-pole keeps its old behaviour
    };
  }
  if (typeof window.openPanel === 'function') install();
  else addEventListener('load', install);
})();
