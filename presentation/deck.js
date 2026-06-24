/* =====================================================================
   DECK FX — premium "designed-website" motion for the slide deck.
   Decoupled by design: it watches #panelBody for each slide render and
   layers on animated leaderboard bars, count-up numbers, a self-drawing
   bar chart, a cursor spotlight and a slow rotating accent frame.
   Loaded AFTER app.js; it never calls into app.js — it just enhances the
   DOM that renderSlide() produces, so it can't break the core deck.
   ===================================================================== */
(function(){
  "use strict";
  var A = window.anime || null;
  var reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---- the small bit of CSS these effects need (additive, new classes) ---- */
  var css = [
  ".panel-box{--mx:50%;--my:50%;}",
  ".deck-spot{position:absolute;inset:0;border-radius:inherit;pointer-events:none;z-index:4;",
  "  opacity:0;transition:opacity .45s ease;mix-blend-mode:screen;",
  "  background:radial-gradient(300px circle at var(--mx) var(--my),rgba(var(--acc-soft,61,123,255),.16),transparent 70%);}",
  ".panel-box:hover .deck-spot{opacity:1;}",
  "@property --ang{syntax:'<angle>';inherits:false;initial-value:0deg;}",
  ".deck-frame{position:absolute;inset:0;border-radius:inherit;z-index:4;pointer-events:none;padding:1.5px;",
  "  background:conic-gradient(from var(--ang,0deg),var(--acc,#3d7bff),transparent 24%,var(--acc,#3d7bff) 50%,transparent 74%,var(--acc,#3d7bff));",
  "  -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);-webkit-mask-composite:xor;",
  "  mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);mask-composite:exclude;",
  "  opacity:.5;animation:deckspin 7s linear infinite;}",
  ".deck-frame.halo{padding:3px;filter:blur(12px);opacity:.32;}",
  "@keyframes deckspin{to{--ang:360deg;}}",
  ".rung b{font-variant-numeric:tabular-nums;}",
  ".rung.lead-glow{box-shadow:0 0 0 1px var(--acc,#3d7bff),0 0 36px -6px var(--acc,#3d7bff);}",
  ".rung{transition:box-shadow .5s ease;}",
  ".chip{display:inline-block;padding:6px 12px;margin:4px 6px 0 0;border:1px solid rgba(var(--acc-soft,61,123,255),.45);border-radius:999px;font-family:var(--mono);font-size:12px;letter-spacing:.04em;color:var(--acc,#3d7bff);}",
  ".kicker2{font-family:var(--mono);font-size:12px;letter-spacing:.22em;text-transform:uppercase;color:var(--acc,#3d7bff);margin-bottom:8px;opacity:.92;}",
  "/* ===== PREMIUM v2 — bolder, glassier, more alive ===== */",
  ".panel-box{box-shadow:0 60px 160px -60px rgba(0,0,0,.92), inset 0 1px 0 rgba(255,255,255,.07);}",
  ".deck-aurora{position:absolute;inset:0;z-index:0;border-radius:inherit;overflow:hidden;pointer-events:none;}",
  ".deck-aurora::before,.deck-aurora::after{content:'';position:absolute;width:62%;height:72%;border-radius:50%;filter:blur(64px);}",
  ".deck-aurora::before{background:radial-gradient(circle,rgba(var(--acc-soft,61,123,255),.45),transparent 70%);top:-22%;right:-12%;animation:auroraA 15s ease-in-out infinite;}",
  ".deck-aurora::after{background:radial-gradient(circle,rgba(var(--acc-soft,61,123,255),.28),transparent 70%);bottom:-26%;left:-12%;animation:auroraB 19s ease-in-out infinite;}",
  "@keyframes auroraA{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(-7%,9%) scale(1.18)}}",
  "@keyframes auroraB{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(9%,-7%) scale(1.22)}}",
  ".deck-grain{position:absolute;inset:0;z-index:5;pointer-events:none;opacity:.045;mix-blend-mode:overlay;background-image:url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='150' height='150'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\");}",
  ".panel-box .tick{position:absolute;width:13px;height:13px;z-index:5;pointer-events:none;border:1.5px solid rgba(var(--acc-soft,61,123,255),.5);}",
  ".panel-box .tick.tl{top:11px;left:11px;border-right:0;border-bottom:0;}",
  ".panel-box .tick.tr{top:11px;right:11px;border-left:0;border-bottom:0;}",
  ".panel-box .tick.bl{bottom:11px;left:11px;border-right:0;border-top:0;}",
  ".panel-box .tick.br{bottom:11px;right:11px;border-left:0;border-top:0;}",
  ".slide-title{letter-spacing:-.015em;text-shadow:0 2px 34px rgba(var(--acc-soft,61,123,255),.20);}",
  "#panelBody:not([data-layout='cover']):not([data-layout='closing']) .slide-title{font-size:clamp(27px,4.7vw,58px);}",
  "#panelBody:not([data-layout='leaderboard']) .slide-content .rung{display:grid;grid-template-columns:clamp(118px,22%,168px) 1fr;gap:5px 18px;align-items:center;padding:13px 18px;margin:9px 0;border:1px solid rgba(255,255,255,.08);border-radius:14px;background:linear-gradient(180deg,rgba(255,255,255,.055),rgba(255,255,255,.015));box-shadow:inset 0 1px 0 rgba(255,255,255,.05);position:relative;overflow:hidden;transition:transform .35s var(--ease),border-color .35s var(--ease),box-shadow .35s var(--ease);}",
  "#panelBody:not([data-layout='leaderboard']) .slide-content .rung::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--acc);box-shadow:0 0 16px rgba(var(--acc-soft,61,123,255),.7);}",
  "#panelBody:not([data-layout='leaderboard']) .slide-content .rung:hover{transform:translateY(-3px);border-color:rgba(var(--acc-soft,61,123,255),.42);box-shadow:0 20px 44px -24px rgba(var(--acc-soft,61,123,255),.65);}",
  "#panelBody:not([data-layout='leaderboard']) .slide-content .rung .c{font-family:var(--mono);font-size:12px;letter-spacing:.04em;color:var(--acc);text-transform:none;}",
  "#panelBody:not([data-layout='leaderboard']) .slide-content .rung b{font-family:var(--display);font-weight:400;font-size:clamp(16px,1.8vw,22px);color:#fff;text-transform:uppercase;letter-spacing:.01em;}",
  "#panelBody:not([data-layout='leaderboard']) .slide-content .rung span.d{grid-column:1 / -1;color:#aeb9d6;font-size:13px;font-family:var(--body);}",
  "#panelBody[data-layout='leaderboard'] .rung{padding:clamp(11px,1.6vw,18px) clamp(15px,1.9vw,24px);border:1px solid rgba(255,255,255,.07);background:linear-gradient(90deg,rgba(255,255,255,.06),rgba(255,255,255,.012));}",
  "#panelBody[data-layout='leaderboard'] .rung::before{background:linear-gradient(90deg,rgba(var(--acc-soft),.34),rgba(var(--acc-soft),.05) 78%);box-shadow:inset 0 0 30px rgba(var(--acc-soft),.3);}",
  "#panelBody[data-layout='leaderboard'] .rung .c{font-size:clamp(13px,1.4vw,16px);}",
  "#panelBody[data-layout='leaderboard'] .rung b{font-size:clamp(23px,3.1vw,40px);font-family:var(--display);text-shadow:0 0 24px rgba(var(--acc-soft),.45);}",
  "#panelBody[data-layout='leaderboard'] .rung:first-child::before{box-shadow:inset 0 0 32px rgba(255,207,77,.42);}",
  ".chip{background:linear-gradient(180deg,rgba(var(--acc-soft,61,123,255),.16),rgba(var(--acc-soft,61,123,255),.04));box-shadow:inset 0 1px 0 rgba(255,255,255,.08),0 0 18px -7px rgba(var(--acc-soft,61,123,255),.7);}",
  "#panelBody .slide-content pre{background:linear-gradient(180deg,#070b16,#05070f);box-shadow:inset 0 1px 0 rgba(255,255,255,.04),0 16px 40px -28px #000;}",
  "#panelBody[data-layout='cover'] .slide-title,#panelBody[data-layout='closing'] .slide-title{background-size:220% 100%;animation:titleShimmer 7s linear infinite;}",
  "@keyframes titleShimmer{to{background-position:220% 0}}",
  "@media (prefers-reduced-motion:reduce){.deck-aurora::before,.deck-aurora::after{animation:none;}#panelBody[data-layout='cover'] .slide-title,#panelBody[data-layout='closing'] .slide-title{animation:none;}}",
  "@media (prefers-reduced-motion:reduce){.deck-frame{animation:none;}}"
  ].join("\n");
  var st = document.createElement('style'); st.textContent = css; document.head.appendChild(st);

  /* ---- one-time decorative layers + cursor spotlight per panel box ---- */
  function ensureLayers(box){
    if(!box || box.__fx) return; box.__fx = true;
    var aurora = document.createElement('div'); aurora.className = 'deck-aurora'; box.insertBefore(aurora, box.firstChild);
    var halo = document.createElement('div'); halo.className = 'deck-frame halo'; box.insertBefore(halo, box.firstChild);
    var ring = document.createElement('div'); ring.className = 'deck-frame';      box.insertBefore(ring, box.firstChild);
    var grain = document.createElement('div'); grain.className = 'deck-grain';    box.appendChild(grain);
    ['tl','tr','bl','br'].forEach(function(p){ var t = document.createElement('span'); t.className = 'tick ' + p; box.appendChild(t); });
    var spot = document.createElement('div'); spot.className = 'deck-spot';        box.appendChild(spot);
    box.addEventListener('pointermove', function(e){
      var r = box.getBoundingClientRect();
      box.style.setProperty('--mx', (e.clientX - r.left) + 'px');
      box.style.setProperty('--my', (e.clientY - r.top)  + 'px');
    });
  }

  /* ---- leaderboard: bars race out from 0 + percentages count up ---- */
  function animateLeaderboard(pb){
    var rungs = pb.querySelectorAll('.rung');
    rungs.forEach(function(r, i){
      if(r.style.getPropertyValue('--w') === '') return;   // not a bar row (e.g. the tech list)
      var target = parseFloat(r.style.getPropertyValue('--w')) || 0;
      var b = r.querySelector('b');
      var endTxt = b ? b.textContent.trim() : '';
      var endNum = parseFloat(endTxt);
      var suffix = endTxt.replace(/[-0-9.]/g, '');           // keeps "%"
      if(!A || reduce){ r.style.setProperty('--w', target); return; }
      r.style.setProperty('--w', '0');
      var o = { w: 0, n: 0 };
      A({ targets:o, w:target, n:isNaN(endNum)?0:endNum, duration:1100, delay:180 + i*120, easing:'easeOutExpo',
          update:function(){
            r.style.setProperty('--w', o.w.toFixed(3));
            if(b && !isNaN(endNum)) b.textContent = o.n.toFixed(1) + suffix;
          },
          complete:function(){ if(i===0){ r.classList.add('lead-glow'); setTimeout(function(){ r.classList.remove('lead-glow'); }, 900); } }
      });
    });
  }

  /* ---- chart: <rect> bars grow + their % labels count up ---- */
  function animateChart(pb){
    if(!A || reduce) return;
    pb.querySelectorAll('svg rect').forEach(function(rect, i){
      var w = rect.getAttribute('width'); if(w == null) return;
      var end = parseFloat(w); if(isNaN(end) || end < 6) return;       // skip thin decorative rects
      rect.setAttribute('width', '0');
      A({ targets:rect, width:[0, end], duration:1100, delay:240 + i*120, easing:'easeOutExpo' });
    });
    pb.querySelectorAll('svg text').forEach(function(t){
      var m = /^([\d.]+)%$/.exec(t.textContent.trim()); if(!m) return;
      var end = parseFloat(m[1]); var o = { n:0 };
      A({ targets:o, n:end, duration:1100, delay:260, easing:'easeOutExpo',
          update:function(){ t.textContent = o.n.toFixed(1) + '%'; } });
    });
  }

  /* ---- cover / closing: split the title into words for a kinetic drop ---- */
  function kineticTitle(pb){
    if(!A || reduce) return;
    var layout = pb.dataset.layout || '';
    if(layout !== 'cover' && layout !== 'closing') return;
    var t = pb.querySelector('.slide-title'); if(!t || t.__split) return;
    t.__split = true;
    var words = t.textContent.trim().split(/\s+/);
    t.innerHTML = words.map(function(w){ return '<span class="tw" style="display:inline-block;will-change:transform">'+w+'</span>'; }).join(' ');
    A({ targets: t.querySelectorAll('.tw'), translateY:[44,0], rotateZ:[5,0], opacity:[0,1],
        delay: A.stagger(70, {start:90}), duration:780, easing:'easeOutExpo' });
  }

  /* ---- run all enhancements on each render ---- */
  function enhance(){
    var pb = document.getElementById('panelBody'); if(!pb) return;
    ensureLayers(pb.closest('.panel-box'));
    var layout = pb.dataset.layout || '';
    if(layout === 'leaderboard' || pb.querySelector('.rung')) animateLeaderboard(pb);
    if(pb.querySelector('svg rect')) animateChart(pb);
    kineticTitle(pb);
  }

  function init(){
    var pb = document.getElementById('panelBody'); if(!pb) return;
    var mo = new MutationObserver(function(){ enhance(); });
    mo.observe(pb, { childList:true });
    enhance(); // in case a slide is already rendered
  }
  if(document.readyState !== 'loading') init(); else addEventListener('DOMContentLoaded', init);
})();
