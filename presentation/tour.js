/* =====================================================================
   STORY TOUR — a guided, drive-through presentation.
   5 checkpoints, one per speaker. Loaded AFTER app.js; uses its globals
   (THREE, scene, car, signObjs, SIGNS, panel, openPanel, begin, started).
   Edit the 5 stops below to change who presents what.
   ===================================================================== */
(function(){
  "use strict";
  if(typeof THREE==='undefined') return;
  const A = window.anime || null;

  // ---- the 5 stops = the five presentation phases. key = which deck opens here. ----
  const STOPS=[
    { key:'intro',      who:'Marian',        label:'Phase 1 — Intro & Class',              multi:true },
    { key:'method',     who:'Diego & Marco', label:'Phase 2 — Methodology & Architecture', multi:true },
    { key:'bots',       who:'Nachi',         label:'Phase 3 — Our Bots in Detail',         multi:true },
    { key:'champion',   who:'Martin',        label:'Phase 4 — The Author of the Champion', multi:true },
    { key:'conclusion', who:'Marco',         label:'Phase 5 — The Conclusion',             multi:true }
  ];
  const N=STOPS.length, R=11;
  let state='off', cp=0, armed=true, raf=0;

  const signObjsOf = ()=> (typeof signObjs!=='undefined'&&signObjs) ? signObjs : [];
  const sign = key => signObjsOf().find(g=>g.userData&&g.userData.key===key);
  const colorOf = key => { const s=(typeof SIGNS!=='undefined'&&SIGNS||[]).find(x=>x.key===key); return s?s.color:0x7aa6ff; };
  const isOpen = ()=> (typeof panel!=='undefined'&&panel) ? panel.classList.contains('open') : false;

  /* ---------- HUD ---------- */
  let hud, doneCard, chapEl, hintEl;
  function buildHud(){
    if(hud) return;
    hud=document.createElement('div'); hud.id='tourHud';
    hud.innerHTML='<div class="chap"></div><div class="hint"></div><button class="ex">✕ exit tour</button>';
    document.body.appendChild(hud);
    chapEl=hud.querySelector('.chap'); hintEl=hud.querySelector('.hint');
    hud.querySelector('.ex').onclick=stop;
    doneCard=document.createElement('div'); doneCard.id='tourDone';
    doneCard.innerHTML='<p>Story tour complete</p><h2>That is the run</h2><button>Free drive ▸</button>';
    document.body.appendChild(doneCard);
    doneCard.querySelector('button').onclick=()=>{ doneCard.classList.remove('on'); stop(); };
  }
  function updateHud(){
    if(!chapEl) return; const s=STOPS[cp];
    chapEl.textContent='Chapter '+(cp+1)+' / '+N+'  ·  '+s.who.toUpperCase()+'  ·  '+s.label;
    hintEl.textContent = s.multi ? 'At the checkpoint: page the slides, use the tabs for each section'
                                 : 'Drive into the glowing checkpoint';
    if(A) A({targets:'#tourHud .chap, #tourHud .hint', translateY:[8,0], opacity:[0,1], delay:A.stagger(60), duration:420, easing:'easeOutExpo'});
  }

  /* ---------- 3D guidance fx ---------- */
  let ring, beam, path, built=false;
  function buildFx(){
    if(built||typeof scene==='undefined'||!scene) return; built=true;
    try{
      const pts=STOPS.map(s=>{ const g=sign(s.key); return new THREE.Vector3(g.position.x,0.25,g.position.z); });
      const curve=new THREE.CatmullRomCurve3(pts,false,'catmullrom',0.35);
      path=new THREE.Mesh(new THREE.TubeGeometry(curve,90,0.42,6,false),
        new THREE.MeshBasicMaterial({color:0x6f9dff,transparent:true,opacity:.4}));
      path.visible=false; scene.add(path);
      ring=new THREE.Mesh(new THREE.TorusGeometry(6,0.4,12,52), new THREE.MeshBasicMaterial({color:0xffffff}));
      ring.rotation.x=Math.PI/2; ring.position.y=0.3; ring.visible=false; scene.add(ring);
      beam=new THREE.Mesh(new THREE.CylinderGeometry(0.7,1.5,16,18,1,true),
        new THREE.MeshBasicMaterial({color:0xffffff,transparent:true,opacity:.2,side:THREE.DoubleSide}));
      beam.position.y=8; beam.visible=false; scene.add(beam);
    }catch(e){ built=false; }
  }
  function placeFx(){
    buildFx(); const g=sign(STOPS[cp].key); if(!g||!ring) return;
    const c=new THREE.Color(colorOf(STOPS[cp].key));
    ring.position.set(g.position.x,0.3,g.position.z); ring.material.color.copy(c); ring.visible=true;
    beam.position.set(g.position.x,8,g.position.z); beam.material.color.copy(c); beam.visible=true;
    if(path) path.visible=true;
  }
  function hideFx(){ [ring,beam,path].forEach(m=>{ if(m) m.visible=false; }); }
  function tickFx(now){
    if(!ring) return; const t=now*0.001;
    if(ring.visible){ const s=1+Math.sin(t*3)*0.07; ring.scale.set(s,s,s); ring.rotation.z=t*0.5; }
    if(beam.visible){ beam.material.opacity=0.14+Math.abs(Math.sin(t*2))*0.12; beam.rotation.y=t*0.4; }
  }

  /* ---------- loop ---------- */
  function update(now){
    if(state==='off'){ raf=0; return; }
    raf=requestAnimationFrame(update);
    tickFx(now);
    if(typeof car==='undefined'||!car) return;
    const open=isOpen();
    if(state==='guiding' && !open){
      const g=sign(STOPS[cp].key); if(!g) return;
      const d=Math.hypot(car.position.x-g.position.x, car.position.z-g.position.z);
      if(armed && d<R) enterChapter();
    } else if(state==='chapter' && !open){
      advance();
    }
  }
  function enterChapter(){
    armed=false; state='chapter';
    if(hud) hud.classList.remove('on');
    if(typeof openPanel==='function') openPanel(STOPS[cp].key);
  }
  function advance(){
    if(cp>=N-1){ finish(); return; }
    cp++; armed=true; state='guiding';
    placeFx(); updateHud(); if(hud) hud.classList.add('on');
  }
  function finish(){
    state='complete'; window.__tourActive=false; hideFx(); if(hud) hud.classList.remove('on');
    buildHud(); doneCard.classList.add('on');
    if(A) A({targets:'#tourDone h2', scale:[.85,1], opacity:[0,1], duration:700, easing:'easeOutBack'});
  }

  /* ---------- public ---------- */
  function start(){
    if(!signObjsOf().length) return;            // arena did not build (no WebGL)
    buildHud();
    if(typeof started==='undefined' || !started){ if(typeof begin==='function') begin(true); }
    cp=0; armed=true; state='guiding'; window.__tourActive=true;
    placeFx(); updateHud(); hud.classList.add('on');
    if(!raf) raf=requestAnimationFrame(update);
  }
  function stop(){
    state='off'; window.__tourActive=false; if(raf){ cancelAnimationFrame(raf); raf=0; } hideFx();
    if(hud) hud.classList.remove('on'); if(doneCard) doneCard.classList.remove('on');
  }

  /* ---------- entry button on the gate ---------- */
  function injectBtn(){
    const pb=document.getElementById('driveBtn');
    if(!pb || document.getElementById('tourBtn')) return;
    const b=document.createElement('button'); b.id='tourBtn'; b.textContent='◆ Story tour';
    b.onclick=start; pb.insertAdjacentElement('afterend', b);
    setTimeout(()=>b.classList.add('ready'), 1700);
    addEventListener('load', ()=>b.classList.add('ready'));
  }
  if(document.readyState!=='loading') injectBtn(); else addEventListener('DOMContentLoaded', injectBtn);

  window.Tour={ start, stop };
})();
