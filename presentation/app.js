const A = window.anime || null;
function pop(el){ if(!A) return; el.style.display='inline-block'; A({targets:el, scale:[1.7,1], duration:550, easing:'easeOutBack'}); }
function bump(sel,s){ if(A) A({targets:sel, scale:[1, s||1.12, 1], duration:430, easing:'easeOutQuad'}); }
let flashEl=null;
function goalFlash(side){ if(!A) return; if(!flashEl){ flashEl=document.createElement('div'); flashEl.style.cssText='position:fixed;inset:0;z-index:12;pointer-events:none;opacity:0'; document.body.appendChild(flashEl); } flashEl.style.background='radial-gradient(circle at 50% 45%,'+(side==='B'?'rgba(61,123,255,.5)':'rgba(255,106,43,.5)')+',transparent 70%)'; A({targets:flashEl, opacity:[0,.9,0], duration:720, easing:'easeOutQuad'}); }

/* INFO, LABELS, SIGNS are defined in content.js (loaded before this file) */

/* ===================== three.js scene ===================== */
const canvas = document.getElementById('scene');
let renderer;
try{ renderer = new THREE.WebGLRenderer({canvas, antialias:true}); }
catch(err){ document.getElementById('fallback').style.display='flex'; document.getElementById('gate').classList.add('gone'); }

const HALF_X = 145, HALF_Z = 200, WALL_H = 40, GOAL_W = 80, GOAL_H = 30;
let scene, camera, car, ball, ballShadow, carShadow, boostMeshes=[], signObjs=[];
const labelEls = {};

if(renderer){
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio||1));
  renderer.setSize(innerWidth, innerHeight);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x05060d);
  scene.fog = new THREE.Fog(0x05060d, 90, 320);

  camera = new THREE.PerspectiveCamera(62, innerWidth/innerHeight, 0.1, 1200);
  camera.position.set(0, 40, 90);

  scene.add(new THREE.HemisphereLight(0x6688ff, 0x0a0a16, 0.6));
  const key = new THREE.DirectionalLight(0xffffff, 1.1); key.position.set(40,80,30); scene.add(key);
  const pB = new THREE.PointLight(0x3d7bff, 2.6, 320); pB.position.set(-40,30,-50); scene.add(pB);
  const pO = new THREE.PointLight(0xff6a2b, 2.6, 320); pO.position.set(40,30,50); scene.add(pO);

  const floor = new THREE.Mesh(new THREE.PlaneGeometry(HALF_X*2, HALF_Z*2),
    new THREE.MeshStandardMaterial({color:0x0a0e1c, metalness:.3, roughness:.7}));
  floor.rotation.x = -Math.PI/2; scene.add(floor);

  const grid = new THREE.GridHelper(HALF_Z*2, 48, 0x2a4a8f, 0x16223e);
  grid.position.y = 0.02; grid.scale.x = HALF_X/HALF_Z; scene.add(grid);

  const ring = new THREE.Mesh(new THREE.RingGeometry(32,33,64),
    new THREE.MeshBasicMaterial({color:0x3d7bff, side:THREE.DoubleSide, transparent:true, opacity:.7}));
  ring.rotation.x = -Math.PI/2; ring.position.y=0.05; scene.add(ring);
  const midLine = new THREE.Mesh(new THREE.PlaneGeometry(HALF_X*2,0.5),
    new THREE.MeshBasicMaterial({color:0x3d7bff, transparent:true, opacity:.4}));
  midLine.rotation.x=-Math.PI/2; midLine.position.y=0.04; scene.add(midLine);

  function wall(w,h,d,x,y,z,c){
    const m = new THREE.Mesh(new THREE.BoxGeometry(w,h,d),
      new THREE.MeshStandardMaterial({color:c, transparent:true, opacity:.18, metalness:.4, roughness:.4, emissive:c, emissiveIntensity:.15}));
    m.position.set(x,y,z); scene.add(m);
    const edge = new THREE.Mesh(new THREE.BoxGeometry(w,0.4,d), new THREE.MeshBasicMaterial({color:c}));
    edge.position.set(x,h,z); scene.add(edge); return m;
  }
  wall(0.6, WALL_H, HALF_Z*2, -HALF_X, WALL_H/2, 0, 0x3d7bff);
  wall(0.6, WALL_H, HALF_Z*2,  HALF_X, WALL_H/2, 0, 0xff6a2b);
  const sideW = (HALF_X*2 - GOAL_W)/2;
  [ -HALF_Z, HALF_Z ].forEach((z,i)=>{
    const c = i===0?0x3d7bff:0xff6a2b;
    wall(sideW, WALL_H, 0.6, -(GOAL_W/2+sideW/2), WALL_H/2, z, c);
    wall(sideW, WALL_H, 0.6,  (GOAL_W/2+sideW/2), WALL_H/2, z, c);
    const cb = new THREE.Mesh(new THREE.BoxGeometry(GOAL_W,0.6,0.6), new THREE.MeshBasicMaterial({color:c}));
    cb.position.set(0,GOAL_H,z); scene.add(cb);
    const back = new THREE.Mesh(new THREE.PlaneGeometry(GOAL_W,GOAL_H),
      new THREE.MeshBasicMaterial({color:c, transparent:true, opacity:.16, side:THREE.DoubleSide}));
    back.position.set(0,GOAL_H/2, z + (i===0?-1.2:1.2)); scene.add(back);
    const gp = new THREE.PointLight(c, 1.6, 110); gp.position.set(0,GOAL_H/2,z); scene.add(gp);
  });

  const padPos = [[-70,-70],[70,-70],[-70,70],[70,70],[0,-155],[0,155]];
  padPos.forEach(p=>{
    const pad = new THREE.Mesh(new THREE.CylinderGeometry(2.4,2.4,0.3,24),
      new THREE.MeshStandardMaterial({color:0xff6a2b, emissive:0xff6a2b, emissiveIntensity:.8, transparent:true, opacity:.85}));
    pad.position.set(p[0],0.2,p[1]); pad.userData={active:true, t:0, x:p[0], z:p[1]}; scene.add(pad); boostMeshes.push(pad);
  });

  SIGNS.forEach(s=>{
    const g = new THREE.Group();
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.4,0.4,10,12),
      new THREE.MeshStandardMaterial({color:s.color, emissive:s.color, emissiveIntensity:.6}));
    pole.position.y=5; g.add(pole);
    const orb = new THREE.Mesh(new THREE.IcosahedronGeometry(1.6,0),
      new THREE.MeshStandardMaterial({color:s.color, emissive:s.color, emissiveIntensity:.9, metalness:.3, roughness:.3}));
    orb.position.y=11; g.add(orb);
    const ringp = new THREE.Mesh(new THREE.TorusGeometry(3.2,0.12,8,40), new THREE.MeshBasicMaterial({color:s.color}));
    ringp.rotation.x=Math.PI/2; ringp.position.y=0.1; g.add(ringp);
    g.position.set(s.pos[0],0,s.pos[1]); g.userData={key:s.key, orb}; scene.add(g); signObjs.push(g);
    const el=document.createElement('div'); el.className='label'; el.textContent=(s.who?s.who+' · ':'')+LABELS[s.key]; document.body.appendChild(el); labelEls[s.key]=el;
  });

  car = new THREE.Group();
  const carBody = new THREE.Mesh(new THREE.BoxGeometry(4.4,1.6,7.6),
    new THREE.MeshStandardMaterial({color:0x2a5ad8, metalness:.5, roughness:.35, emissive:0x12203f, emissiveIntensity:.3}));
  carBody.position.y=1.5; car.add(carBody);
  const cab = new THREE.Mesh(new THREE.BoxGeometry(3.4,1.3,3.6),
    new THREE.MeshStandardMaterial({color:0x9cc0ff, metalness:.6, roughness:.2}));
  cab.position.set(0,2.7,-0.4); car.add(cab);
  const wheelGeo = new THREE.CylinderGeometry(1.1,1.1,0.9,16);
  const wheelMat = new THREE.MeshStandardMaterial({color:0x0b1020, roughness:.7});
  [[-2.2,2.6],[2.2,2.6],[-2.2,-2.6],[2.2,-2.6]].forEach(([x,z])=>{
    const wMesh = new THREE.Mesh(wheelGeo, wheelMat); wMesh.rotation.z=Math.PI/2; wMesh.position.set(x,1.0,z); car.add(wMesh);
  });
  const flame = new THREE.Mesh(new THREE.ConeGeometry(0.9,3.2,12), new THREE.MeshBasicMaterial({color:0xff8a3d, transparent:true, opacity:0}));
  flame.rotation.x = -Math.PI/2; flame.position.set(0,1.4,4.6); car.add(flame); car.userData.flame=flame;
  car.position.set(0,0,65); scene.add(car);

  const shadowMat = new THREE.MeshBasicMaterial({color:0x000000, transparent:true, opacity:.34});
  carShadow = new THREE.Mesh(new THREE.CircleGeometry(4.4,24), shadowMat.clone()); carShadow.rotation.x=-Math.PI/2; carShadow.position.y=0.06; scene.add(carShadow);
  ballShadow = new THREE.Mesh(new THREE.CircleGeometry(3,24), shadowMat.clone()); ballShadow.rotation.x=-Math.PI/2; ballShadow.position.y=0.06; scene.add(ballShadow);

  ball = new THREE.Mesh(new THREE.SphereGeometry(3,48,32),
    new THREE.MeshStandardMaterial({color:0xf2f5ff, metalness:.35, roughness:.22, emissive:0x2a3a66, emissiveIntensity:.3}));
  ball.position.set(0,3,0); scene.add(ball);
}

/* ===================== state & input ===================== */
const BALL_R=3, CAR_R=4.2, GRAV=70, MAXS=115, BOOSTS=180;
let carAngle=Math.PI, carSpeed=0, camShake=0, carVy=0;
const ballV = new THREE.Vector3(0,0,0);
let boost=100, scoreB=0, scoreO=0, lastSign=null, started=false;
const keys={};
const setKey=(k,v)=>{ keys[k]=v; };
addEventListener('keydown',e=>{ const k=e.key.toLowerCase();
  if(['arrowup','arrowdown','arrowleft','arrowright',' '].includes(k)) e.preventDefault();
  if(panel.classList.contains('open')){ if(presenting){ if(k==='arrowright'||k==='d'||k===' '){e.preventDefault();nextStep();} else if(k==='arrowleft'||k==='a'){e.preventDefault();prevStep();} else if(k==='escape') closePanel(); } else { if(k==='arrowright'||k==='d'){e.preventDefault();slideTo(1);} else if(k==='arrowleft'||k==='a'){e.preventDefault();slideTo(-1);} else if(k==='escape') closePanel(); } return; }
  if(k==='w'||k==='arrowup')setKey('up',1); if(k==='s'||k==='arrowdown')setKey('down',1);
  if(k==='a'||k==='arrowleft')setKey('left',1); if(k==='d'||k==='arrowright')setKey('right',1);
  if(k==='shift')setKey('boost',1); if(k===' '&&car&&car.position.y<=0.6){ carVy=36; } if(k==='r') resetBall(); if(k==='e'&&lastSign) openPanel(lastSign);
});
addEventListener('keyup',e=>{ const k=e.key.toLowerCase();
  if(k==='w'||k==='arrowup')setKey('up',0); if(k==='s'||k==='arrowdown')setKey('down',0);
  if(k==='a'||k==='arrowleft')setKey('left',0); if(k==='d'||k==='arrowright')setKey('right',0);
  if(k==='shift')setKey('boost',0);
});
document.querySelectorAll('#touch b').forEach(b=>{
  const k=b.dataset.k;
  const on=e=>{e.preventDefault();setKey(k,1);b.classList.add('act');};
  const off=e=>{e.preventDefault();setKey(k,0);b.classList.remove('act');};
  b.addEventListener('touchstart',on,{passive:false}); b.addEventListener('touchend',off);
  b.addEventListener('mousedown',on); b.addEventListener('mouseup',off); b.addEventListener('mouseleave',off);
});
function resetBall(){ if(!ball) return; ball.position.set(0,3,0); ballV.set(0,0,0); }

/* ===================== audio (synth music + sfx) ===================== */
const Audio0 = (()=>{
  let ac=null, master=null, comp=null, st=false, muted=false, timer=null, reading=false, musicEl=null;
  const BPM=126, SPB=60/BPM/4; let nextT=0, step=0, bar=0;
  const prog=[{root:45,tri:[57,60,64]},{root:41,tri:[53,57,60]},{root:48,tri:[55,60,64]},{root:43,tri:[55,59,62]}];
  const mtof=m=>440*Math.pow(2,(m-69)/12);
  function kick(t){const o=ac.createOscillator(),g=ac.createGain();o.frequency.setValueAtTime(150,t);o.frequency.exponentialRampToValueAtTime(48,t+.12);g.gain.setValueAtTime(.9,t);g.gain.exponentialRampToValueAtTime(.0001,t+.34);o.connect(g).connect(master);o.start(t);o.stop(t+.36);}
  function hat(t,op){const b=ac.createBuffer(1,ac.sampleRate*0.05,ac.sampleRate),d=b.getChannelData(0);for(let i=0;i<d.length;i++)d[i]=Math.random()*2-1;const s=ac.createBufferSource();s.buffer=b;const hp=ac.createBiquadFilter();hp.type='highpass';hp.frequency.value=7000;const g=ac.createGain();g.gain.setValueAtTime(.0001,t);g.gain.linearRampToValueAtTime(.2,t+.002);g.gain.exponentialRampToValueAtTime(.0001,t+(op?.12:.04));s.connect(hp).connect(g).connect(master);s.start(t);s.stop(t+.16);}
  function bass(t,f){const o=ac.createOscillator(),g=ac.createGain(),lp=ac.createBiquadFilter();o.type='sawtooth';o.frequency.value=f;lp.type='lowpass';lp.frequency.value=440;lp.Q.value=6;g.gain.setValueAtTime(.0001,t);g.gain.linearRampToValueAtTime(.32,t+.01);g.gain.exponentialRampToValueAtTime(.0001,t+SPB*1.9);o.connect(lp).connect(g).connect(master);o.start(t);o.stop(t+SPB*2);}
  function arp(t,f){const o=ac.createOscillator(),g=ac.createGain();o.type='square';o.frequency.value=f;g.gain.setValueAtTime(.0001,t);g.gain.linearRampToValueAtTime(.09,t+.005);g.gain.exponentialRampToValueAtTime(.0001,t+.18);o.connect(g).connect(master);o.start(t);o.stop(t+.2);}
  function pad(t,fs,dur){fs.forEach((f,i)=>{const o=ac.createOscillator(),g=ac.createGain();o.type='triangle';o.frequency.value=f;o.detune.value=(i-1)*6;g.gain.setValueAtTime(.0001,t);g.gain.linearRampToValueAtTime(.045,t+.4);g.gain.setValueAtTime(.045,t+dur-.5);g.gain.exponentialRampToValueAtTime(.0001,t+dur);o.connect(g).connect(master);o.start(t);o.stop(t+dur+.05);});}
  function stepFn(s,b,t){const ch=prog[b];if(s%4===0)kick(t);hat(t,s%4===2);if(s%2===0)bass(t,mtof(ch.root));arp(t,mtof(ch.tri[s%3]+12));if(s===0)pad(t,ch.tri.map(m=>mtof(m)),SPB*16);}
  function loop(){while(nextT<ac.currentTime+0.12){stepFn(step,bar,nextT);nextT+=SPB;step++;if(step>=16){step=0;bar=(bar+1)%prog.length;}}timer=setTimeout(loop,25);}
  function fade(el,to){ const from=el.volume, t0=performance.now(); (function s(t){ const p=Math.min(1,(t-t0)/260); el.volume=Math.max(0,Math.min(1,from+(to-from)*p)); if(p<1) requestAnimationFrame(s); })(t0); }
  function setVol(){ const target=(muted||reading)?0:0.5; if(musicEl){ fade(musicEl,target); } else if(ac&&master){ master.gain.cancelScheduledValues(ac.currentTime); master.gain.exponentialRampToValueAtTime((muted||reading)?.0001:.5, ac.currentTime+.25); } }
  function start(mute){ if(st)return; muted=!!mute; st=true;
    if(typeof MUSIC_FILE!=='undefined' && MUSIC_FILE){ musicEl=new Audio(MUSIC_FILE); musicEl.preload='auto'; musicEl.volume=0; musicEl.loop=false;
      var _clip=(typeof MUSIC_CLIP!=='undefined'&&MUSIC_CLIP)?MUSIC_CLIP:null, _cs=(_clip&&_clip.start)||0, _ce=(_clip&&_clip.end)||null;
      musicEl.addEventListener('loadedmetadata',function(){ try{ if(_cs) musicEl.currentTime=_cs; }catch(e){} });
      function _wrap(){ var e=_ce||(musicEl.duration||1e9); if(musicEl.currentTime>=e-0.05||musicEl.ended){ try{ musicEl.currentTime=_cs; if(musicEl.paused) musicEl.play(); }catch(err){} } }
      musicEl.addEventListener('timeupdate',_wrap); musicEl.addEventListener('ended',_wrap);
      musicEl.play().then(()=>{ try{ if(musicEl.currentTime<_cs) musicEl.currentTime=_cs; }catch(e){} setVol(); }).catch(function(){ musicEl=null; startSynth0(); }); return; }
    function startSynth0(){
    try{ac=new (window.AudioContext||window.webkitAudioContext)();}catch(e){return;}
    master=ac.createGain();master.gain.value=.0001;comp=ac.createDynamicsCompressor();master.connect(comp);comp.connect(ac.destination);
    nextT=ac.currentTime+.08;loop();master.gain.exponentialRampToValueAtTime(muted?.0001:.5,ac.currentTime+1.2);if(ac.state==='suspended')ac.resume(); }
    startSynth0(); }
  function toggle(){ if(!st)return false; muted=!muted; setVol(); return !muted; }
  function blip(freq,dur,type,vol){ if(!st||muted||!ac)return; const o=ac.createOscillator(),g=ac.createGain(); o.type=type||'sine'; o.frequency.setValueAtTime(freq,ac.currentTime); g.gain.setValueAtTime(.0001,ac.currentTime); g.gain.linearRampToValueAtTime(vol||.25,ac.currentTime+.005); g.gain.exponentialRampToValueAtTime(.0001,ac.currentTime+(dur||.18)); o.connect(g).connect(master); o.start(); o.stop(ac.currentTime+(dur||.18)+.02); }
  function hit(strength){ blip(180+strength*220, .16, 'triangle', .3); }
  function goal(){ [0,4,7,12].forEach((n,i)=>setTimeout(()=>blip(440*Math.pow(2,n/12),.22,'square',.22), i*80)); }
  function read(on){ if(!st)return; reading=on; setVol(); }
  return { start, toggle, read, hit, goal, isOn:()=>st&&!muted, isStarted:()=>st };
})();

/* ===================== HUD / panel ===================== */
const soundBtn=document.getElementById('sound');
soundBtn.addEventListener('click',()=>{ if(!Audio0.isStarted()){Audio0.start(false);setSound(true);return;} setSound(Audio0.toggle()); });
function setSound(on){ soundBtn.classList.toggle('playing',on); soundBtn.classList.toggle('off',!on); bump('#sound',1.2); }

const panel=document.getElementById('panel'), panelTabs=document.getElementById('panelTabs'), panelBody=document.getElementById('panelBody');
Object.keys(INFO).forEach(k=>{ const b=document.createElement('button'); b.textContent=LABELS[k]; b.dataset.k=k; b.onclick=()=>showSection(k); panelTabs.appendChild(b); });
let curSection=null, curSlide=0, stepIdx=0, presenting=false;
const STEPS=[]; Object.keys(INFO).forEach(k=>INFO[k].slides.forEach((_,i)=>STEPS.push({section:k,slide:i})));
function goToStep(i){ stepIdx=Math.max(0,Math.min(STEPS.length-1,i)); const s=STEPS[stepIdx]; curSection=s.section; curSlide=s.slide; [...panelTabs.children].forEach(b=>b.classList.toggle('on',b.dataset.k===s.section)); renderSlide(); }
function nextStep(){ if(stepIdx<STEPS.length-1) goToStep(stepIdx+1); }
function prevStep(){ if(stepIdx>0) goToStep(stepIdx-1); }
function showSection(k){ curSection=k; curSlide=0; stepIdx=STEPS.findIndex(s=>s.section===k); [...panelTabs.children].forEach(b=>b.classList.toggle('on',b.dataset.k===k)); renderSlide(); }
function slideTo(d){ const n=INFO[curSection].slides.length; curSlide=Math.max(0,Math.min(n-1,curSlide+d)); stepIdx=STEPS.findIndex(s=>s.section===curSection&&s.slide===curSlide); renderSlide(); }
function renderSlide(){ const sec=INFO[curSection], sl=sec.slides[curSlide], n=sec.slides.length, total=STEPS.length;
  const pageNow=presenting?stepIdx+1:curSlide+1, pageTot=presenting?total:n, prog=(pageNow/pageTot)*100;
  let nav;
  if(presenting) nav=`<button class="sb" data-step="-1"${stepIdx===0?' disabled':''}>&lsaquo;</button><span class="navlabel">${LABELS[curSection]}</span><button class="sb" data-step="1"${stepIdx===total-1?' disabled':''}>&rsaquo;</button>`;
  else if(n>1) nav=`<button class="sb" data-d="-1"${curSlide===0?' disabled':''}>&lsaquo;</button><span class="navlabel">${LABELS[curSection]}</span><button class="sb" data-d="1"${curSlide===n-1?' disabled':''}>&rsaquo;</button>`;
  else nav=`<span class="navlabel">${LABELS[curSection]}</span>`;
  panelBody.innerHTML=`<div class="slide-head"><span class="kick">${sec.k}</span><span class="pg">${pageNow} / ${pageTot}</span></div><h2 class="slide-title">${sl.h}</h2><div class="slide-content">${sl.body}</div><div class="slide-foot"><span class="brand2"><span class="d"></span>RLGYM</span><span class="prog"><i style="width:${prog}%"></i></span><span class="navc">${nav}</span></div>`;
  panelBody.dataset.layout = sl.layout || (sl.body.indexOf('botcard')>-1?'split':'content');
  const _ac=(typeof ACCENTS!=='undefined'&&ACCENTS[curSection])||{css:'#3d7bff',rgb:'61, 123, 255',n:'01'};
  const _pb=panelBody.closest('.panel-box'); if(_pb){ _pb.style.setProperty('--acc',_ac.css); _pb.style.setProperty('--acc-soft',_ac.rgb); _pb.style.setProperty('--acc-n','"'+_ac.n+'"'); }
  panelBody.querySelectorAll('.sb[data-step]').forEach(b=>b.onclick=()=>{ (+b.dataset.step>0?nextStep:prevStep)(); });
  panelBody.querySelectorAll('.sb[data-d]').forEach(b=>b.onclick=()=>slideTo(+b.dataset.d));
  if(A) A({targets:'#panelBody .slide-head, #panelBody .slide-title, #panelBody .slide-content > *, #panelBody .slide-foot',translateY:[12,0],opacity:[0,1],delay:A.stagger(40),duration:440,easing:'easeOutExpo'}); }
function openPanel(k){ presenting=false; for(const kk in keys) keys[kk]=0; showSection(k||'overview'); panel.classList.add('open'); Audio0.read(true); if(A) A({targets:'.panel-box',scale:[.96,1],duration:400,easing:'easeOutExpo'}); }
function closePanel(){ panel.classList.remove('open'); Audio0.read(false); }
var __mb=document.getElementById('menuBtn'); if(__mb) __mb.onclick=()=>enterPresent(false);
document.getElementById('panelX').onclick=()=>closePanel();
panel.addEventListener('click',e=>{ if(e.target===panel) closePanel(); });

/* ===================== gate ===================== */
const gate=document.getElementById('gate'), ld=document.getElementById('ld');
if(A){ A.timeline({easing:'easeOutExpo'}).add({targets:'#gate .mk',translateY:[60,0],opacity:[0,1],scale:[.8,1],duration:850}).add({targets:'#gate .tag, #gate .ld',opacity:[0,1],translateY:[12,0],duration:500},'-=450'); }
let dotsN=0; const ldT=setInterval(()=>{dotsN=(dotsN+1)%4; ld.textContent='loading'+'.'.repeat(dotsN);},340);
function ready(){ clearInterval(ldT); ld.textContent='ready'; var _d=document.getElementById('driveBtn'); if(_d)_d.classList.add('ready'); var _m=document.getElementById('driveMuted'); if(_m)_m.classList.add('ready'); }
if(renderer){ requestAnimationFrame(()=>requestAnimationFrame(ready)); }
function begin(sound){ gate.classList.add('gone'); started=true; if(sound){Audio0.start(false);setSound(true);} if(A){ document.querySelectorAll('.hud').forEach(e=>{e.style.opacity=0;}); A({targets:'.hud',translateY:[-14,0],opacity:[0,1],delay:A.stagger(80),duration:650,easing:'easeOutExpo'}); } }
document.getElementById('driveBtn').onclick=()=>begin(true);
document.getElementById('driveMuted').onclick=()=>begin(false);
function enterPresent(fromStart){ presenting=true; if(!started) begin(true); for(const kk in keys) keys[kk]=0; goToStep(fromStart?0:stepIdx); panel.classList.add('open'); Audio0.read(true); if(A) A({targets:'.panel-box',scale:[.96,1],duration:400,easing:'easeOutExpo'}); }
var __pb=document.getElementById('presentBtn'); if(__pb) __pb.onclick=()=>enterPresent(true);

/* ===================== loop ===================== */
const tmpV=new THREE.Vector3(); const clock={last:performance.now()};
const prompt=document.getElementById('prompt');
const boostFill=document.getElementById('boostFill'), speedVal=document.getElementById('speedVal');
const scoreBEl=document.getElementById('scoreB'), scoreOEl=document.getElementById('scoreO');

function flashGoal(side){ scene.background = new THREE.Color(side==='B'?0x12203f:0x3a1c0c); setTimeout(()=>{scene.background=new THREE.Color(0x05060d);},180); }

function frame(now){
  requestAnimationFrame(frame);
  if(!renderer) return;
  const dt=Math.min(0.05,(now-clock.last)/1000); clock.last=now;

  if(started && !panel.classList.contains('open')){
    const thr=(keys.up?1:0)-(keys.down?1:0);
    const steer=(keys.left?1:0)-(keys.right?1:0);
    const boosting=keys.boost && boost>0;
    const cap=(boosting?BOOSTS:MAXS);
    carSpeed += (thr*38 + (boosting?40:0))*dt;
    carSpeed *= 0.985;
    if(Math.abs(carSpeed)>cap) carSpeed=cap*Math.sign(carSpeed);
    if(!thr && !boosting){ carSpeed*=0.96; if(Math.abs(carSpeed)<0.4)carSpeed=0; }
    var airborne = car.position.y > 0.5;
    // in the air you can rotate freely to point (and thrust) in ANY direction — not speed-gated like on the ground
    carAngle += steer * (airborne ? 2.7 : 1.7) * dt * (airborne ? 1 : Math.max(-1,Math.min(1,carSpeed/12)));
    if(airborne) carSpeed += (thr*22)*dt;   // extra air thrust so W/S actually move you while floating
    car.position.x += Math.sin(carAngle)*carSpeed*dt;
    car.position.z += Math.cos(carAngle)*carSpeed*dt;
    car.position.x=Math.max(-HALF_X+3,Math.min(HALF_X-3,car.position.x));
    car.position.z=Math.max(-HALF_Z+3,Math.min(HALF_Z-3,car.position.z));
    car.rotation.y=carAngle;
    // floaty low-gravity jump + RL-style aerials: SPACE jumps, holding boost in the air lifts you to the ceiling
    if(boosting && car.position.y>0.4) carVy += 52*dt;
    carVy -= 15*dt;                                          // gentle gravity = it floats
    car.position.y += carVy*dt;
    if(car.position.y<=0){ car.position.y=0; if(carVy<0) carVy=0; }
    if(car.position.y>=78){ car.position.y=78; if(carVy>0) carVy=0; }   // soft ceiling
    car.rotation.x = Math.max(-0.5, Math.min(0.5, carVy*0.012));        // nose tilts with vertical motion
    if(boosting){ boost=Math.max(0,boost-32*dt); car.userData.flame.material.opacity=0.6+Math.random()*0.3; }
    else { boost=Math.min(100,boost+10*dt); car.userData.flame.material.opacity=0; }
    boostFill.style.width=boost+'%';
    speedVal.textContent=Math.round(Math.abs(carSpeed)*3.0);

    boostMeshes.forEach(p=>{
      p.rotation.y+=dt*1.5;
      if(p.userData.active){
        if(Math.hypot(car.position.x-p.userData.x, car.position.z-p.userData.z)<4){ boost=100; p.userData.active=false; p.visible=false; Audio0.hit(0.2); bump('#boostWrap',1.16); }
      } else { p.userData.t+=dt; if(p.userData.t>6){ p.userData.active=true; p.visible=true; p.userData.t=0; } }
    });

    ballV.y -= GRAV*dt;
    ball.position.addScaledVector(ballV, dt);
    if(ball.position.y<BALL_R){ ball.position.y=BALL_R; ballV.y*=-0.6; ballV.x*=0.985; ballV.z*=0.985; if(Math.abs(ballV.y)<2)ballV.y=0; }
    if(Math.abs(ball.position.x)>HALF_X-BALL_R){ ball.position.x=Math.sign(ball.position.x)*(HALF_X-BALL_R); ballV.x*=-0.8; Audio0.hit(0.1); }
    if(Math.abs(ball.position.z)>HALF_Z-BALL_R){
      const inMouth = Math.abs(ball.position.x)<GOAL_W/2 && ball.position.y<GOAL_H;
      if(inMouth){
        if(ball.position.z>HALF_Z-BALL_R){ scoreB++; scoreBEl.textContent=scoreB; pop(scoreBEl); goalFlash('B'); flashGoal('B'); Audio0.goal(); resetBall(); }
        else { scoreO++; scoreOEl.textContent=scoreO; pop(scoreOEl); goalFlash('O'); flashGoal('O'); Audio0.goal(); resetBall(); }
      } else { ball.position.z=Math.sign(ball.position.z)*(HALF_Z-BALL_R); ballV.z*=-0.8; Audio0.hit(0.1); }
    }
    tmpV.subVectors(ball.position, car.position); tmpV.y=0;
    const dist=tmpV.length();
    if(dist < CAR_R+BALL_R){
      tmpV.normalize();
      const power = 14 + Math.abs(carSpeed)*0.9;
      ballV.x += tmpV.x*power; ballV.z += tmpV.z*power; ballV.y += 9 + Math.abs(carSpeed)*0.15;
      ball.position.addScaledVector(tmpV, (CAR_R+BALL_R-dist));
      camShake = Math.max(camShake, Math.min(1, Math.abs(carSpeed)/MAXS));
      Audio0.hit(Math.min(1,Math.abs(carSpeed)/MAXS));
    }
    ball.rotation.x+=ballV.z*0.01; ball.rotation.z-=ballV.x*0.01;

    carShadow.position.set(car.position.x,0.06,car.position.z);
    ballShadow.position.set(ball.position.x,0.06,ball.position.z);
    ballShadow.scale.setScalar(Math.max(0.5,1-(ball.position.y-BALL_R)/40));

    let near=null;
    signObjs.forEach(g=>{
      g.userData.orb.rotation.y+=dt*1.2;
      const d=Math.hypot(car.position.x-g.position.x, car.position.z-g.position.z);
      labelEls[g.userData.key].classList.toggle('near', d<14);
      if(d<14){ near=g.userData.key; }
    });
    if(near){ lastSign=near; prompt.classList.add('show'); prompt.innerHTML=`Press <b>E</b> to read · ${LABELS[near]}`; }
    else { lastSign=null; prompt.classList.remove('show'); }
  }

  if(car){
    var GC=window.goalCam;
    if(GC && GC.t>0){ GC.t-=0.016; camera.position.lerp(GC.pos,0.05); camera.lookAt(GC.look.x,GC.look.y,GC.look.z); }
    else { const back=55, up=27 + car.position.y*0.55;
    const desired=tmpV.set(car.position.x - Math.sin(carAngle)*back, up, car.position.z - Math.cos(carAngle)*back);
    camera.position.lerp(desired, started?0.08:0.02);
    if(camShake>0.01){ camera.position.x += (Math.random()-0.5)*camShake*2.4; camera.position.y += (Math.random()-0.5)*camShake*1.7; camShake*=0.86; } else camShake=0;
    camera.lookAt(car.position.x, 3 + car.position.y*0.7, car.position.z); }
    var fovT = 60 + ((keys.boost&&boost>0&&started)?12:0) + Math.min(8, Math.abs(carSpeed)*0.12);
    camera.fov += (fovT - camera.fov)*0.08; camera.updateProjectionMatrix();
  }

  signObjs.forEach(g=>{
    const el=labelEls[g.userData.key];
    tmpV.set(g.position.x, 13, g.position.z).project(camera);
    if(tmpV.z>1){ el.style.opacity=0; return; }
    el.style.opacity=0.95;
    el.style.left=((tmpV.x*0.5+0.5)*innerWidth)+'px';
    el.style.top=((-tmpV.y*0.5+0.5)*innerHeight)+'px';
  });

  if(!window.__deckOpen) renderer.render(scene,camera);   // free the GPU for smooth video while a deck overlay is open
}
if(renderer) requestAnimationFrame(frame);

addEventListener('resize',()=>{ if(!renderer)return; camera.aspect=innerWidth/innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth,innerHeight); });
