/* =====================================================================
   LIVE MATCH — autonomous AI cars play a real match while you spectate.
   4 bots (2 blue, 2 orange) chase the ball, jump, bump and SCORE (the
   scoreboard ticks for real); you are the spectator car weaving between
   them to the checkpoints. Goal cuts the camera + a confetti burst, and a
   match clock runs in the HUD. Loaded AFTER app.js (uses its globals).
   ===================================================================== */
(function(){
  "use strict";
  if(typeof THREE==='undefined') return;
  const TEAM={ blue:0x3d7bff, orange:0xff6a2b };
  const ATTACK={ blue:1, orange:-1 };
  const SPAWN=[
    {t:'blue',  x:-45, z:-135}, {t:'blue',  x:45, z:-135},
    {t:'orange',x:-45, z: 135}, {t:'orange',x:45, z: 135}
  ];
  let bots=[], bursts=[], built=false, raf=0, last=performance.now();
  let prevB=0, prevO=0, clockEl=null, t0=0;

  function makeCar(color){
    const g=new THREE.Group();
    const body=new THREE.Mesh(new THREE.BoxGeometry(4,1.4,7),
      new THREE.MeshStandardMaterial({color, metalness:.6, roughness:.35, emissive:color, emissiveIntensity:.22}));
    body.position.y=1.4; g.add(body);
    const nose=new THREE.Mesh(new THREE.BoxGeometry(3.4,0.9,1.6),
      new THREE.MeshStandardMaterial({color, metalness:.6, roughness:.35}));
    nose.position.set(0,1.1,3.4); g.add(nose);
    const cab=new THREE.Mesh(new THREE.BoxGeometry(2.9,1.0,3),
      new THREE.MeshStandardMaterial({color:0xeaf1ff, metalness:.7, roughness:.15}));
    cab.position.set(0,2.3,-0.3); g.add(cab);
    const wing=new THREE.Mesh(new THREE.BoxGeometry(4.2,0.25,1.1),
      new THREE.MeshStandardMaterial({color:0x0b1020}));
    wing.position.set(0,2.55,-3.2); g.add(wing);
    [[-1.9,-3.4],[1.9,-3.4]].forEach(([x,z])=>{ const p=new THREE.Mesh(new THREE.BoxGeometry(0.25,1.0,0.25),wing.material); p.position.set(x,2.05,z); g.add(p); });
    const wg=new THREE.CylinderGeometry(1,1,0.85,14), wm=new THREE.MeshStandardMaterial({color:0x0a0e18, roughness:.85});
    const wheels=[];
    [[-2,2.3],[2,2.3],[-2,-2.3],[2,-2.3]].forEach(([x,z])=>{ const w=new THREE.Mesh(wg,wm); w.rotation.z=Math.PI/2; w.position.set(x,0.9,z); g.add(w); wheels.push(w); });
    g.userData.wheels=wheels;
    const trail=new THREE.Mesh(new THREE.ConeGeometry(0.7,2.6,10), new THREE.MeshBasicMaterial({color:0xff8a3d, transparent:true, opacity:0}));
    trail.rotation.x=-Math.PI/2; trail.position.set(0,1.2,4.4); g.add(trail); g.userData.trail=trail;
    return g;
  }

  function build(){
    if(built || typeof scene==='undefined' || !scene || typeof car==='undefined' || !car) return;
    built=true; t0=performance.now();
    SPAWN.forEach(s=>{
      const g=makeCar(TEAM[s.t]); g.position.set(s.x,0,s.z); scene.add(g);
      const sh=new THREE.Mesh(new THREE.CircleGeometry(3.8,20), new THREE.MeshBasicMaterial({color:0x000000, transparent:true, opacity:.3}));
      sh.rotation.x=-Math.PI/2; sh.position.y=0.05; scene.add(sh);
      bots.push({ g, sh, t:s.t, x:s.x, z:s.z, y:0, vy:0, ang:s.t==='blue'?0:Math.PI, spd:0, jcd:Math.random()*2 });
    });
    clockEl=document.createElement('div'); clockEl.id='matchClock'; clockEl.textContent='LIVE  00:00';
    document.body.appendChild(clockEl); clockEl.classList.add('on');
    if(typeof scoreB!=='undefined') prevB=scoreB; if(typeof scoreO!=='undefined') prevO=scoreO;
  }

  function clamp(b){ const mx=HALF_X-3, mz=HALF_Z-3;
    if(b.x<-mx)b.x=-mx; else if(b.x>mx)b.x=mx;
    if(b.z<-mz)b.z=-mz; else if(b.z>mz)b.z=mz; }

  function celebrate(side, color){
    const z=side*HALF_Z;
    window.goalCam={ t:1.7, pos:new THREE.Vector3(0,15,z*0.45), look:{x:0,y:6,z:z} };
    window.__cheer=true; setTimeout(function(){ window.__cheer=false; }, 2200);
    const M=34, pos=new Float32Array(M*3), vel=new Float32Array(M*3);
    for(let i=0;i<M;i++){ pos[i*3]=(Math.random()-.5)*10; pos[i*3+1]=10+Math.random()*6; pos[i*3+2]=z+(Math.random()-.5)*6;
      vel[i*3]=(Math.random()-.5)*18; vel[i*3+1]=10+Math.random()*16; vel[i*3+2]=(Math.random()-.5)*18; }
    const g=new THREE.BufferGeometry(); g.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
    const mat=new THREE.PointsMaterial({color, size:1.8, transparent:true, opacity:1});
    const pts=new THREE.Points(g,mat); scene.add(pts);
    bursts.push({pts, vel, life:1.6, max:1.6});
    if(typeof Audio0!=='undefined' && Audio0.goal) Audio0.goal();
  }
  function updateBursts(dt){
    for(let i=bursts.length-1;i>=0;i--){
      const b=bursts[i]; b.life-=dt;
      const p=b.pts.geometry.attributes.position.array;
      for(let k=0;k<p.length;k+=3){ b.vel[k+1]-=GRAV*dt; p[k]+=b.vel[k]*dt; p[k+1]+=b.vel[k+1]*dt; p[k+2]+=b.vel[k+2]*dt; }
      b.pts.geometry.attributes.position.needsUpdate=true;
      b.pts.material.opacity=Math.max(0,b.life/b.max);
      if(b.life<=0){ scene.remove(b.pts); b.pts.geometry.dispose(); b.pts.material.dispose(); bursts.splice(i,1); }
    }
  }
  function checkGoals(){
    if(typeof scoreB==='undefined') return;
    if(scoreB>prevB){ prevB=scoreB; celebrate(1, 0x3d7bff); }
    if(typeof scoreO!=='undefined' && scoreO>prevO){ prevO=scoreO; celebrate(-1, 0xff6a2b); }
  }

  function step(dt){
    if(typeof ball==='undefined' || !ball) return;
    const bx=ball.position.x, bz=ball.position.z, by=ball.position.y;
    for(const b of bots){
      const dir=ATTACK[b.t];
      const want=Math.atan2(bx-b.x, (bz-dir*4)-b.z);
      let da=want-b.ang; while(da>Math.PI)da-=2*Math.PI; while(da<-Math.PI)da+=2*Math.PI;
      b.ang += Math.max(-2.6*dt, Math.min(2.6*dt, da));
      b.spd += 70*dt; b.spd *= 0.96; if(b.spd>92) b.spd=92;
      b.x += Math.sin(b.ang)*b.spd*dt; b.z += Math.cos(b.ang)*b.spd*dt; clamp(b);
      b.jcd -= dt;
      const dball=Math.hypot(b.x-bx, b.z-bz);
      if(b.y<=0 && b.jcd<=0 && ((dball<34 && by>7) || Math.random()<0.005)){ b.vy=48; b.jcd=1.3+Math.random()*1.1; }
      b.vy -= GRAV*dt; b.y += b.vy*dt; if(b.y<0){ b.y=0; b.vy=0; }
      b.g.position.set(b.x,b.y,b.z);
      b.g.rotation.y=b.ang;
      b.g.rotation.x=Math.max(-0.35, Math.min(0.35, -b.vy*0.012));
      if(b.g.userData.wheels) b.g.userData.wheels.forEach(w=>{ w.rotation.x += b.spd*dt*0.5; });
      b.sh.position.set(b.x,0.05,b.z); b.sh.scale.setScalar(Math.max(0.45, 1-b.y/30));
      if(b.g.userData.trail) b.g.userData.trail.material.opacity = b.spd>27 ? 0.5 : 0;
      const d=Math.hypot(b.x-bx, b.z-bz);
      if(d < (4.2+BALL_R) && Math.abs(b.y-by) < 9 && typeof ballV!=='undefined'){
        const nx=(bx-b.x)/(d||1), nz=(bz-b.z)/(d||1), power=24+b.spd*0.8;
        ballV.x += nx*power + Math.sin(b.ang)*8;
        ballV.z += nz*power + dir*6;
        ballV.y += 16;
        if(typeof Audio0!=='undefined' && Audio0.hit) Audio0.hit(0.55);
      }
      if(typeof car!=='undefined' && car){
        const dp=Math.hypot(b.x-car.position.x, b.z-car.position.z);
        if(dp<6.2 && dp>0.001){ const ux=(car.position.x-b.x)/dp, uz=(car.position.z-b.z)/dp, push=(6.2-dp)*0.45;
          car.position.x += ux*push; car.position.z += uz*push; }
      }
    }
    for(let i=0;i<bots.length;i++) for(let j=i+1;j<bots.length;j++){
      const a=bots[i], b=bots[j], d=Math.hypot(a.x-b.x, a.z-b.z);
      if(d<6 && d>0.001){ const ux=(a.x-b.x)/d, uz=(a.z-b.z)/d, p=(6-d)*0.5; a.x+=ux*p; b.x-=ux*p; a.z+=uz*p; b.z-=uz*p; }
    }
  }

  function tickClock(now){
    if(!clockEl) return;
    const s=Math.floor((now-t0)/1000), mm=String(Math.floor(s/60)).padStart(2,'0'), ss=String(s%60).padStart(2,'0');
    clockEl.textContent='LIVE  '+mm+':'+ss;
  }

  function loop(now){
    raf=requestAnimationFrame(loop);
    const dt=Math.min(0.05,(now-last)/1000); last=now;
    updateBursts(dt);
    if(typeof started==='undefined' || !started) return;
    const open = (typeof panel!=='undefined' && panel) ? panel.classList.contains('open') : false;
    if(!built) build();
    if(open) return;
    if(window.__tourActive){ tickClock(now); return; }   // freeze the match while a speaker presents the tour
    step(dt); checkGoals(); tickClock(now);
  }
  requestAnimationFrame(loop);
})();
