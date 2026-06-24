/* =====================================================================
   STADIUM — tiered stands, a crowd, and floodlight pylons around the
   pitch, so the match feels like it's in a real arena. Loaded AFTER
   app.js; builds into the global `scene`. Purely decorative (no physics).
   ===================================================================== */
(function(){
  "use strict";
  if(typeof THREE==='undefined') return;
  if(typeof scene==='undefined' || !scene) return;     // arena didn't build (no WebGL)
  const HX=145, HZ=200;
  const stadium=new THREE.Group(); scene.add(stadium);

  // surrounding dark floor so the void is filled
  const skirt=new THREE.Mesh(new THREE.PlaneGeometry(420,420),
    new THREE.MeshStandardMaterial({color:0x06070e, roughness:1}));
  skirt.rotation.x=-Math.PI/2; skirt.position.y=-0.4; stadium.add(skirt);

  // tiered stands (concentric rising ledges)
  const tierMat=new THREE.MeshStandardMaterial({color:0x0f1320, roughness:.92, metalness:.05});
  const edgeMat=new THREE.MeshBasicMaterial({color:0x1c2942});
  const nT=9, stepOut=5, rise=3.2, w=4.6;
  const cPos=[], cCol=[];
  const palette=[[0.16,0.20,0.30],[0.95,0.45,0.18],[0.26,0.50,1.0],[0.88,0.88,0.95],[0.72,0.26,0.62]];
  function ledge(sx,sz,px,py,pz){
    const m=new THREE.Mesh(new THREE.BoxGeometry(sx,1.2,sz),tierMat); m.position.set(px,py,pz); stadium.add(m);
    const e=new THREE.Mesh(new THREE.BoxGeometry(sx,0.3,sz),edgeMat); e.position.set(px,py+0.75,pz); stadium.add(e);
  }
  for(let i=0;i<nT;i++){
    const xh=HX+6+i*stepOut, zh=HZ+6+i*stepOut, y=1+i*rise;
    ledge(xh*2,w, 0,y,-zh); ledge(xh*2,w, 0,y, zh);
    ledge(w,zh*2, -xh,y,0); ledge(w,zh*2,  xh,y,0);
    const dens=Math.floor((xh+zh)*0.45);
    for(let k=0;k<dens;k++){
      const side=k%4; let x,z;
      if(side<2){ x=(Math.random()*2-1)*xh; z=(side?zh:-zh)+(Math.random()-.5)*w; }
      else      { z=(Math.random()*2-1)*zh; x=(side===2?-xh:xh)+(Math.random()-.5)*w; }
      cPos.push(x, y+1.3+Math.random()*0.7, z);
      const c=palette[(Math.random()*palette.length)|0], j=0.65+Math.random()*0.5;
      cCol.push(c[0]*j, c[1]*j, c[2]*j);
    }
  }
  const cg=new THREE.BufferGeometry();
  cg.setAttribute('position', new THREE.Float32BufferAttribute(cPos,3));
  cg.setAttribute('color',    new THREE.Float32BufferAttribute(cCol,3));
  const crowdPts=new THREE.Points(cg, new THREE.PointsMaterial({size:1.7, vertexColors:true, sizeAttenuation:true}));
  stadium.add(crowdPts);
  // cheering crowd: each seat bobs on its own phase so the stands ripple; a goal (window.__cheer) makes them jump
  const _cbase=Float32Array.from(cPos), _cph=new Float32Array(cPos.length/3);
  for(let i=0;i<_cph.length;i++) _cph[i]=Math.random()*Math.PI*2;
  const _cpa=cg.getAttribute('position'); let _clast=performance.now();
  (function cheer(now){ requestAnimationFrame(cheer);
    if(now-_clast<33) return; _clast=now;
    const t=now*0.004, amp=(typeof window!=='undefined' && window.__cheer)?2.4:0.7, a=_cpa.array;
    for(let i=0;i<_cph.length;i++) a[i*3+1]=_cbase[i*3+1]+Math.sin(t+_cph[i])*amp;
    _cpa.needsUpdate=true;
  })(performance.now());

  // floodlight pylons at the four corners (emissive heads, no extra real lights)
  [[-1,-1],[1,-1],[-1,1],[1,1]].forEach(([sx,sz])=>{
    const px=sx*(HX+16), pz=sz*(HZ+16);
    const pole=new THREE.Mesh(new THREE.CylinderGeometry(0.9,0.9,46,8),
      new THREE.MeshStandardMaterial({color:0x161c2c, roughness:.7}));
    pole.position.set(px,23,pz); stadium.add(pole);
    const head=new THREE.Mesh(new THREE.BoxGeometry(11,3.4,2),
      new THREE.MeshBasicMaterial({color:0xe8f0ff}));
    head.position.set(px,46,pz); head.lookAt(0,0,0); stadium.add(head);
    const glow=new THREE.Mesh(new THREE.SphereGeometry(7,12,12),
      new THREE.MeshBasicMaterial({color:0xbfd4ff, transparent:true, opacity:.06}));
    glow.position.set(px,46,pz); stadium.add(glow);
  });
})();
