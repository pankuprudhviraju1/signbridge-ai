const signs=[['HELLO',94],['THANKS',91],['YES',96],['NO',89],['I LOVE YOU',93]];let timer;
const byId=(id)=>document.getElementById(id);
function show([label,score]){byId('sign').textContent=label;byId('score').textContent=score+'%';byId('bar').style.width=score+'%'}
byId('demo').onclick=()=>{clearInterval(timer);let i=0;byId('words').textContent='';byId('status').textContent='RECOGNIZING';show(signs[i]);byId('words').textContent=signs[i][0];timer=setInterval(()=>{i++;if(i===signs.length){clearInterval(timer);byId('status').textContent='STABLE';return}show(signs[i]);byId('words').textContent+=' '+signs[i][0]},850)};
byId('camera').onclick=async()=>{try{const stream=await navigator.mediaDevices.getUserMedia({video:true});byId('video').srcObject=stream;await byId('video').play();document.querySelector('.hand').style.display='none';byId('status').textContent='CAMERA ACTIVE'}catch{byId('status').textContent='CAMERA DENIED'}};
window.onLandmarks=(prediction)=>show([prediction.label,Math.round(prediction.confidence*100)]);
