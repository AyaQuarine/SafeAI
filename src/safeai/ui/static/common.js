/* SafeAI Common V1: lecture JSON uniquement. Aucune requête n'envoie les fichiers locaux. */
'use strict';
const q = id => document.getElementById(id);
const STATUS = {completed:'Terminée',empty:'Silence confirmé',partial:'Partielle',failed:'Échec'};
function el(tag, cls, value) {
  const node = document.createElement(tag);
  if(cls) node.className=cls;
  if(value!==undefined && value!==null) node.textContent=String(value);
  return node;
}
function seconds(v) {return Number(v).toFixed(2).replace('.',',')+' s';}
function formatRange(a,b) {return `${seconds(a)} – ${seconds(b)}`;}
function uniqueSpeakers(diar) {
  return [...new Set((diar?.turns||[]).map(t=>t.speaker_id).filter(Boolean))].sort();
}
function overlapIntervals(turns) {
  const hits=[];
  for(let i=0;i<turns.length;i++) for(let j=i+1;j<turns.length;j++) {
    const a=turns[i],b=turns[j];
    if(!a.speaker_id || !b.speaker_id || a.speaker_id===b.speaker_id) continue;
    const start=Math.max(a.start_s,b.start_s),end=Math.min(a.end_s,b.end_s);
    if(end>start) hits.push([start,end]);
  }
  return hits;
}
function validate(aligned, diar) {
  if(!aligned || typeof aligned!=="object" || Array.isArray(aligned)) throw Error('JSON aligné invalide.');
  if(typeof aligned.session_id!=='string' || !aligned.session_id) throw Error('session_id manquant.');
  if(!['completed','empty','partial','failed'].includes(aligned.status)) throw Error('Statut de session inconnu.');
  if(!Number.isFinite(aligned.duration_s) || aligned.duration_s<0) throw Error('Durée invalide.');
  if(!Array.isArray(aligned.utterances) || !Array.isArray(aligned.diagnostics||[])) throw Error('Énoncés ou diagnostics invalides.');
  for(const u of aligned.utterances) {
    if(u.session_id !== aligned.session_id || typeof u.text!=='string' || !Number.isFinite(u.start_s) || !Number.isFinite(u.end_s) || u.start_s<0 || u.end_s<=u.start_s || u.end_s>aligned.duration_s+0.01) throw Error('Énoncé incohérent dans la session.');
  }
  if(diar) {
    if(!Array.isArray(diar.turns) || diar.session_id!==aligned.session_id || !Number.isFinite(diar.duration_s) || Math.abs(diar.duration_s-aligned.duration_s)>.02) throw Error('Diarisation incompatible : session_id ou durée différent.');
    for(const t of diar.turns) if(!Number.isFinite(t.start_s)||!Number.isFinite(t.end_s)||t.start_s<0||t.end_s<=t.start_s||t.end_s>diar.duration_s+0.01) throw Error('Tour de parole mal formé.');
  }
  return true;
}
function setError(message) {
  const target=q('load-error');target.textContent=message||'';target.classList.toggle('hidden',!message);
}
function render(aligned,diar) {
  validate(aligned,diar);setError('');
  q('m-status').textContent=STATUS[aligned.status];q('m-status').className=`m-status ${aligned.status}`;
  q('m-duration').textContent=seconds(aligned.duration_s);
  const speakers=uniqueSpeakers(diar);
  q('m-speakers').textContent=diar?String(speakers.length):'—';
  q('m-utterances').textContent=String(aligned.utterances.length);
  const timeline=q('speaker-timeline');timeline.replaceChildren();
  const timelineEmpty=q('timeline-empty');
  const turns=diar?.turns||[];
  const intersections=overlapIntervals(turns);
  q('timeline-subtitle').textContent=diar
    ?`${turns.length} tours · ${intersections.length} intersection(s) entre locuteurs distincts. Un compteur de 2 tours « overlap » peut correspondre à une seule intersection.`
    :'Fichier de diarisation non chargé : chronologie indisponible.';
  timelineEmpty.textContent=diar && !turns.length?'Aucun tour de parole dans cette session.':'';
  speakers.forEach((speaker,i)=>{
    const row=el('div','timeline-row');row.append(el('span','timeline-label',speaker));
    const track=el('div','timeline-track');track.setAttribute('role','img');
    track.setAttribute('aria-label',`Tours du locuteur ${speaker}`);
    turns.filter(t=>t.speaker_id===speaker).forEach(t=>{
      const bar=el('span',`timeline-turn sp${i%3}${t.overlap?' overlap':''}`);
      bar.style.left=`${100*t.start_s/aligned.duration_s}%`;
      bar.style.width=`${100*(t.end_s-t.start_s)/aligned.duration_s}%`;
      bar.title=formatRange(t.start_s,t.end_s)+(t.overlap?' · Tour chevauchant':'');
      track.append(bar);
    });
    row.append(track);timeline.append(row);
  });
  if(speakers.length){const ticks=el('div','timeline-ticks');[0,.25,.5,.75,1].forEach(x=>ticks.append(el('span','',seconds(aligned.duration_s*x))));timeline.append(ticks);}
  const utterances=q('utterances');utterances.replaceChildren();
  q('transcript-count').textContent=`${aligned.utterances.length} énoncé(s)`;
  if(!aligned.utterances.length)utterances.append(el('p','muted',aligned.status==='empty'?'Aucune parole transcrite : silence confirmé.':'Aucun énoncé disponible.'));
  aligned.utterances.forEach(u=>{
    const row=el('article','utterance');row.append(el('span','utterance-time',`${u.start_s.toFixed(2)}–${u.end_s.toFixed(2)}`));
    const contents=el('div');
    contents.append(el('span',`utterance-speaker ${u.attribution==='overlap'?'overlap':u.speaker_id?'':'unknown'}`,u.attribution==='overlap'?'Chevauchement':u.speaker_id||'Locuteur inconnu'));
    contents.append(el('div','utterance-text',u.text));row.append(contents);utterances.append(row);
  });
  const diagnostics=q('diagnostics');diagnostics.replaceChildren();
  const messages=aligned.diagnostics||[];
  if(!messages.length)diagnostics.append(el('li','', 'Aucun diagnostic supplémentaire.'));
  else messages.forEach(note=>diagnostics.append(el('li','',note)));
  const metadata=q('metadata');metadata.replaceChildren();
  const fields=[['Session',aligned.session_id],['Provenance',aligned.provenance||'—'],['Source',aligned.source||'—'],['Statut diarisation',diar?.status||'non chargée'],['Tours chevauchants',diar?String(turns.filter(t=>t.overlap).length):'—'],['Intersections',diar?String(intersections.length):'—']];
  fields.forEach(([k,v])=>{metadata.append(el('dt','',k));metadata.append(el('dd','',v));});
}
async function readFile(input,required=false) {
  const file=input.files?.[0];
  if(!file){if(required) throw Error('Sélectionne le JSON aligné.');return null;}
  if(file.size>5*1024*1024) throw Error(`Fichier JSON trop volumineux : ${file.name}.`);
  return JSON.parse(await file.text());
}
async function loadExample(){
  const kind=q('example').value;
  try {
    const [alignedResponse,diarResponse] = await Promise.all([
      fetch(`/static/examples/${kind}_aligned.json`),fetch(`/static/examples/${kind}_diarization.json`)
    ]);
    if(!alignedResponse.ok||!diarResponse.ok) throw Error('Exemple indisponible.');
    render(await alignedResponse.json(),await diarResponse.json());
  }catch(error){setError(error.message);}
}
async function loadFiles(){
  try{const aligned=await readFile(q('aligned-file'),true);const diar=await readFile(q('diarization-file'));render(aligned,diar);}
  catch(error){setError(`Chargement refusé : ${error.message}`);}
}
q('load-example').addEventListener('click',loadExample);
q('load-files').addEventListener('click',loadFiles);
loadExample();
