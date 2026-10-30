// Vendored unchanged from 2wrr6c7mmj-jpg/notam-auto-update shared/snapshot-transport.mjs (commit b54a8d2).
// NOTAM Watch publishes its snapshot with this transport; update both together.
import {COMPRESSED_FILE, validateCompression, compressSnapshot, decompressSnapshot} from './snapshot-compression.mjs';
// Lossless static transport. Hashes bind raw bytes before decoding or parsing.
export const SNAPSHOT_LIMIT=64*1024*1024,PART_SIZE=8*1024*1024,ASSET_LIMIT=25*1024*1024;
export const PART_FILE=/^snapshot-[a-f0-9]{16}\.part-[0-9]{3}-[a-f0-9]{16}\.bin$/;
export const isSnapshotAsset=name=>/^snapshot-[a-f0-9]{16}\.json$/.test(name)||PART_FILE.test(name)||COMPRESSED_FILE.test(name);
const check=(v,m)=>{if(!v)throw Error(m);};
const keys=(v,expected)=>v&&typeof v==='object'&&!Array.isArray(v)&&Object.keys(v).sort().join(',')===expected.split(',').sort().join(',');
export async function snapshotHash(bytes){return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),x=>x.toString(16).padStart(2,'0')).join('');}
export function validateTransport(manifest){
 check(/^[a-f0-9]{64}$/.test(manifest?.sha256||'')&&manifest.file===`snapshot-${manifest.sha256.slice(0,16)}.json`,'Snapshot identity invalid');
 validateCompression(manifest);
 if(!Object.hasOwn(manifest,'transport'))return null;const t=manifest.transport;
 check(keys(t,'schemaVersion,kind,byteLength,parts')&&t.schemaVersion===1&&t.kind==='byte-chunks'&&Number.isSafeInteger(t.byteLength)&&t.byteLength>ASSET_LIMIT&&t.byteLength<=SNAPSHOT_LIMIT&&Array.isArray(t.parts)&&t.parts.length===Math.ceil(t.byteLength/PART_SIZE),'Snapshot transport invalid');
 let sum=0;const names=new Set();for(let i=0;i<t.parts.length;i++){const p=t.parts[i];check(keys(p,'file,sha256,byteLength')&&/^[a-f0-9]{64}$/.test(p.sha256||'')&&p.file===`snapshot-${manifest.sha256.slice(0,16)}.part-${String(i).padStart(3,'0')}-${p.sha256.slice(0,16)}.bin`&&!names.has(p.file)&&Number.isSafeInteger(p.byteLength)&&p.byteLength===Math.min(PART_SIZE,t.byteLength-sum),'Snapshot part invalid');sum+=p.byteLength;names.add(p.file);}check(sum===t.byteLength,'Snapshot length mismatch');return t;
}
export async function makeSnapshotTransport(bytes,{compression=false}={}){
 check(bytes instanceof Uint8Array&&bytes.byteLength>0&&bytes.byteLength<=SNAPSHOT_LIMIT,'Snapshot size invalid');
 const packed=compression?await compressSnapshot(bytes,await snapshotHash(bytes)):null;
 const compressed=packed?.descriptor,extra=packed?new Map([[compressed.file,packed.bytes]]):new Map();
 if(bytes.byteLength<=ASSET_LIMIT)return{transport:null,parts:extra,...(compressed?{compressed}:{})};
 const full=await snapshotHash(bytes),parts=extra,list=[];for(let at=0,index=0;at<bytes.byteLength;at+=PART_SIZE,index++){const part=bytes.subarray(at,at+PART_SIZE),sha256=await snapshotHash(part),file=`snapshot-${full.slice(0,16)}.part-${String(index).padStart(3,'0')}-${sha256.slice(0,16)}.bin`;parts.set(file,part);list.push({file,sha256,byteLength:part.byteLength});}
 return{transport:{schemaVersion:1,kind:'byte-chunks',byteLength:bytes.byteLength,parts:list},parts,...(compressed?{compressed}:{})};
}
export async function readSnapshotBytes(manifest,reader,{signal,maxBytes=SNAPSHOT_LIMIT}={}){
 check(Number.isSafeInteger(maxBytes)&&maxBytes>0&&maxBytes<=SNAPSHOT_LIMIT,'Snapshot bound invalid');signal?.throwIfAborted();const t=validateTransport(manifest);let bytes;
 const compressed=validateCompression(manifest);
 if(compressed&&typeof DecompressionStream==='function'){
  check(compressed.decodedByteLength<=maxBytes,'Snapshot exceeds bound');
  try {
   const packed=await reader(compressed.file,compressed.byteLength);
   const decoded=await decompressSnapshot(packed,compressed,{signal});
   check(await snapshotHash(decoded)===manifest.sha256,'Snapshot SHA256 mismatch');
   signal?.throwIfAborted();return decoded;
  } catch(error) { signal?.throwIfAborted(); /* Retained legacy assets are an independently hash-verified fallback. */ }
 }
 if(!t){bytes=await reader(manifest.file,maxBytes);check(bytes instanceof Uint8Array&&bytes.byteLength>0&&bytes.byteLength<=maxBytes,'Snapshot bytes invalid');}
 else{check(t.byteLength<=maxBytes,'Snapshot exceeds bound');bytes=new Uint8Array(t.byteLength);let at=0;for(const p of t.parts){signal?.throwIfAborted();const part=await reader(p.file,p.byteLength);check(part instanceof Uint8Array&&part.byteLength===p.byteLength&&await snapshotHash(part)===p.sha256,'Snapshot part hash or length mismatch');bytes.set(part,at);at+=part.byteLength;}}
 signal?.throwIfAborted();check(await snapshotHash(bytes)===manifest.sha256,'Snapshot SHA256 mismatch');signal?.throwIfAborted();return bytes;
}
// Retained immutable assets use the same 16-hex filename identity as retained JSON.
// Active snapshots additionally require the full descriptor/full SHA above.
export async function joinRetainedParts(entries){
 check(Array.isArray(entries)&&entries.length>=4&&entries.length<=8,'Retained part count invalid');
 const rows=entries.map(([file,bytes])=>{check(PART_FILE.test(file)&&bytes instanceof Uint8Array,'Retained part invalid');return{file,bytes,prefix:file.slice(9,25),index:Number(file.slice(31,34)),partHash:file.slice(35,51)};}).sort((a,b)=>a.index-b.index);
 let length=0;for(let i=0;i<rows.length;i++){const r=rows[i];check(r.index===i&&r.prefix===rows[0].prefix&&r.bytes.byteLength>0&&r.bytes.byteLength<=PART_SIZE&&(i===rows.length-1||r.bytes.byteLength===PART_SIZE)&&(await snapshotHash(r.bytes)).slice(0,16)===r.partHash,'Retained part identity mismatch');length+=r.bytes.byteLength;}
 check(length>ASSET_LIMIT&&length<=SNAPSHOT_LIMIT,'Retained length invalid');const bytes=new Uint8Array(length);let at=0;for(const r of rows){bytes.set(r.bytes,at);at+=r.bytes.byteLength;}check((await snapshotHash(bytes)).slice(0,16)===rows[0].prefix,'Retained snapshot identity mismatch');return{file:`snapshot-${rows[0].prefix}.json`,bytes};
}
