// Vendored unchanged from 2wrr6c7mmj-jpg/notam-auto-update shared/aviation-activity.mjs (commit b54a8d2).
// NOTAM Watch and this dataset classify launches with the same rules; update both together.
/**
 * Military-topic recognition for aviation NOTAMs, used by the military map tab.
 * Like maritime-activity.mjs, a topic is never evidence of activity time,
 * validity, cancellation or geometry; it only says what the NOTAM is about.
 * Rules read the E) field only, so Q-line codes and altitudes (MSL) are not text.
 */
export const AVIATION_ACTIVITY_VERSION = 1;

export const AVIATION_ACTIVITY_LABELS = Object.freeze({
  launch: 'ミサイル・ロケット発射', firing: '射撃・爆撃', exercise: '軍事演習・訓練',
  military: '軍の活動', gnss: 'GPS等の電波障害',
});

// Named multinational / national exercises seen in the region's NOTAMs.
const EXERCISE_NAMES = /\b(?:KEEN\s+(?:SWORD|EDGE)|ACTIVE\s+SHIELD|RESOLUTE\s+DRAGON|ORIENT\s+SHIELD|VALIANT\s+SHIELD|COPE\s+NORTH|RED\s+FLAG\s+(?:ALASKA|EXER(?:CISE)?)|MALABAR|FREEDOM\s+(?:EDGE|SHIELD|FLAG)|ULCHI|BALIKATAN|HAN\s+KUANG|PACIFIC\s+VANGUARD|TALISMAN\s+SABRE|RIMPAC)\b/;

const RULES = [
  {id: 'missile', tag: 'launch',
    pattern: /\bMISSILES?\b|\bANTI-?BALLISTIC\b|\bOBJECT\s+PROPELLED\s+BY\s+ROCKET\b|\bROCKET\s+LAUNCH(?:ING|ED)?\b|\bLAUNCH\s+VEH(?:ICLE)?\b|\bSATELLITE\s+LAUNCH\b|\bAEROSPACE\s+(?:FLT|FLIGHT)\b|\bSPACE\s+(?:LAUNCH|VEHICLE)\b/},
  {id: 'space-debris', tag: 'launch',
    pattern: /\b(?:ROCKET|LAUNCH\s+VEH(?:ICLE)?|SPACE)\s+(?:DEBRIS|STAGE|FRAGMENTS?|WRECKAGE)\b|\bRE-?ENTRY\b|\b(?:UNBURNED\s+)?DEBRIS\s+(?:FALL|IMPACT|DROP)|\bFALL\s+AREA\s+OF\b|\bEST\s+FALL\s+AREA\b/},
  {id: 'firing', tag: 'firing',
    pattern: /\bFRNG\b|\b(?:GUN|LIVE|ARTILLERY|AIR[- ]TO[- ](?:AIR|GROUND|SURFACE))\s*(?:FRNG|FIRING|FIRE)\b|\bFIRING\b|\bGUNNERY\b|\bARTILLERY\b|\bMORTARS?\b|\bHOWITZERS?\b|\b\d{2,3}\s*(?:MORTAR|HOWITZER)\b|\bBOMBING\b|\bSHOOTING\s+(?:EXER|PRACTICE|RANGE\s+ACT)/},
  {id: 'named-exercise', tag: 'exercise', pattern: EXERCISE_NAMES},
  {id: 'exercise', tag: 'exercise',
    pattern: /\b(?:MIL(?:ITARY)?|AIR|NAVAL|JOINT|COMBINED|AIR\s+DEFEN[CS]E)\s+(?:EXER(?:CISES?)?|DRILLS?|TRG|TRAINING)\b|\bEXER(?:CISE)?S?\s+(?:WILL\s+BE\s+CONDUCTED|AREA|ACT)\b|^\s*(?:AIR\s+)?EXER\b/},
  {id: 'forces', tag: 'military',
    pattern: /\bSELF\s+DEFEN[CS]E\s+FORCES?\s+ACT|\bU\.?S\.?\s+MIL(?:ITARY)?\s+ACT|\bMIL(?:ITARY)?\s+(?:ACT|ACTIVIT(?:Y|IES)|OPS|OPERATIONS?|MISSIONS?|FLT|FLIGHTS?)\b|\bAIR\s+DEFEN[CS]E\b|\bAIR(?:BORNE)?\s+(?:REFUEL(?:L)?ING|REFUELING)\b|\bAAR\s+(?:AREA|ACT|TRACK)\b|\bAIR\s*DROP\b|\bMASS\s+MOVEMENT\s+OF\s+(?:MIL\s+)?ACFT\b/},
  {id: 'military-source', tag: 'military',
    // Japanese NOTAMs name the requesting service; a JSDF/US-forces source makes
    // parachute, drone and other operations military without guessing from Q-codes.
    pattern: /\bSOURCE\s*\/\s*(?:JSDF|JGSDF|JMSDF|JASDF|USFJ|USAF|USN|USMC|US\s*ARMY)\b|\b(?:JAPAN\s+)?(?:GROUND|MARITIME|AIR)\s+SELF\s+DEFEN[CS]E\s+FORCE\b/},
  {id: 'gnss', tag: 'gnss',
    pattern: /\b(?:GNSS|GPS)\b[^.]{0,80}\b(?:INTERFERENCE|JAMMING|JAMMED|SPOOFING|UNRELIABLE|LOST|DEGRADED|OUTAGES?)\b|\b(?:INTERFERENCE|JAMMING|SPOOFING)\b[^.]{0,40}\b(?:GNSS|GPS)\b/},
];

// Not military even though a rule word appears.
const CIVIL = /\bFIREWORKS?\b|\bMODEL\s+ROCKETS?\b|\bLASER\s+(?:LIGHT\s+)?SHOW\b|\bWATER\s+BOMBING\b|\bFIRE\s*FIGHTING\b|\bRAINMAKING\b|\bCLOUD\s+SEEDING\b|\bEXER(?:CISE)?\s+(?:EXTREME\s+)?(?:CAUTION|CTN)\b|\bBIRD\b[^.]{0,20}\bSHOOTING\b/;
// A whole-NOTAM cancellation or completion is not a new activity.
const ENDED = /^[\s\S]{0,80}(?:\bNOTAM\s+CNL|\sCNL|\sCMPL)\.?\s*$/;

/** E) field, without the trailing F)/G) altitude fields. */
export function aviationBody(text = '') {
  const t = String(text).toUpperCase().replace(/\r\n?/g, '\n');
  const marker = /(?:^|[\s(])E\)\s*/.exec(t);
  return (marker ? t.slice(marker.index + marker[0].length) : t).replace(/\s+[FG]\)[\s\S]*$/, '').trim();
}

/** @returns {{version:number, tags:string[], evidence:{tag:string, rule:string, matched:string}[]}} */
export function classifyAviationActivity(item) {
  const body = aviationBody(item?.text);
  const evidence = [];
  if (body && !ENDED.test(body)) {
    const scrubbed = body.replace(new RegExp(CIVIL.source, 'g'), ' ');
    for (const rule of RULES) {
      const match = rule.pattern.exec(scrubbed);
      if (match) evidence.push({tag: rule.tag, rule: rule.id, matched: match[0].trim().replace(/\s+/g, ' ')});
    }
    // A civil-only NOTAM (fireworks, model rocket) keeps no firing/launch topic.
    if (CIVIL.test(body) && evidence.every(entry => entry.tag === 'firing' || entry.tag === 'launch')) evidence.length = 0;
  }
  return {version: AVIATION_ACTIVITY_VERSION, tags: [...new Set(evidence.map(entry => entry.tag))], evidence};
}
