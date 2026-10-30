// Vendored unchanged from 2wrr6c7mmj-jpg/notam-auto-update shared/maritime-activity.mjs (commit b54a8d2).
// NOTAM Watch and this dataset classify launches with the same rules; update both together.
/**
 * Maritime topic classification, shared by compilation, search and display.
 * A topic is never evidence of activity time, current validity or cancellation.
 * Facts-only records are classified by their approved source category: their
 * generated public summary must never be mistaken for the original warning.
 */
export const ACTIVITY_VERSION = 1;

export const ACTIVITY_LABELS = Object.freeze({
  missile: 'ミサイル', firing: '射撃', bombing: '爆撃',
  exercise: '訓練・演習', military: '軍事活動', launch: '発射・打上げ',
  drone: '無人機', 'source-category': '射撃・発射等',
});

// Rules are deliberately independent of dates and geometry. Add a named rule
// and a positive/negative fixture together when supporting a new source idiom.
const RULES = [
  {id: 'missile', tag: 'missile', military: true,
    pattern: /ミサイル(?:射撃|発射|試験)?|(?:导弹|導彈|飞弹|飛彈)(?:射击|射擊|发射|發射|试验|試驗)?|\bMISSILE(?:S|[- ]FIRING|[- ]LAUNCH(?:ING)?)?\b|미사일|ракетн(?:ые|ых|ой|ая)\s+(?:стрельб\w*|пуск\w*|испытан\w*)/iu},
  {id: 'firing', tag: 'firing', military: true,
    pattern: /(?:実弾|实弹|實彈)?(?:射爆撃|射撃|射击|射擊)|(?:实弹|實彈)(?:演练|演練|演习|演習|训练|訓練)|砲撃|炮击|炮擊|\b(?:GUNNERY|GUN[- ]?FIRINGS?|LIVE[- ]FIR(?:E|INGS?)|FIRINGS?(?:\s+(?:EXERCISES?|PRACTICE|OPERATIONS?))?|SHOOTING\s+(?:EXERCISES?|PRACTICE)|TARGET\s+PRACTICE)\b|사격|стрельб[а-яё]*/iu},
  {id: 'bombing', tag: 'bombing', military: true,
    pattern: /(?:射)?爆撃|轰炸|轟炸|\b(?:AERIAL\s+)?(?:BOMBING|BOMBARDMENT)\b|폭격|бомбометани[а-яё]*/iu},
  {id: 'military-exercise', tag: 'exercise', military: true,
    pattern: /(?:軍事|军事|軍|军|海軍|海军|海上自衛隊|自衛隊|防衛装備庁).{0,32}(?:活動|活动|訓練|训练|演習|演习|演練|演练|演訓|試験|试验)|军演|軍演|\b(?:MILITARY|NAVAL|ARMED\s+FORCES|NAVY|AIR\s+FORCE)\s+(?:FIRING\s+)?(?:EXERCISES?|TRAINING|OPERATIONS?|ACTIVIT(?:Y|IES)|TESTS?)\b|(?:군사|해군|군).{0,12}(?:훈련|연습)|(?:военн[а-яё]*|морск[а-яё]*)\s+(?:учени[а-яё]*|маневр[а-яё]*)/iu},
  {id: 'military-activity', tag: 'military', military: true,
    pattern: /軍事活動|军事活动|军事训练|軍事訓練|军事任务|軍事任務|\bMILITARY\s+(?:ACTIVIT(?:Y|IES)|OPERATIONS?|MISSIONS?)\b|군사활동/iu},
  {id: 'hazardous-exercise', tag: 'exercise', military: null,
    pattern: /危険な訓練|危險(?:的)?(?:演習|訓練)|\bHAZARDOUS\s+(?:EXERCISES?|TRAINING)\b/iu},
  {id: 'launch', tag: 'launch', military: null,
    pattern: /(?:ミサイル|ロケット|飛しょう体|飛翔体|誘導弾).{0,12}(?:発射|打上げ|打ち上げ|試験)|(?:火箭|导弹|導彈|飞弹|飛彈).{0,12}(?:发射|發射|试验|試驗)|\b(?:MISSILE|ROCKET|LAUNCH\s+VEHICLE)\s+(?:LAUNCH(?:ING)?|FIRING|TESTS?)\b|(?:미사일|로켓).{0,12}(?:발사|시험)|пуск[а-яё]*\s+ракет[а-яё]*/iu},
  // 宇宙ごみ・ロケット部品の落下／再突入は、打上げと同じ「発射・打上げ」に含める。
  {id: 'space-debris', tag: 'launch', military: null,
    pattern: /宇宙(?:ごみ|ゴミ|デブリ|物体|機)|スペースデブリ|(?:ロケット|衛星|人工衛星).{0,16}(?:落下|残骸|破片|再突入|大気圏突入)|(?:落下物|残骸|破片|再突入).{0,16}(?:ロケット|衛星|宇宙)|(?:火箭|运载火箭|運載火箭|卫星|衛星|航天器|太空).{0,16}(?:残骸|殘骸|碎片|坠落|墜落|落区|再入|返回)|(?:残骸|殘骸|碎片|坠落|墜落|落区).{0,16}(?:火箭|卫星|衛星|航天)|\b(?:SPACE\s+DEBRIS|SPACE\s+OBJECTS?|ROCKET\s+(?:DEBRIS|STAGE|BODY|BOOSTER|FRAGMENTS?)|LAUNCH\s+VEHICLE\s+(?:DEBRIS|STAGE|FRAGMENTS?)|SATELLITE\s+(?:DEBRIS|RE-?ENTRY)|(?:ATMOSPHERIC\s+)?RE-?ENTRY\s+(?:OF\s+)?(?:SPACE|ROCKET|SATELLITE|DEBRIS)|(?:DEBRIS|FRAGMENTS?)\s+(?:FALL|IMPACT|LANDING)\s+(?:AREA|ZONE))\b|\b(?:WRECKAGE|REMAINS)\s+(?:FROM|OF)\s+(?:THE\s+)?(?:ROCKET|LAUNCH\s+VEHICLE|SATELLITE)S?\b|\bROCKET\s+WRECKAGE\b|우주(?:쓰레기|물체)|(?:로켓|위성).{0,12}(?:잔해|낙하|재진입)|(?:космическ(?:ий|ого|ие)\s+(?:мусор|объект)|обломк[а-яё]*\s+ракет[а-яё]*|падени[а-яё]*\s+(?:ступен|обломк)[а-яё]*)/iu},
  {id: 'military-drone', tag: 'drone', military: true,
    pattern: /(?:軍事|军事|軍用|军用|自衛隊).{0,24}(?:無人機|无人机|ドローン)|\bMILITARY\s+(?:DRONE|UAV|UNMANNED\s+(?:AIRCRAFT|VEHICLE))S?\b/iu},
];

const CIVIL = /人工降雨|人工增雨|人工増雨|人工炮射|\b(?:RAINMAKING|CLOUD\s+SEEDING|WATER\s+BOMBING|FIRE\s*FIGHTING)\b/i;
const REFERENCE = /^(?:以前の|過去の|旧警報|参考|参照|REFER\s+TO|PREVIOUS|HISTORICAL)/i;
const NEGATION = /(?:実施|実行)(?:しません|しない|されません|されない)|行われません|行われない|中止|取消|取り消|\bNO\s+(?:AERIAL\s+|LIVE[- ]?)?(?:BOMBING|FIRING|GUNNERY|EXERCISE|MILITARY|MISSILE)\b|\b(?:WILL\s+NOT|NOT\s+TAKING\s+PLACE|CANCELLED|CANCELED)\b|不进行|不進行|취소/i;
const FACILITY = /^(?:場(?!域)|场(?!域)|艇|艦|舰|施設|区域の灯|\s+(?:RANGE|AREA)\s+(?:LIGHT|BUOY|BEACON|MARKER|BOUNDARY))|^\s+RANGE\s*(?:[.,]|$)/i;
const DISPOSAL = /不発弾|不發彈|弾薬処理|爆発物処理|\b(?:UNEXPLODED|DISPOSAL|DEMOLITION)\b/i;

function clauses(text) {
  return String(text || '').normalize('NFKC').replace(/\r\n?/g, '\n')
    .split(/\n|[。；;]|(?<=\.)[ \t]+/).map(line => line.trim()).filter(Boolean);
}

/** @returns {{version:number, tags:string[], military:boolean|null, evidence:object[]}} */
export function classifyMaritimeActivity(item) {
  const evidence = [];
  if (item?.publicProjection !== 'facts-only') {
    const lines = clauses(item?.text);
    for (let index = 0; index < lines.length; index++) {
      const clause = lines[index];
      if (CIVIL.test(clause) || REFERENCE.test(clause) || DISPOSAL.test(clause)) continue;
      // A following standalone cancellation applies to the preceding heading.
      if (NEGATION.test(clause) || /^(?:中止|取消|実施しません|CANCELLED|CANCELED)[。.]*$/i.test(lines[index + 1] || '')) continue;
      for (const rule of RULES) {
        const match = rule.pattern.exec(clause);
        if (!match || FACILITY.test(clause.slice(match.index + match[0].length))) continue;
        evidence.push({tag: rule.tag, rule: rule.id, matched: match[0], military: rule.military});
      }
    }
  }
  const tags = [...new Set(evidence.map(match => match.tag))];
  if (!tags.length && (item?.purpose === 'firing' || item?.topic?.verified === true)) {
    tags.push('source-category');
    evidence.push({tag: 'source-category', rule: 'retained-source-category', matched: '', military: null});
  }
  return {version: ACTIVITY_VERSION, tags,
    military: evidence.some(match => match.military === true) ? true : null,
    evidence: evidence.filter((match, index) => evidence.findIndex(other => other.rule === match.rule && other.matched === match.matched) === index),
  };
}

export function activityLabels(activity) {
  return (activity?.tags || []).map(tag => ACTIVITY_LABELS[tag]).filter(Boolean);
}

export function validateMaritimeActivity(activity, {factsOnly = false} = {}) {
  const check = condition => { if (!condition) throw new Error('Invalid maritime activity evidence'); };
  check(activity && Object.keys(activity).sort().join(',') === 'evidence,military,tags,version');
  check(activity.version === ACTIVITY_VERSION && Array.isArray(activity.tags) && activity.tags.length <= 8);
  check(new Set(activity.tags).size === activity.tags.length && activity.tags.every(tag => Object.hasOwn(ACTIVITY_LABELS, tag)));
  check(activity.military === true || activity.military === null);
  check(Array.isArray(activity.evidence) && activity.evidence.length <= 1000);
  for (const entry of activity.evidence) {
    check(entry && Object.keys(entry).sort().join(',') === 'matched,military,rule,tag');
    check(activity.tags.includes(entry.tag) && typeof entry.rule === 'string' && entry.rule.length <= 60);
    check(typeof entry.matched === 'string' && entry.matched.length <= 200 && (entry.military === true || entry.military === null));
    if (factsOnly) check(entry.tag === 'source-category' && entry.rule === 'retained-source-category' && entry.matched === '' && entry.military === null);
  }
  check(activity.tags.every(tag => activity.evidence.some(entry => entry.tag === tag)));
  check(activity.military === (activity.evidence.some(entry => entry.military === true) ? true : null));
  return activity;
}

const projections = new WeakMap();
/** Pure, memoized projection. Never changes original text, dates or geometry. */
export function withMaritimeActivity(item) {
  if (projections.has(item)) return projections.get(item);
  const activity = classifyMaritimeActivity(item);
  const purpose = activity.tags.length ? 'firing' : item.purpose;
  const result = activity.tags.length ? {...item, purpose, activity} : item;
  projections.set(item, result);
  return result;
}
