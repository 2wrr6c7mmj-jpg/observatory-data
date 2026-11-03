// Vendored unchanged from 2wrr6c7mmj-jpg/notam-auto-update notam-static-trial/src/maritime-window.mjs (commit d21694d).
// NOTAM Watch decides whether a maritime warning is in period with this; update both together.
const bounded=p=>Number.isFinite(p?.start)&&Number.isFinite(p?.end)&&p.end>p.start;
const overlaps=(p,from,to)=>p.start<=to&&p.end>from;
/** Compare compiled activity and enclosing validity evidence without parsing source text. */
export function maritimeWindowStatus(n,from,to){
 const s=n.staticSchedule||{};
 if(s.cancelled||n.cancelled||Number.isFinite(s.selfCancelAt)&&from>=s.selfCancelAt)return 'inactive';
 if(!Number.isFinite(from)||!Number.isFinite(to)||to<from)return 'unknown';
 // A partial notice can still prove individual activity windows. Do not fill
 // their gaps or infer an envelope from a min/max of those windows.
 if(s.openEndedPeriods?.some(p=>Number.isFinite(p.start)&&p.end===null&&p.start<=to))return 'active';
 if(s.periods?.some(p=>bounded(p)&&overlaps(p,from,to)))return 'active';
 // Fully interpreted reserve windows remain conditional. Outside both the
 // primary and reserve windows, the record is demonstrably out of period.
 if(s.conditionalComplete===true&&s.periods?.some(bounded)){
  return s.conditionalPeriods?.some(p=>bounded(p)&&overlaps(p,from,to))?'unknown':'inactive';
 }
 // A calendar/deadline envelope proves bounds, not continuous activity.
 if(s.semanticTime?.calendarPeriods?.some(p=>bounded(p)&&overlaps(p,from,to)))return 'validity';
 const validity=(s.validityPeriods||[]).filter(bounded);
 if(validity.some(p=>overlaps(p,from,to)))return 'validity';
 if(validity.length&&s.validityComplete===true)return 'inactive';
 return s.requiresReview||['partial','unknown'].includes(s.completeness)?'unknown':'inactive';
}
export function maritimeStartsWithin(n,now,hours){
 const s=n.staticSchedule||{};
 if(s.cancelled||n.cancelled||!Number.isFinite(now)||!Number.isFinite(hours)||hours<=0)return false;
 const current=maritimeWindowStatus(n,now,now);
 if(current==='active'||current==='validity')return false;
 const horizon=now+hours*3600000;
 return [...(s.periods||[]).filter(bounded),...(s.validityPeriods||[]).filter(bounded),...(s.semanticTime?.calendarPeriods||[]).filter(bounded),...(s.openEndedPeriods||[]).filter(p=>Number.isFinite(p.start)&&p.end===null)].some(p=>p.start>now&&p.start<=horizon);
}
