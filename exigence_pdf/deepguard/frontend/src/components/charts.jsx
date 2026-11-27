/** Hand-rolled SVG charts - no chart library, same look as the approved
 *  single-file app: brand blue lines, soft fills, mono tick labels. */
import { useId } from 'react'

const F = (n) => Number(n).toLocaleString('en-US')

function niceMax(v) { if (v <= 0) return 1; const m = 10 ** Math.floor(Math.log10(v)); return Math.ceil(v / m) * m }

export function Sparkline({ data, w = 96, h = 28, stroke = 'var(--brand,#3b6bff)', fill = true }) {
  if (!data?.length) return <svg width={w} height={h} />
  const max = Math.max(...data, 1), min = Math.min(...data, 0)
  const span = max - min || 1
  const pts = data.map((v, i) => [i / (data.length - 1 || 1) * w, h - 3 - (v - min) / span * (h - 6)])
  const line = pts.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join('')
  const gid = useId()
  return (
    <svg width={w} height={h} className="overflow-visible">
      {fill && <>
        <defs>
          <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={stroke} stopOpacity=".3" />
            <stop offset="100%" stopColor={stroke} stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={`${line} L${w},${h} L0,${h} Z`} fill={`url(#${gid})`} stroke="none" />
      </>}
      <path d={line} fill="none" stroke={stroke} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={pts[pts.length - 1][0]} cy={pts[pts.length - 1][1]} r="2.4" fill={stroke} />
    </svg>
  )
}

export function LineArea({ data, labels, height = 220, series = null, yFormat = F, area = true, className = '' }) {
  // data: array of series [{name, values, color}] or plain array of numbers.
  const many = series || [{ name: '', values: data, color: '#3b6bff' }]
  const W = 640, H = height, padL = 42, padR = 10, padT = 14, padB = 26
  const all = many.flatMap(s => s.values.filter(v => v != null))
  const maxV = niceMax(Math.max(...all, 1))
  const n = many[0].values.length
  const x = (i) => padL + i / (n - 1 || 1) * (W - padL - padR)
  const y = (v) => padT + (1 - v / maxV) * (H - padT - padB)
  const gid = useId()
  return (
    <div className={className}>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height }}>
        {many.map((s, si) => {
          const line = s.values.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v || 0).toFixed(1)}`).join('')
          return (
            <g key={s.name || si}>
              {area && si === 0 && (
                <path d={`${line} L${x(n - 1)},${H - padB} L${padL},${H - padB} Z`} fill={`url(#${gid})`} />
              )}
              <path d={line} fill="none" stroke={s.color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
              {s.values.map((v, i) => v != null && (
                <circle key={i} cx={x(i)} cy={y(v)} r="2.6" fill={s.color}>
                  <title>{`${labels?.[i] || ''} — ${s.name || ''} ${yFormat(v)}`.trim()}</title>
                </circle>
              ))}
            </g>
          )
        })}
        <defs>
          <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#3b6bff" stopOpacity=".28" />
            <stop offset="100%" stopColor="#3b6bff" stopOpacity="0" />
          </linearGradient>
        </defs>
        {[0, .25, .5, .75, 1].map(f => (
          <g key={f}>
            <line x1={padL} x2={W - padR} y1={y(maxV * f)} y2={y(maxV * f)} stroke="var(--line)" strokeDasharray="3 4" />
            <text x={padL - 6} y={y(maxV * f) + 3.5} textAnchor="end" className="num" fontSize="9" fill="var(--mute)">{yFormat(Math.round(maxV * f))}</text>
          </g>
        ))}
        {labels?.map((l, i) => (i % Math.ceil(n / 8) === 0 || i === n - 1) && (
          <text key={i} x={x(i)} y={H - 8} textAnchor="middle" className="num" fontSize="9" fill="var(--mute)">{l}</text>
        ))}
      </svg>
      {many.length > 1 && (
        <div className="flex flex-wrap gap-4 mt-1 px-2">
          {many.map(s => (
            <span key={s.name} className="flex items-center gap-1.5 text-[11px] text-mute">
              <span className="h-2 w-2 rounded-full" style={{ background: s.color }} />{s.name}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

export function Donut({ data, size = 170, thickness = 20, center = null }) {
  // data: [{label, value, color}]
  const total = data.reduce((a, d) => a + d.value, 0) || 1
  const r = (size - thickness) / 2, C = 2 * Math.PI * r
  let acc = 0
  const segs = data.filter(d => d.value > 0).map(d => {
    const frac = d.value / total
    const seg = { ...d, dash: `${frac * C} ${C - frac * C}`, offset: -acc * C }
    acc += frac
    return seg
  })
  return (
    <div className="flex items-center gap-6 flex-wrap justify-center">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="shrink-0">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--line)" strokeWidth={thickness} />
        {segs.map(s => (
          <circle key={s.label} cx={size / 2} cy={size / 2} r={r} fill="none" stroke={s.color}
            strokeWidth={thickness} strokeDasharray={s.dash} strokeDashoffset={s.offset}
            transform={`rotate(-90 ${size / 2} ${size / 2})`} strokeLinecap="butt">
            <title>{`${s.label}: ${s.value}`}</title>
          </circle>
        ))}
        {center && <foreignObject x="0" y="0" width={size} height={size}>
          <div xmlns="http://www.w3.org/1999/xhtml" className="w-full h-full flex flex-col items-center justify-center">
            {center}
          </div>
        </foreignObject>}
      </svg>
      <div className="space-y-2.5">
        {data.map(d => (
          <div key={d.label} className="flex items-center gap-2.5 text-sm">
            <span className="h-2.5 w-2.5 rounded-sm" style={{ background: d.color }} />
            <span className="text-mute">{d.label}</span>
            <span className="num font-semibold">{F(d.value)}</span>
            <span className="num text-faint text-xs">({Math.round(d.value / total * 100)}%)</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export function Bars({ data, height = 190, color = '#8b5cf6', labels, yFormat = F }) {
  const W = 640, H = height, padL = 36, padB = 24, padT = 12
  const maxV = niceMax(Math.max(...data, 1))
  const bw = (W - padL - 8) / (data.length || 1)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height }}>
      {[0, .5, 1].map(f => (
        <g key={f}>
          <line x1={padL} x2={W - 4} y1={padT + (1 - f) * (H - padT - padB)} y2={padT + (1 - f) * (H - padT - padB)} stroke="var(--line)" strokeDasharray="3 4" />
          <text x={padL - 6} y={padT + (1 - f) * (H - padT - padB) + 3.5} textAnchor="end" className="num" fontSize="9" fill="var(--mute)">{yFormat(Math.round(maxV * f))}</text>
        </g>
      ))}
      {data.map((v, i) => {
        const h = Math.max(2, v / maxV * (H - padT - padB))
        return (
          <rect key={i} x={padL + i * bw + bw * 0.22} y={H - padB - h} width={bw * 0.56} height={h}
            rx="4" fill={color} opacity={v ? '.9' : '.25'}>
            <title>{`${labels?.[i] || ''} — ${F(v)}`}</title>
          </rect>
        )
      })}
      {labels?.map((l, i) => (i % Math.ceil(data.length / 8) === 0 || i === data.length - 1) && (
        <text key={i} x={padL + i * bw + bw / 2} y={H - 7} textAnchor="middle" className="num" fontSize="9" fill="var(--mute)">{l}</text>
      ))}
    </svg>
  )
}
