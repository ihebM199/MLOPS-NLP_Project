import { useCallback, useEffect, useState } from 'react'
import { Api } from '../../api/client.js'
import { useToast } from '../../context/ToastContext.jsx'
import { Icon, P, Card, Badge, Skeleton, EmptyState } from '../../components/ui.jsx'

const fmtMs = (ts) => new Date(ts).toLocaleString()

function fmtMetric(v) {
  if (v == null) return '—'
  if (typeof v === 'number') return v >= 1 ? v.toFixed(3) : v.toFixed(4)
  return String(v)
}

function MetricTone({ k }) {
  if (/auc|accuracy|f1|precision|recall/i.test(k)) return 'text-ok'
  return 'text-ink'
}

export default function AdminModel() {
  const pushToast = useToast()
  const [meta, setMeta] = useState(null)
  const [health, setHealth] = useState(null)
  const [error, setError] = useState('')
  const [updated, setUpdated] = useState(null)

  const load = useCallback(async () => {
    try {
      const [m, h] = await Promise.all([Api.metadata(), Api.health()])
      setMeta(m); setHealth(h); setError(''); setUpdated(Date.now())
    } catch (e) {
      setError(e.message)
      pushToast('Model load failed', e.message, 'bad')
    }
  }, [pushToast])

  useEffect(() => { load() }, [load])
  useEffect(() => {
    const t = setInterval(load, 30000)
    return () => clearInterval(t)
  }, [load])

  if (error && !meta) {
    return <EmptyState icon={P.cpu} title="Model API unreachable"
      message={`${error} — start the backend with python manage.py runserver.`}
      action={<button onClick={load} className="btn-primary"><Icon d={P.refresh} className="h-4 w-4" /> Retry</button>} />
  }
  if (!meta || !health) {
    return <div className="space-y-5">
      <Card className="p-5"><Skeleton className="h-8 w-64" /><Skeleton className="h-24 w-full mt-4" /></Card>
      <Card className="p-5"><Skeleton className="h-48 w-full" /></Card>
    </div>
  }

  const ok = health.status === 'ok'
  const metrics = meta.metrics || {}
  const metricBlocks = [
    ['Test', metrics.test], ['Validation', metrics.valid], ['Cross-dataset', metrics.cross_dataset],
  ].filter(([, v]) => v && typeof v === 'object')
  const robustness = metrics.robustness || {}
  const limitations = meta.limitations || []

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold tracking-tight">Model monitoring</h2>
          <p className="text-sm text-mute mt-1">
            Live model card and service health — read directly from <span className="num">/model/metadata</span> and
            <span className="num"> /health</span>.
          </p>
        </div>
        <div className="flex gap-2.5">
          <Badge tone={ok ? 'ok' : 'bad'} dot>{ok ? 'MODEL SERVING' : 'DEGRADED'}</Badge>
          <Badge tone="violet">v{meta.model_version}</Badge>
          <button onClick={load} className="btn-ghost !py-2"><Icon d={P.refresh} className="h-4 w-4" /> Refresh</button>
        </div>
      </div>

      {/* Production card */}
      <Card className="p-5 relative overflow-hidden">
        <div className="absolute -top-20 right-10 h-44 w-64 rounded-full bg-violet/15 blur-3xl pointer-events-none" />
        <div className="relative grid lg:grid-cols-[auto_1fr_auto] gap-6 items-center">
          <div className="h-14 w-14 rounded-2xl bg-violet/12 text-violet border border-violet/25 grid place-items-center">
            <Icon d={P.cpu} className="h-7 w-7" />
          </div>
          <div>
            <div className="flex items-center gap-2.5 flex-wrap">
              <h3 className="font-bold text-lg">{meta.name}</h3>
              <Badge tone="ok" dot>PRODUCTION</Badge>
              {meta.git_sha && <span className="text-[11px] text-faint num">git {String(meta.git_sha).slice(0, 8)}</span>}
            </div>
            <p className="text-xs text-mute mt-1.5 max-w-xl leading-relaxed">
              {meta.decision_rule} · classes: 0 = {meta.classes?.['0']}, 1 = {meta.classes?.['1']}
              {meta.trained_at ? ` · trained ${fmtMs(meta.trained_at)}` : ''}
            </p>
          </div>
          <div className="grid grid-cols-3 gap-5">
            {[
              ['Architecture', meta.architecture],
              ['Backbone', meta.backbone],
              ['Mode', meta.mode],
            ].map(([k, v]) => (
              <div key={k} className="min-w-0">
                <p className="text-[10px] text-mute font-semibold uppercase tracking-wider">{k}</p>
                <p className="text-sm font-extrabold num mt-1 truncate" title={v}>{v || '—'}</p>
              </div>
            ))}
          </div>
        </div>
      </Card>

      {/* Service health + config grid */}
      <div className="grid lg:grid-cols-3 gap-5">
        <Card className="p-5">
          <h3 className="font-semibold mb-4 flex items-center gap-2"><Icon d={P.activity} className="h-4 w-4 text-brand" /> Service health</h3>
          <div className="space-y-3">
            {[
              ['Status', ok ? 'ok' : 'bad', health.status],
              ['Model loaded', health.model_loaded ? 'ok' : 'bad', String(health.model_loaded)],
              ['Grad-CAM', health.gradcam_available ? 'ok' : 'warn', String(health.gradcam_available)],
              ['GPU', health.gpu_available ? 'ok' : 'mute', String(health.gpu_available)],
              ['Providers', 'mute', (health.onnxruntime_providers || []).join(', ')],
              ['Uptime', 'mute', `${Math.round(health.uptime_s)} s`],
              ['Input size', 'mute', `${meta.image_size}×${meta.image_size}`],
              ['Parameters', 'mute', meta.parameters != null ? (meta.parameters / 1e6).toFixed(1) + ' M' : '—'],
            ].map(([name, tone, val]) => (
              <div key={name} className="flex items-center justify-between gap-3 rounded-xl bg-panel2 border border-line px-4 py-2.5">
                <span className="text-sm">{name}</span>
                <span className={`chip ${tone === 'ok' ? 'text-ok bg-ok/10 border-ok/25' : tone === 'bad' ? 'text-bad bg-bad/10 border-bad/25'
                  : tone === 'warn' ? 'text-warn bg-warn/10 border-warn/25' : 'text-mute bg-panel border-line'} num`}>
                  <span className="h-1.5 w-1.5 rounded-full bg-current" />{val}
                </span>
              </div>
            ))}
          </div>
        </Card>

        <Card className="lg:col-span-2 p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="font-semibold">Decision configuration</h3>
              <p className="text-xs text-mute mt-0.5">Serving values reported by the API</p>
            </div>
            <Badge tone="brand">threshold {meta.threshold}</Badge>
          </div>
          <div className="grid sm:grid-cols-2 gap-4">
            {[
              ['Decision rule', meta.decision_rule],
              ['Model version', `v${meta.model_version}`],
              ['Streams', 'spatial (CNN backbone) + frequency (FFT)'],
              ['Preprocessing', `resize ${meta.image_size}² · ImageNet norm · log1p|FFT|`],
              ['Serving runtime', 'ONNX Runtime (' + (health.onnxruntime_providers || []).join(', ') + ')'],
              ['Explainability', health.gradcam_available ? 'Grad-CAM via PyTorch (lazy)' : 'Grad-CAM unavailable'],
            ].map(([k, v]) => (
              <div key={k} className="rounded-xl bg-panel2 border border-line p-3.5">
                <p className="text-[10px] text-mute font-semibold uppercase tracking-wider">{k}</p>
                <p className="text-xs num font-semibold mt-1 leading-relaxed">{v}</p>
              </div>
            ))}
          </div>
          <p className="text-[11px] text-faint mt-4 num">Last synced {updated ? fmtMs(updated) : '—'} · auto-refresh every 30 s</p>
        </Card>
      </div>

      {/* Metrics of the trained checkpoint (recovered from the notebook) */}
      <Card>
        <div className="px-5 py-4 border-b border-line flex items-center justify-between flex-wrap gap-2">
          <div>
            <h3 className="font-semibold">Evaluation metrics — trained checkpoint</h3>
            <p className="text-xs text-mute mt-0.5">
              Reported by the evaluation run of <span className="num">AdaptiveFusion_best_model.pth</span> (125.1 MB,
              trained on Colab) — recovered from the training notebook outputs inside the project archive and
              shipped as <span className="num">reports/metrics.json</span>.
            </p>
          </div>
          <Badge tone={metricBlocks.length ? 'ok' : 'mute'}>{metricBlocks.length ? 'TRAINED CHECKPOINT' : 'NO METRICS FILE'}</Badge>
        </div>
        {metricBlocks.length === 0 ? (
          <EmptyState icon={P.chart} title="No evaluation metrics loaded"
            message="The serving pipeline is live; the metrics block fills in as soon as reports/metrics.json exists on the backend." />
        ) : (
          <>
            <div className="px-5 py-3.5 bg-brand/[.04] border-b border-line flex items-start gap-3 text-xs text-mute leading-relaxed">
              <Icon d={P.info} className="h-4 w-4 text-brand shrink-0 mt-0.5" />
              <p>
                These are the metrics of the <b className="text-ink">trained</b> checkpoint as reported by its
                evaluation notebook. The archive never contained the weights themselves
                (<span className="num">*.pth</span> / <span className="num">*.onnx</span> are git-ignored and
                DVC/Kaggle-hosted), so this package serves CI dummy weights with the identical contract —
                drop the real <span className="num">.pth</span> + <span className="num">.onnx</span> into
                <span className="num"> models/</span> and restart to activate them.
              </p>
            </div>
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4 p-5">
              {metricBlocks.map(([label, block]) => (
                <div key={label} className="rounded-xl border border-line bg-panel2 p-4">
                  <p className="text-xs font-semibold text-mute uppercase tracking-wider mb-3">{label}</p>
                  <div className="space-y-2">
                    {Object.entries(block).map(([k, v]) => (
                      <div key={k} className="flex items-center justify-between text-sm">
                        <span className="text-mute">{k.replace(/_/g, ' ')}</span>
                        <span className={`num font-bold ${MetricTone({ k })}`}>{fmtMetric(v)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </Card>

      {/* Fusion weights of the trained checkpoint */}
      {metrics.fusion_weights && typeof metrics.fusion_weights === 'object' && (
        <Card className="p-5">
          <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
            <h3 className="font-semibold">Adaptive fusion gate — test-split statistics</h3>
            <Badge tone="violet">INPUT-DEPENDENT</Badge>
          </div>
          <div className="grid sm:grid-cols-2 gap-4">
            {['fake', 'real'].map(cls => {
              const w = metrics.fusion_weights[cls]
              if (!w) return null
              const sp = w.spatial_mean ?? w.spatial ?? 0.5
              return (
                <div key={cls} className="rounded-xl border border-line bg-panel2 p-4">
                  <p className="text-xs font-semibold text-mute uppercase tracking-wider mb-3">
                    {cls === 'fake' ? 'Fake images' : 'Real images'}
                  </p>
                  <div className="h-2.5 rounded-full overflow-hidden flex bg-line mb-2">
                    <div className="bg-brand h-full" style={{ width: `${sp * 100}%` }} />
                    <div className="bg-violet h-full" style={{ width: `${(1 - sp) * 100}%` }} />
                  </div>
                  <div className="flex justify-between text-xs text-mute">
                    <span className="num">spatial <b className="text-ink">{sp.toFixed(4)}</b></span>
                    <span className="num">frequency <b className="text-ink">{(w.frequency_mean ?? 1 - sp).toFixed(4)}</b></span>
                  </div>
                </div>
              )
            })}
          </div>
          <p className="text-[11px] text-faint mt-3 leading-relaxed">
            {metrics.fusion_weights.note || 'Per-class gate statistics on the test split.'} With the trained
            checkpoint the gate routes fakes through the spatial stream and reals through the frequency stream.
          </p>
        </Card>
      )}

      {/* Training & evaluation artifacts (figures recovered from the notebook) */}
      <Card>
        <div className="px-5 py-4 border-b border-line flex items-center justify-between flex-wrap gap-2">
          <div>
            <h3 className="font-semibold">Training &amp; evaluation artifacts</h3>
            <p className="text-xs text-mute mt-0.5">
              Figures produced by the trained model's evaluation — extracted from the notebook outputs in the
              archive (<span className="num">media/figures/</span>).
            </p>
          </div>
          <Badge tone="brand">5 FIGURES</Badge>
        </div>
        <div className="grid sm:grid-cols-2 gap-4 p-5">
          {[
            ['architecture_diagram.png', 'Architecture — dual-stream AdaptiveFusion',
              'Spatial EfficientNetV2-RW-S + frequency CNN joined by the learned fusion gate (31,938,770 parameters).'],
            ['test_evaluation.png', 'Test evaluation (10,000 images)',
              'Confusion matrix, ROC curve and score distributions — Acc 99.79%, AUC 1.000, EER 0.0021.'],
            ['fusion_weights_by_class.png', 'Fusion weights by class',
              'Gate statistics per class — 4,906 distinct gate values on the test split.'],
            ['gradcam_test_examples.png', 'Grad-CAM on test images',
              'Explanations on 2 real + 2 fake test images (the same method as the workbench overlay).'],
            ['qualitative_test_examples.png', 'Qualitative results',
              'Model predictions with probabilities and fusion weights on the test split.'],
          ].map(([file, title, caption]) => (
            <a key={file} href={`/media/figures/${file}`} target="_blank" rel="noreferrer"
              className="rounded-xl border border-line bg-panel2 overflow-hidden group hover:border-brand/40 transition">
              <img src={`/media/figures/${file}`} alt={title} loading="lazy"
                className="w-full max-h-64 object-contain bg-bg group-hover:scale-[1.01] transition" />
              <div className="p-3.5">
                <p className="text-sm font-semibold">{title}</p>
                <p className="text-[11px] text-mute mt-1 leading-relaxed">{caption}</p>
              </div>
            </a>
          ))}
        </div>
      </Card>

      {/* Robustness */}
      {Object.keys(robustness).length > 0 && (
        <Card>
          <div className="px-5 py-4 border-b border-line">
            <h3 className="font-semibold">Robustness (corruptions)</h3>
            <p className="text-xs text-mute mt-0.5">Accuracy under corruptions, from metrics.json</p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[420px]">
              <thead><tr>
                <th className="th">Corruption</th><th className="th">Accuracy</th><th className="th">Drop</th>
              </tr></thead>
              <tbody>
                {Object.entries(robustness).map(([k, v]) => (
                  <tr key={k}>
                    <td className="td font-medium">{k}</td>
                    <td className="td num font-bold text-ok">{v.accuracy != null ? v.accuracy.toFixed(4) : '—'}</td>
                    <td className="td num text-warn">{v.drop != null ? v.drop.toFixed(4) : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Limitations */}
      <Card>
        <div className="px-5 py-4 border-b border-line flex items-center justify-between">
          <div>
            <h3 className="font-semibold">Model limitations</h3>
            <p className="text-xs text-mute mt-0.5">Shipped with the model card — read before operational use</p>
          </div>
          <Icon d={P.alert} className="h-4 w-4 text-warn" />
        </div>
        <ul className="p-5 space-y-3">
          {limitations.map(l => (
            <li key={l} className="flex gap-3 text-sm text-mute leading-relaxed">
              <span className="h-1.5 w-1.5 rounded-full bg-warn shrink-0 mt-2" />{l}
            </li>
          ))}
        </ul>
      </Card>
    </div>
  )
}
