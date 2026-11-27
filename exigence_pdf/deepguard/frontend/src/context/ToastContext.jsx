import { createContext, useCallback, useContext, useRef, useState } from 'react'

const ToastContext = createContext(null)
let idc = 0

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([])
  const timers = useRef({})

  const dismiss = useCallback((id) => {
    setToasts(t => t.filter(x => x.id !== id))
    clearTimeout(timers.current[id])
    delete timers.current[id]
  }, [])

  const push = useCallback((title, message = '', tone = 'brand', ms = 4200) => {
    const id = ++idc
    setToasts(t => [...t.slice(-3), { id, title, message, tone }])
    timers.current[id] = setTimeout(() => dismiss(id), ms)
  }, [dismiss])

  return (
    <ToastContext.Provider value={push}>
      {children}
      <div className="fixed z-[90] bottom-4 right-4 left-4 sm:left-auto flex flex-col gap-2 sm:w-96 pointer-events-none">
        {toasts.map(t => (
          <div key={t.id} className="card pointer-events-auto glass flex items-start gap-3 p-3.5 animate-rise pop">
            <span className={`mt-0.5 h-2 w-2 shrink-0 rounded-full ${
              t.tone === 'ok' ? 'bg-ok' : t.tone === 'bad' ? 'bg-bad' : t.tone === 'warn' ? 'bg-warn' : 'bg-brand'}`} />
            <div className="min-w-0 flex-1">
              <p className="text-sm font-semibold leading-tight">{t.title}</p>
              {t.message && <p className="mt-0.5 text-xs text-mute leading-snug line-clamp-2">{t.message}</p>}
            </div>
            <button onClick={() => dismiss(t.id)} aria-label="Dismiss"
              className="text-faint hover:text-ink shrink-0 transition text-sm leading-none">✕</button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}

export const useToast = () => useContext(ToastContext)
