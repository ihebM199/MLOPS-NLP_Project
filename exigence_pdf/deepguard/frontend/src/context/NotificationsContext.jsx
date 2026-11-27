import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react'
import { getDB, pushNotif, subscribeStore } from '../api/client.js'
import { useAuth } from './AuthContext.jsx'

const NotificationsContext = createContext(null)

/** Notification engine — store-backed (localStorage), same behaviour as the
 *  platform prototype: unread count + latest feed for the signed-in user,
 *  a toast for events that arrive while the session is open, and instant
 *  re-sync after mutations so the badge never lags. */
export function NotificationsProvider({ children, pushToast }) {
  const { user } = useAuth()
  const [unread, setUnread] = useState(0)
  const [latest, setLatest] = useState([])
  const seen = useRef(new Set())
  const [booted, setBooted] = useState(false)

  const read = useCallback(() => {
    if (!user) { setUnread(0); setLatest([]); return }
    const mine = getDB().notifs.filter(n => n.userId === user.id)
    for (const n of mine) {
      if (!booted && !seen.current.size) {
        seen.current.add(n.id) // first sync after login: just seed, no toast spam
      } else if (!seen.current.has(n.id)) {
        seen.current.add(n.id)
        const tone = n.severity === 'critical' ? 'bad' : n.severity === 'warning' ? 'warn'
          : n.severity === 'success' ? 'ok' : 'brand'
        pushToast?.(n.title, n.message, tone)
      }
    }
    setUnread(mine.filter(n => !n.read).length)
    setLatest(mine.slice(0, 30))
    setBooted(true)
  }, [user, booted, pushToast])

  useEffect(() => { read() }, [read])
  useEffect(() => subscribeStore(read), [read])

  useEffect(() => {
    if (!user) { seen.current = new Set(); setBooted(false); setUnread(0); setLatest([]); return }
    seen.current = new Set()
    setBooted(false)
  }, [user?.id])

  const markRead = useCallback((id) => {
    const db = getDB()
    const n = db.notifs.find(x => x.id === id)
    if (n) { n.read = true }
    seen.current.add(id)
    setUnread(db.notifs.filter(x => x.userId === user.id && !x.read).length)
  }, [user])

  const markAllRead = useCallback(() => {
    const db = getDB()
    db.notifs.forEach(n => { if (n.userId === user.id) n.read = true })
    setUnread(0)
    read()
  }, [user, read])

  const clearRead = useCallback(() => {
    const db = getDB()
    db.notifs = db.notifs.filter(n => !(n.userId === user.id && n.read))
    read()
  }, [user, read])

  const notify = useCallback((type, severity, title, message, link) => {
    pushNotif(user.id, type, severity, title, message, link)
  }, [user])

  return (
    <NotificationsContext.Provider value={{ unread, latest, sync: read, markRead, markAllRead, clearRead, notify }}>
      {children}
    </NotificationsContext.Provider>
  )
}

export const useNotifications = () => useContext(NotificationsContext)
