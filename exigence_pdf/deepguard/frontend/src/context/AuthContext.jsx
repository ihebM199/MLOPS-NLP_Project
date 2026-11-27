import { createContext, useCallback, useContext, useState } from 'react'
import { getDB, loginStore, logoutStore, me, signupStore, subscribeStore, updateUser } from '../api/client.js'

const AuthContext = createContext(null)

/** LocalStorage-backed session (same accounts as the approved prototype:
 *  admin@deepguard.io / Admin@2026, demo@deepguard.io / Demo@2026). */
export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => me())

  // Keep every provider in sync when pages mutate the store (settings, users…).
  subscribeStore(() => setUser(me()))

  const login = useCallback(async (email, password) => {
    const u = loginStore(email, password)
    setUser(u)
    return u
  }, [])

  const signup = useCallback(async (payload) => {
    const u = signupStore(payload)
    setUser(u)
    return u
  }, [])

  const logout = useCallback(async () => {
    logoutStore()
    setUser(null)
  }, [])

  const refreshUser = useCallback(async () => {
    const u = me()
    setUser(u)
    return u
  }, [])

  const patchUser = useCallback((patch) => {
    const u = updateUser(patch)
    setUser({ ...u })
    return u
  }, [])

  return (
    <AuthContext.Provider value={{ user, getDB, login, signup, logout, refreshUser, patchUser, setUser }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)
