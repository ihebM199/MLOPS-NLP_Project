import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App.jsx'
import './styles.css'
import './dg5.css'   /* the approved landing/auth design (scoped under .dg5) */
import './daf.css'   /* the original repo output-page classes (scoped under .daf) */
import { ThemeProvider } from './context/ThemeContext.jsx'
import { ToastProvider } from './context/ToastContext.jsx'
import { AuthProvider } from './context/AuthContext.jsx'
import { NotificationsProvider } from './context/NotificationsContext.jsx'
import { useToast } from './context/ToastContext.jsx'

function Root() {
  const pushToast = useToast()
  return (
    <NotificationsProvider pushToast={pushToast}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </NotificationsProvider>
  )
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ThemeProvider>
      <ToastProvider>
        <AuthProvider>
          <Root />
        </AuthProvider>
      </ToastProvider>
    </ThemeProvider>
  </React.StrictMode>,
)
