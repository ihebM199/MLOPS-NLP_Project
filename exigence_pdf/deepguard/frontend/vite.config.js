import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev server proxies the model API + sample media to the Django backend
// (terminal 1: python manage.py runserver, port 8000); in production
// `npm run build` emits dist/, which the Django backend serves directly
// at http://127.0.0.1:8000 (same single-URL behaviour as the original
// FastAPI deployment on Colab).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/predict': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
      '/model': 'http://127.0.0.1:8000',
      '/media': 'http://127.0.0.1:8000',
    },
  },
  build: {
    outDir: 'dist',
    chunkSizeWarningLimit: 900,
  },
})
