import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// `VITE_API_URL` empty => same origin (FastAPI serves the built SPA on Colab).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_DEV_API_TARGET || 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    chunkSizeWarningLimit: 900,
  },
  test: {
    environment: 'jsdom',
    setupFiles: './src/test/setup.js',
    css: false,
    include: ['src/**/*.{test,spec}.{js,jsx}'],
  },
})
