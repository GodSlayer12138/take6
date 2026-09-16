import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: { host: '127.0.0.1', proxy: { '/api/ai': `http://127.0.0.1:${process.env.NTW_API_PORT || 8765}` } },
  preview: { host: '127.0.0.1', proxy: { '/api/ai': `http://127.0.0.1:${process.env.NTW_API_PORT || 8765}` } },
  worker: { format: 'es' },
})
