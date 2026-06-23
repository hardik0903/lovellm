import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
 
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/chat':      'http://localhost:8000',
      '/upload':    'http://localhost:8000',
      '/health':    'http://localhost:8000',
      '/telemetry': 'http://localhost:8000',
    }
  }
})
 