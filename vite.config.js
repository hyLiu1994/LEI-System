import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5000',  // 假设你的 Flask 后端运行在 5000 端口
        changeOrigin: true,
        secure: false,
      }
    }
  },
  resolve: {
    alias: {
      './images/layers.png': './node_modules/leaflet/dist/images/layers.png',
      './images/layers-2x.png': './node_modules/leaflet/dist/images/layers-2x.png',
      './images/marker-icon.png': './node_modules/leaflet/dist/images/marker-icon.png',
    },
  },
})