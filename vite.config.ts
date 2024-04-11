import reactRefresh from '@vitejs/plugin-react-refresh';
import * as path from 'path';
import { defineConfig } from 'vite';

const projectRootDir = path.resolve(__dirname);

// https://vitejs.dev/config/
export default defineConfig({
  build: {
    target: 'es2020',
  },
  optimizeDeps: {
    esbuildOptions: {
      target: 'es2020',
    },
  },
  plugins: [reactRefresh()],
});
