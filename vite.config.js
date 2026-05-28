import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    // By explicitly passing 'chrome', Vite bypasses VS Code's default handlers
    // and forces your operating system to launch the external Chrome window.
    open: 'chrome' 
  }
});