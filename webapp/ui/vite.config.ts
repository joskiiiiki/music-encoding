import adapter from '@sveltejs/adapter-auto';
import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [
		tailwindcss(),
		sveltekit({
			compilerOptions: {
				// Force runes mode for the project, except for libraries. Can be removed in svelte 6.
				runes: ({ filename }) =>
					filename.split(/[/\\]/).includes('node_modules') ? undefined : true
			},

			// adapter-auto only supports some environments, see https://svelte.dev/docs/kit/adapter-auto for a list.
			// If your environment is not supported, or you settled on a specific environment, switch out the adapter.
			// See https://svelte.dev/docs/kit/adapters for more information about adapters.
			adapter: adapter()
		})
	],
	server: {
		proxy: {
			// The backend is a separate process (FastAPI, see webapp/api). Proxying /api
			// keeps the browser on a single origin, which avoids CORS entirely and lets
			// the <audio> element issue Range requests against the same host it loaded
			// the page from.
			'/api': {
				target: process.env.WEBAPP_API ?? 'http://127.0.0.1:8000',
				changeOrigin: true
			}
		}
	}
});
