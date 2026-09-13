// The app is a thin client over a local API (webapp/api) that is proxied at /api, so
// there is nothing useful to server-render: data arrives per interaction, and fetching
// it during SSR would need an absolute backend URL for no benefit.
export const ssr = false;
