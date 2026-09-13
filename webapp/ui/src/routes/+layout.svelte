<script lang="ts">
	import '../app.css';
	import favicon from '$lib/assets/favicon.svg';
	import { page } from '$app/state';
	import Header from '$lib/components/Header.svelte';
	import Player from '$lib/components/Player.svelte';
	import * as Tooltip from '$lib/components/ui/tooltip/index.js';

	let { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
	<title>MTG-Jamendo similarity explorer</title>
</svelte:head>

<!-- One tooltip provider for the app: bits-ui requires it above any Tooltip.Root. -->
<Tooltip.Provider>
	<div class="bg-background text-foreground min-h-dvh pb-20">
		<Header pathname={page.url.pathname} />
		<main class="mx-auto w-full max-w-[1400px] px-4 pb-16">
			{@render children()}
		</main>
		<!-- Fixed bottom bar: audio keeps playing while you keep browsing. -->
		<Player />
	</div>
</Tooltip.Provider>
