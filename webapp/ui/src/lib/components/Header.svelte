<script lang="ts">
	import { Badge } from '$lib/components/ui/badge/index.js';
	import { Button } from '$lib/components/ui/button/index.js';

	let { pathname = '/' }: { pathname?: string } = $props();

	const onBrowse = $derived(pathname === '/');
	const onTrack = $derived(pathname.startsWith('/track/'));
</script>

<header class="border-b">
	<div class="mx-auto flex max-w-[1400px] flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
		<!-- The brand mark stays a plain anchor: it is a wordmark, not a nav control. -->
		<a href="/" class="flex items-center gap-2 font-semibold tracking-tight">
			<span class="bg-primary text-primary-foreground rounded px-1.5 py-0.5 text-xs">BT</span>
			MTG-Jamendo similarity explorer
		</a>
		<nav class="flex items-center gap-1" aria-label="Sections">
			<Button href="/" variant={onBrowse ? 'secondary' : 'ghost'} size="sm">Browse</Button>
			{#if onTrack}
				<!-- The current section has no link target of its own (a track page needs an
				     id), so it is marked rather than offered as a control. -->
				<Badge variant="secondary" class="h-7 px-2.5">Track</Badge>
			{/if}
		</nav>
		<p class="text-muted-foreground ml-auto hidden text-xs sm:block">
			32,783 tracks · 128-d Barlow Twins embeddings · exact cosine
		</p>
	</div>
</header>
