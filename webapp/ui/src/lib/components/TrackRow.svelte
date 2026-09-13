<script lang="ts">
	import { formatDuration, type Track } from '$lib/api';
	import { player } from '$lib/playerStore.svelte.js';
	import AudioBadge from './AudioBadge.svelte';

	let {
		track,
		rank = null,
		compact = false
	}: { track: Track; rank?: number | null; compact?: boolean } = $props();

	const isCurrent = $derived(player.track?.idx === track.idx);
	const isPlaying = $derived(isCurrent && player.playing);
	// Cosine to whatever this list is ranked against. Shown because "similar" is the
	// whole claim of the app, and a number is harder to over-read than a thumbnail.
	const score = $derived(track.score);
</script>

<div
	class="hover:bg-muted/60 group flex items-center gap-3 border-b px-2 py-2 transition-colors last:border-b-0 {isCurrent
		? 'bg-muted/80'
		: ''}"
>
	{#if rank !== null}
		<span class="text-muted-foreground w-6 shrink-0 text-right text-xs tabular-nums">{rank}</span>
	{/if}

	<button
		type="button"
		class="border-border hover:bg-accent flex size-8 shrink-0 items-center justify-center rounded-full border text-xs disabled:opacity-40"
		disabled={track.audio === 'none'}
		title={track.audio === 'none' ? 'No audio available for this track' : 'Play'}
		aria-label={isPlaying ? 'Pause' : `Play ${track.title}`}
		onclick={() => player.toggle(track)}
	>
		{#if isPlaying}
			<svg viewBox="0 0 16 16" class="size-3 fill-current"><rect x="3" y="2" width="3.5" height="12" /><rect x="9.5" y="2" width="3.5" height="12" /></svg>
		{:else}
			<svg viewBox="0 0 16 16" class="size-3 fill-current"><path d="M4 2.5v11l9-5.5z" /></svg>
		{/if}
	</button>

	<div class="min-w-0 flex-1">
		<div class="flex items-baseline gap-2">
			<a
				href="/track/{track.idx}"
				class="hover:underline truncate text-sm font-medium"
				title={track.title}
			>
				{track.title}
			</a>
			{#if score !== null && score !== undefined}
				<span class="text-muted-foreground shrink-0 text-xs tabular-nums">
					{score.toFixed(3)}
				</span>
			{/if}
		</div>
		<div class="text-muted-foreground flex items-center gap-2 truncate text-xs">
			<span class="truncate">{track.artist || 'unknown artist'}</span>
			{#if track.released}<span>· {track.released}</span>{/if}
			{#if !compact && track.album}<span class="truncate">· {track.album}</span>{/if}
			{#if track.duration}<span class="tabular-nums">· {formatDuration(track.duration)}</span>{/if}
		</div>
		{#if !compact && track.tags.length}
			<div class="mt-1 flex flex-wrap gap-1">
				{#each track.tags.slice(0, 4) as tag (tag)}
					<span class="bg-secondary text-secondary-foreground rounded px-1.5 py-px text-[10px]">
						{tag}
					</span>
				{/each}
			</div>
		{/if}
	</div>

	<AudioBadge kind={track.audio} />
</div>
