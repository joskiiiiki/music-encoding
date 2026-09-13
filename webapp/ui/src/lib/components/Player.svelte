<script lang="ts">
	import { formatDuration } from '$lib/api';
	import { player } from '$lib/playerStore.svelte.js';
	import AudioBadge from './AudioBadge.svelte';

	// Scrubbing: local state while the thumb is held, so the position does not fight
	// the audio element's `timeupdate` events mid-drag.
	let scrub = $state<number | null>(null);
	const position = $derived(scrub ?? player.currentTime);
	const total = $derived(player.duration || player.track?.duration || 0);
</script>

{#if player.track}
	<div class="bg-background/95 fixed inset-x-0 bottom-0 z-40 border-t backdrop-blur">
		<div class="mx-auto flex max-w-[1400px] items-center gap-3 px-4 py-2.5">
			<button
				type="button"
				class="border-border hover:bg-accent flex size-9 shrink-0 items-center justify-center rounded-full border"
				onclick={() => player.toggle()}
				aria-label={player.playing ? 'Pause' : 'Play'}
			>
				{#if player.loading}
					<span class="text-xs">…</span>
				{:else if player.playing}
					<svg viewBox="0 0 16 16" class="size-3.5 fill-current"
						><rect x="3" y="2" width="3.5" height="12" /><rect x="9.5" y="2" width="3.5" height="12" /></svg
					>
				{:else}
					<svg viewBox="0 0 16 16" class="size-3.5 fill-current"
						><path d="M4 2.5v11l9-5.5z" /></svg
					>
				{/if}
			</button>

			<div class="min-w-0 flex-1">
				<div class="flex items-center gap-2">
					<a href="/track/{player.track.idx}" class="hover:underline truncate text-sm font-medium">
						{player.track.title}
					</a>
					<AudioBadge kind={player.track.audio} />
				</div>
				<div class="text-muted-foreground flex items-center gap-2 text-xs">
					<span class="truncate">{player.track.artist}</span>
					{#if player.track.audio === 'preview'}
						<span class="truncate"
							>· preview matched by {player.track.artist
								? 'artist + title'
								: 'title'} — a different recording</span
						>
						<button
							type="button"
							class="hover:text-foreground underline underline-offset-2"
							onclick={() => player.forgetMatch()}
							title="Forget this match and search again"
						>
							not it?
						</button>
					{/if}
				</div>
			</div>

			<input
				type="range"
				class="accent-primary hidden h-1 w-48 cursor-pointer sm:block"
				min="0"
				max={Math.max(total, 1)}
				step="0.1"
				value={position}
				oninput={(event) => (scrub = Number(event.currentTarget.value))}
				onchange={(event) => {
					player.seek(Number(event.currentTarget.value));
					scrub = null;
				}}
				aria-label="Seek"
			/>
			<span class="text-muted-foreground shrink-0 text-xs tabular-nums">
				{formatDuration(position)} / {formatDuration(total)}
			</span>

			<button
				type="button"
				class="text-muted-foreground hover:text-foreground shrink-0 px-1 text-xs"
				onclick={() => player.stop()}
				aria-label="Close player"
			>
				✕
			</button>
		</div>
		{#if player.error}
			<p class="text-destructive mx-auto max-w-[1400px] px-4 pb-2 text-xs">{player.error}</p>
		{/if}
	</div>
{/if}
