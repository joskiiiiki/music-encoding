<script lang="ts">
	import Play from '@lucide/svelte/icons/play';
	import Pause from '@lucide/svelte/icons/pause';
	import Loader from '@lucide/svelte/icons/loader';
	import X from '@lucide/svelte/icons/x';
	import { formatDuration } from '$lib/api';
	import { player } from '$lib/playerStore.svelte.js';
	import AudioBadge from './AudioBadge.svelte';
	import { Button } from '$lib/components/ui/button/index.js';
	import { Slider } from '$lib/components/ui/slider/index.js';

	const total = $derived(player.duration || player.track?.duration || 0);
</script>

{#if player.track}
	<div class="bg-background/95 fixed inset-x-0 bottom-0 z-40 border-t backdrop-blur">
		<div class="mx-auto flex max-w-[1400px] items-center gap-3 px-4 py-2.5">
			<Button
				variant="outline"
				size="icon"
				class="shrink-0 rounded-full"
				onclick={() => player.toggle()}
				aria-label={player.playing ? 'Pause' : 'Play'}
			>
				{#if player.loading}
					<Loader class="animate-spin" />
				{:else if player.playing}
					<Pause />
				{:else}
					<Play />
				{/if}
			</Button>

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
						<span class="truncate">· preview — a different recording</span>
						<Button
							variant="link"
							size="xs"
							class="h-auto p-0 text-xs"
							onclick={() => player.forgetMatch()}
							title="Forget this match and search again"
						>
							not it?
						</Button>
					{/if}
				</div>
			</div>

			<div class="hidden w-48 sm:block">
				<!-- Driven one-way from playback position; dragging seeks live, so the thumb
				     tracks the drag without a second source of truth to keep in sync. -->
				<Slider
					type="single"
					min={0}
					max={Math.max(total, 1)}
					step={0.1}
					value={player.currentTime}
					onValueChange={(value) => player.seek(value)}
					aria-label="Seek"
				/>
			</div>
			<span class="text-muted-foreground shrink-0 text-xs tabular-nums">
				{formatDuration(player.currentTime)} / {formatDuration(total)}
			</span>

			<Button
				variant="ghost"
				size="icon-xs"
				class="text-muted-foreground shrink-0"
				onclick={() => player.stop()}
				aria-label="Close player"
			>
				<X />
			</Button>
		</div>
		{#if player.error}
			<p class="text-destructive mx-auto max-w-[1400px] px-4 pb-2 text-xs">{player.error}</p>
		{/if}
	</div>
{/if}
