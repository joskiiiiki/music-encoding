<script lang="ts">
	import Play from '@lucide/svelte/icons/play';
	import Pause from '@lucide/svelte/icons/pause';
	import Loader from '@lucide/svelte/icons/loader';
	import X from '@lucide/svelte/icons/x';
	import Footprints from '@lucide/svelte/icons/footprints';
	import SkipForward from '@lucide/svelte/icons/skip-forward';
	import { formatDuration } from '$lib/api';
	import { player } from '$lib/playerStore.svelte.js';
	import AudioBadge from './AudioBadge.svelte';
	import { Button } from '$lib/components/ui/button/index.js';
	import { Checkbox } from '$lib/components/ui/checkbox/index.js';
	import { ScrollArea } from '$lib/components/ui/scroll-area/index.js';
	import { Slider } from '$lib/components/ui/slider/index.js';

	const total = $derived(player.duration || player.track?.duration || 0);
	const currentStep = $derived(player.trail.length - 1);
</script>

{#if player.track}
	<div class="bg-background/95 fixed inset-x-0 bottom-0 z-40 border-t backdrop-blur">
		{#if player.walkEnabled}
			<div class="mx-auto max-w-[1400px] px-4 pt-2">
				<div class="flex items-center gap-3">
					<span class="text-muted-foreground shrink-0 text-[10px] tracking-wide uppercase">
						walk
					</span>
					<div class="flex shrink-0 items-center gap-1.5">
						<Checkbox id="walk-new-artist" bind:checked={player.avoidSameArtist} />
						<label for="walk-new-artist" class="cursor-pointer text-[10px]">new artist</label>
					</div>

					<!-- The chain, oldest first. Each chip is the hop that led to that track, so
					     the numbers show how similarity decays as the walk drifts; clicking one
					     rewinds the walk to that point. -->
					<ScrollArea class="min-w-0 flex-1" orientation="horizontal">
						<div class="flex items-center gap-1 pr-3">
							{#each player.trail as step, index (step.idx)}
								{#if index > 0}
									<span class="text-muted-foreground shrink-0 text-[10px]">→</span>
								{/if}
								<button
									type="button"
									class="hover:bg-accent flex shrink-0 items-center gap-1 rounded px-1.5 py-0.5 text-[10px] {index ===
									currentStep
										? 'bg-accent text-accent-foreground'
										: 'text-muted-foreground'}"
									title={step.artist}
									onclick={() => player.jumpTo(index)}
								>
									<span class="max-w-[130px] truncate">{step.title}</span>
									{#if step.cos !== null}
										<span class="tabular-nums opacity-60">{step.cos.toFixed(2)}</span>
									{/if}
								</button>
							{/each}
							{#if player.nextUp}
								<span class="text-muted-foreground shrink-0 text-[10px]">→</span>
								<span class="text-muted-foreground shrink-0 px-1.5 text-[10px] italic" title="lined up next">
									{player.nextUp.title.slice(0, 26)}
									{#if player.nextUp.score !== null}
										<span class="tabular-nums">{player.nextUp.score.toFixed(2)}</span>
									{/if}
								</span>
							{/if}
						</div>
					</ScrollArea>
				</div>
			</div>
		{/if}

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
				<!-- `onValueCommit`, deliberately NOT `onValueChange`. bits-ui fires
				     onValueChange from its internal value setter, and this slider's value is
				     driven by the playback position -- so position updates, duration
				     revisions on a VBR file, or the slider's own clamping could all write the
				     value back and call seek(), which disturbs the audio pipeline and makes
				     playback stutter. Seeking on release only breaks that loop; the thumb
				     still tracks the drag because bits-ui updates its own state meanwhile. -->
				<Slider
					type="single"
					min={0}
					max={Math.max(total, 1)}
					step={0.1}
					value={player.currentTime}
					onValueCommit={(value) => player.seek(value)}
					aria-label="Seek"
				/>
			</div>
			<span class="text-muted-foreground shrink-0 text-xs tabular-nums">
				{formatDuration(player.currentTime)} / {formatDuration(total)}
			</span>

			<Button
				variant={player.walkEnabled ? 'default' : 'ghost'}
				size="xs"
				class="shrink-0"
				onclick={() => player.toggleWalk()}
				title="Follow each track with its nearest neighbour, endlessly"
			>
				<Footprints />
				walk
			</Button>

			{#if player.walkEnabled}
				<Button
					variant="ghost"
					size="icon-xs"
					class="shrink-0"
					onclick={() => player.stepNow()}
					disabled={!player.nextUp}
					title="Step to the next-nearest track now"
					aria-label="Next in walk"
				>
					<SkipForward />
				</Button>
			{/if}

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
