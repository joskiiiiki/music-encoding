<script lang="ts">
	import Play from '@lucide/svelte/icons/play';
	import Pause from '@lucide/svelte/icons/pause';
	import Loader from '@lucide/svelte/icons/loader';
	import X from '@lucide/svelte/icons/x';
	import Footprints from '@lucide/svelte/icons/footprints';
	import SkipForward from '@lucide/svelte/icons/skip-forward';
	import ChevronDown from '@lucide/svelte/icons/chevron-down';
	import ChevronUp from '@lucide/svelte/icons/chevron-up';
	import ChevronLeft from '@lucide/svelte/icons/chevron-left';
	import ChevronRight from '@lucide/svelte/icons/chevron-right';
	import { formatDuration } from '$lib/api';
	import { player } from '$lib/playerStore.svelte.js';
	import AudioBadge from './AudioBadge.svelte';
	import { Button } from '$lib/components/ui/button/index.js';
	import { Checkbox } from '$lib/components/ui/checkbox/index.js';
	import { Slider } from '$lib/components/ui/slider/index.js';

	const total = $derived(player.duration || player.track?.duration || 0);
	const currentStep = $derived(player.trail.length - 1);

	// Where the thumb should sit. The slider is otherwise driven straight from
	// `player.currentTime`, which only advances on timeupdate (~4x/s) -- so on a click the
	// thumb jumped to the click point, the next re-render yanked it back to the stale
	// playback position until the seek landed, and it read as "snaps back". Holding the
	// scrubbed value until playback catches up fixes that, and it has to be released or
	// the next track inherits it.
	let scrub = $state<number | null>(null);
	const position = $derived(scrub ?? player.currentTime);

	// A new track starts at 0: never carry a scrub across it.
	$effect(() => {
		void player.track?.idx;
		scrub = null;
	});

	$effect(() => {
		if (scrub === null) return;
		const target = scrub;
		// Seek landed: playback has reached where the user put the thumb.
		if (Math.abs(player.currentTime - target) < 0.5) {
			scrub = null;
			return;
		}
		// ...but never let it stick if the seek is refused (or the file is shorter than
		// the requested position), which would freeze the position display.
		const timer = setTimeout(() => {
			if (scrub === target) scrub = null;
		}, 1500);
		return () => clearTimeout(timer);
	});

	let trailCollapsed = $state(false);

	// Trail scrolling, driven by the arrow buttons rather than a scrollbar.
	let trailEl = $state<HTMLDivElement | null>(null);
	let canScrollBack = $state(false);
	let canScrollForward = $state(false);

	function nudgeTrail(direction: -1 | 1) {
		if (!trailEl) return;
		// Most of a viewport per press, so a long chain takes a few clicks rather than dozens.
		const step = Math.max(140, trailEl.clientWidth * 0.7);
		trailEl.scrollBy({ left: direction * step, behavior: 'smooth' });
	}

	/** Recompute the arrows, and optionally follow the walk to the newest step. */
	function syncTrail(el: HTMLElement, follow: boolean) {
		if (follow && el.scrollWidth > el.clientWidth) {
			// Only follow if the chain is already at the end, so a new step cannot yank the
			// view away from an earlier part of the trail being read.
			const pinned = el.scrollLeft + el.clientWidth >= el.scrollWidth - 8;
			if (pinned) el.scrollLeft = el.scrollWidth;
		}
		canScrollBack = el.scrollLeft > 1;
		canScrollForward = el.scrollLeft + el.clientWidth < el.scrollWidth - 1;
	}

	// Drive the arrows from the geometry itself. A Svelte *action* rather than an effect:
	// actions are guaranteed to run when their element is created, whereas an effect keyed
	// on the trail's length did not re-run as the chain grew (and neither did a
	// ResizeObserver created inside one). Observing the scroller and the chip row covers
	// both cases that matter -- the chain getting longer, and the window changing width.
	function trailSync(node: HTMLElement) {
		const inner = node.firstElementChild as HTMLElement | null;
		const observer = new ResizeObserver(() => syncTrail(node, true));
		observer.observe(node);
		if (inner) observer.observe(inner);
		syncTrail(node, true);
		return { destroy: () => observer.disconnect() };
	}
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

			<!-- Sized to its content, capped, and allowed to shrink: the slider takes the
			     remaining width, so this must not claim `flex-1` or the two split the bar. -->
			<div class="min-w-0 max-w-[300px] sm:max-w-[380px]">
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

			<div class="hidden min-w-[120px] flex-1 sm:block">
				<!-- Keyed on the track so each new song gets a fresh slider: without it the
				     instance keeps the previous track's internal state, and the position read
				     as "not reset".

				     ONLY `onValueCommit` -- deliberately no `onValueChange`. bits-ui calls
				     onValueChange from inside its internal value setter, so handling it while
				     this slider's value is *driven* by playback closes a loop: position ->
				     value prop -> bits-ui's internal write -> onValueChange -> position -> ...,
				     which spins the main thread and freezes the whole app as soon as a track
				     plays. `scrub` is therefore only written on commit, which still holds the
				     thumb where you clicked instead of snapping it back. -->
				{#key player.track.idx}
					<Slider
						type="single"
						min={0}
						max={Math.max(total, 1)}
						step={0.1}
						value={position}
						onValueCommit={(value) => {
							scrub = value;
							player.seek(value);
						}}
						aria-label="Seek"
					/>
				{/key}
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

		{#if player.walkEnabled}
			<!-- Below the transport controls, and collapsible: the trail is useful but it is
			     also a lot of small text to keep on screen the whole time. -->
			<div class="mx-auto max-w-[1400px] px-4 pb-2">
				<div class="flex items-center gap-2">
					<Button
						variant="ghost"
						size="icon-xs"
						class="text-muted-foreground shrink-0"
						onclick={() => (trailCollapsed = !trailCollapsed)}
						aria-expanded={!trailCollapsed}
						title={trailCollapsed ? 'Show the walk trail' : 'Hide the walk trail'}
					>
						{#if trailCollapsed}
							<ChevronUp />
						{:else}
							<ChevronDown />
						{/if}
					</Button>
					<span class="text-muted-foreground shrink-0 text-[10px] tracking-wide uppercase">
						walk
					</span>
					<span class="text-muted-foreground shrink-0 text-[10px] tabular-nums">
						{currentStep + 1}/{player.trail.length}
					</span>
					<div class="flex shrink-0 items-center gap-1.5">
						<Checkbox id="walk-new-artist" bind:checked={player.avoidSameArtist} />
						<label for="walk-new-artist" class="cursor-pointer text-[10px]">new artist</label>
					</div>

					{#if !trailCollapsed}
						<!-- The chain, oldest first. Each chip is the hop that led to that track, so
						     the numbers show how similarity decays as the walk drifts; clicking one
						     rewinds the walk to that point.

						     Scrolled with the two arrow buttons rather than a scrollbar: a bar is
						     drawn over the bottom of the viewport here, which forced the row taller
						     than the chips needed and still landed on the text. The scroller is
						     `overflow-x: auto` with its scrollbar hidden, so the arrows are the only
						     affordance and the row sizes to the chips. -->
						<Button
							variant="ghost"
							size="icon-xs"
							class="shrink-0"
							disabled={!canScrollBack}
							onclick={() => nudgeTrail(-1)}
							aria-label="Scroll back through the walk"
						>
							<ChevronLeft />
						</Button>
						<div
							class="min-w-0 flex-1 overflow-x-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
							bind:this={trailEl}
							use:trailSync
							onscroll={(event) => syncTrail(event.currentTarget, false)}
							data-can-back={canScrollBack}
							data-can-forward={canScrollForward}
						>
							<div class="flex w-max items-center gap-1">
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
									<span
										class="text-muted-foreground shrink-0 px-1.5 text-[10px] italic"
										title="lined up next"
									>
										{player.nextUp.title.slice(0, 26)}
										{#if player.nextUp.score !== null}
											<span class="tabular-nums">{player.nextUp.score.toFixed(2)}</span>
										{/if}
									</span>
								{/if}
							</div>
						</div>
						<Button
							variant="ghost"
							size="icon-xs"
							class="shrink-0"
							disabled={!canScrollForward}
							onclick={() => nudgeTrail(1)}
							aria-label="Scroll forward through the walk"
						>
							<ChevronRight />
						</Button>
					{/if}
				</div>
			</div>
		{/if}

		{#if player.error}
			<p class="text-destructive mx-auto max-w-[1400px] px-4 pb-2 text-xs">{player.error}</p>
		{/if}
	</div>
{/if}
