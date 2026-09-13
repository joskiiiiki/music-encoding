<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import {
		api,
		formatDuration,
		type GraphResponse,
		type Space,
		type TrackDetail
	} from '$lib/api';
	import AudioBadge from '$lib/components/AudioBadge.svelte';
	import EnrichmentPanel from '$lib/components/EnrichmentPanel.svelte';
	import SimilarityGraph from '$lib/components/SimilarityGraph.svelte';
	import TrackRow from '$lib/components/TrackRow.svelte';
	import { Alert, AlertDescription } from '$lib/components/ui/alert/index.js';
	import { Badge } from '$lib/components/ui/badge/index.js';
	import { Button } from '$lib/components/ui/button/index.js';
	import { Card, CardContent } from '$lib/components/ui/card/index.js';
	import { ScrollArea } from '$lib/components/ui/scroll-area/index.js';
	import * as Select from '$lib/components/ui/select/index.js';
	import { Skeleton } from '$lib/components/ui/skeleton/index.js';
	import { Slider } from '$lib/components/ui/slider/index.js';
	import { player } from '$lib/playerStore.svelte.js';

	const idx = $derived(Number(page.params.idx));

	// View parameters live in the URL so a particular neighbourhood is reloadable and
	// shareable -- the point of an exploration tool is that you can return to what you
	// found, or send it to someone.
	let k = $state(Number(page.url.searchParams.get('k') ?? 20));
	let minSim = $state(Number(page.url.searchParams.get('min_sim') ?? 0.55));
	let colorBy = $state<'genre' | 'artist'>(
		(page.url.searchParams.get('color') as 'genre' | 'artist') ?? 'genre'
	);
	let space = $state<Space>((page.url.searchParams.get('space') as Space) ?? 'whitened');

	let track = $state<TrackDetail | null>(null);
	let graph = $state<GraphResponse | null>(null);
	let loading = $state(true);
	let error = $state<string | null>(null);
	let focus = $state(false);
	let sequence = 0;

	async function load() {
		const mine = ++sequence;
		loading = true;
		error = null;
		try {
			// The graph carries the neighbour list too, so the list and the picture can
			// never disagree about which tracks are "similar" here.
			const [detail, graphData] = await Promise.all([
				api.track(idx),
				api.graph(idx, k, space, 0)
			]);
			if (mine !== sequence) return;
			track = detail;
			graph = graphData;
			syncUrl();
		} catch (cause) {
			if (mine === sequence) {
				graph = null;
				error = cause instanceof Error ? cause.message : String(cause);
			}
		} finally {
			if (mine === sequence) loading = false;
		}
	}

	function syncUrl() {
		const params = new URLSearchParams();
		if (k !== 20) params.set('k', String(k));
		if (minSim !== 0.55) params.set('min_sim', String(minSim));
		if (colorBy !== 'genre') params.set('color', colorBy);
		if (space !== 'whitened') params.set('space', space);
		const query = params.toString();
		void goto(query ? `/track/${idx}?${query}` : `/track/${idx}`, {
			replaceState: true,
			noScroll: true
		});
	}

	function reseed(next: number) {
		const params = page.url.searchParams.toString();
		void goto(params ? `/track/${next}?${params}` : `/track/${next}`, { noScroll: false });
	}

	$effect(() => {
		void [idx, k, space];
		void load();
	});

	$effect(() => {
		void [k, minSim, colorBy, space];
		if (track) syncUrl();
	});

	// Neighbours as a ranked list, seed excluded -- the graph's node set minus itself.
	const neighbours = $derived(
		graph ? graph.nodes.filter((node) => !node.seed).sort((a, b) => (b.score ?? 0) - (a.score ?? 0)) : []
	);
</script>

<div class="pt-4">
	{#if error}
		<Alert variant="destructive">
			<AlertDescription>{error}</AlertDescription>
		</Alert>
	{:else if loading && !track}
		<div class="space-y-3">
			<Skeleton class="h-7 w-64" />
			<Skeleton class="h-4 w-96" />
			<Skeleton class="h-[520px] w-full rounded-lg" />
		</div>
	{:else if track}
		<div class="flex flex-wrap items-start gap-x-6 gap-y-3">
			<div class="min-w-0 flex-1">
				<div class="flex flex-wrap items-center gap-2">
					<h1 class="text-xl font-semibold tracking-tight">{track.title}</h1>
					<AudioBadge kind={track.audio} />
				</div>
				<p class="text-muted-foreground text-sm">
					{track.artist || 'unknown artist'}
					{#if track.album}· {track.album}{/if}
					{#if track.released}· {track.released}{/if}
					{#if track.duration}· {formatDuration(track.duration)}{/if}
					· <span class="font-mono text-[11px]">{track.track_id}</span>
				</p>
				<div class="mt-2 flex flex-wrap items-center gap-1">
					{#each track.tags as tag (tag)}
						<Badge
							variant="secondary"
							href="/?genre={encodeURIComponent(tag)}"
							class="h-4 rounded px-1.5 text-[10px] font-normal"
							title="Show all {tag} tracks"
						>
							{tag}
						</Badge>
					{/each}
					{#if track.instrument}
						<Badge
							variant="outline"
							href="/?instrument={encodeURIComponent(track.instrument)}"
							class="text-muted-foreground h-4 rounded px-1.5 text-[10px] font-normal"
							title="Show all tracks with {track.instrument}"
						>
							{track.instrument}
						</Badge>
					{/if}
					{#if track.mood}
						<Badge
							variant="outline"
							href="/?mood={encodeURIComponent(track.mood)}"
							class="text-muted-foreground h-4 rounded px-1.5 text-[10px] font-normal"
							title="Show all {track.mood} tracks"
						>
							{track.mood}
						</Badge>
					{/if}
				</div>
			</div>
			<Button
				size="lg"
				disabled={track.audio === 'none'}
				onclick={() => player.toggle(track!)}
				title={track.audio === 'none' ? 'No audio available for this track' : 'Play'}
			>
				{player.track?.idx === track.idx && player.playing ? 'Pause' : 'Play'}
			</Button>
		</div>

		{#if graph}
			{#snippet panel(g: GraphResponse)}
				<!-- One wrapper element on purpose: a snippet with two roots renders as two
				     nodes, and inside the grid below each would become its own cell -- which
				     puts the controls in the 1fr column and squeezes the graph into the
				     sidebar column. -->
				<div>
					<div class="mb-2 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs">
						{#if focus}
							<Button variant="outline" size="xs" onclick={() => (focus = false)}>
								✕ close
							</Button>
						{:else}
							<Button variant="outline" size="xs" onclick={() => (focus = true)}>
								⤢ focus
							</Button>
						{/if}

						<!-- Each slider gets a definite-width wrapper: Slider's own base
						     class is w-full, and as a flex item inside a content-sized
						     <label> that resolves to zero width. -->
						<label class="flex items-center gap-2">
							<span class="text-muted-foreground">neighbours</span>
							<div class="w-24 shrink-0">
								<Slider
									type="single"
									min={5}
									max={50}
									step={1}
									value={k}
									onValueChange={(value) => (k = value)}
									aria-label="Number of neighbours"
								/>
							</div>
							<span class="w-6 tabular-nums">{k}</span>
						</label>

						<label class="flex items-center gap-2">
							<span class="text-muted-foreground">edge ≥</span>
							<div class="w-24 shrink-0">
								<Slider
									type="single"
									min={0}
									max={0.95}
									step={0.05}
									value={minSim}
									onValueChange={(value) => (minSim = value)}
									aria-label="Minimum edge cosine"
								/>
							</div>
							<span class="w-8 tabular-nums">{minSim.toFixed(2)}</span>
						</label>

						<label class="flex items-center gap-1">
							<span class="text-muted-foreground">colour</span>
							<Select.Root type="single" bind:value={colorBy}>
								<Select.Trigger size="sm" class="w-[110px]" aria-label="Colour nodes by">
									<Select.Value />
								</Select.Trigger>
								<Select.Content>
									<Select.Item value="genre">genre</Select.Item>
									<Select.Item value="artist">artist</Select.Item>
								</Select.Content>
							</Select.Root>
						</label>

						<label class="flex items-center gap-1">
							<span class="text-muted-foreground">space</span>
							<Select.Root type="single" bind:value={space}>
								<Select.Trigger size="sm" class="w-[120px]" aria-label="Embedding space">
									<Select.Value />
								</Select.Trigger>
								<Select.Content>
									<Select.Item value="whitened">whitened</Select.Item>
									<Select.Item value="raw">raw</Select.Item>
								</Select.Content>
							</Select.Root>
						</label>
					</div>

					<SimilarityGraph
						graph={g}
						{minSim}
						{colorBy}
						onReseed={reseed}
						height={focus ? 700 : 520}
					/>
				</div>
			{/snippet}

			<div class="mt-4 grid gap-4 xl:grid-cols-[1fr_320px]">
				{#if focus}
					<!-- Focus mode is the one container with a viewport-sized scroll, so it
					     gets the same ScrollArea as the lists rather than a default bar. -->
					<div class="bg-background fixed inset-0 z-50 p-4">
						<ScrollArea class="h-[calc(100dvh-2rem)]" type="always">
							{@render panel(graph)}
						</ScrollArea>
					</div>
				{:else}
					{@render panel(graph)}
				{/if}

				<div class="space-y-4">
					<EnrichmentPanel enrichment={graph.enrichment} k={graph.nodes.length - 1} />

					<Card class="gap-0 py-0">
						<div class="flex items-baseline justify-between border-b px-3 py-2">
							<h3 class="text-sm font-medium">Similar tracks</h3>
							<span class="text-muted-foreground text-xs">cosine</span>
						</div>
						<!-- max-h rather than h: `k` is adjustable down to 5, and a fixed height
						     would leave a large empty panel under a short list. -->
						<ScrollArea class="max-h-[520px]" type="always">
							{#each neighbours as neighbour, index (neighbour.idx)}
								<TrackRow track={neighbour} rank={index + 1} compact />
							{/each}
						</ScrollArea>
					</Card>

					<p class="text-muted-foreground text-[11px] leading-snug">
						Ranked by cosine in the <strong>{space}</strong> space. Whitened is the corpus ZCA
						transform, which spreads the cloud and tends to organise better; raw is what the
						encoder emits. Click any node to recentre the graph on it.
					</p>
				</div>
			</div>
		{/if}
	{/if}
</div>
