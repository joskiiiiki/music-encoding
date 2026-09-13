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
		<p class="text-destructive rounded border border-dashed p-4 text-sm">{error}</p>
	{:else if loading && !track}
		<p class="text-muted-foreground text-sm">loading track…</p>
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
						<a
							href="/?genre={encodeURIComponent(tag)}"
							class="bg-secondary text-secondary-foreground hover:bg-accent rounded px-1.5 py-0.5 text-[10px]"
						>
							{tag}
						</a>
					{/each}
					{#if track.instrument}
						<a
							href="/?instrument={encodeURIComponent(track.instrument)}"
							class="bg-muted text-muted-foreground hover:bg-accent rounded px-1.5 py-0.5 text-[10px]"
						>
							{track.instrument}
						</a>
					{/if}
					{#if track.mood}
						<a
							href="/?mood={encodeURIComponent(track.mood)}"
							class="bg-muted text-muted-foreground hover:bg-accent rounded px-1.5 py-0.5 text-[10px]"
						>
							{track.mood}
						</a>
					{/if}
				</div>
			</div>
			<button
				type="button"
				class="bg-primary text-primary-foreground rounded-lg px-3 py-2 text-xs font-medium disabled:opacity-50"
				disabled={track.audio === 'none'}
				onclick={() => player.toggle(track!)}
			>
				{player.track?.idx === track.idx && player.playing ? 'Pause' : 'Play'}
			</button>
		</div>

		{#if graph}
			<div class="mt-4 grid gap-4 xl:grid-cols-[1fr_320px]">
				<div class={focus ? 'bg-background fixed inset-0 z-50 overflow-auto p-4' : ''}>
					<div class="mb-2 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs">
						{#if focus}
							<button
								type="button"
								class="border-border hover:bg-accent rounded border px-2 py-1"
								onclick={() => (focus = false)}
							>
								✕ close
							</button>
						{:else}
							<button
								type="button"
								class="border-border hover:bg-accent rounded border px-2 py-1"
								onclick={() => (focus = true)}
							>
								⤢ focus
							</button>
						{/if}

						<label class="flex items-center gap-2">
							<span class="text-muted-foreground">neighbours</span>
							<input type="range" min="5" max="50" bind:value={k} class="accent-primary w-24" />
							<span class="w-6 tabular-nums">{k}</span>
						</label>

						<label class="flex items-center gap-2">
							<span class="text-muted-foreground">edge ≥</span>
							<input
								type="range"
								min="0"
								max="0.95"
								step="0.05"
								bind:value={minSim}
								class="accent-primary w-24"
							/>
							<span class="w-8 tabular-nums">{minSim.toFixed(2)}</span>
						</label>

						<label class="flex items-center gap-1">
							<span class="text-muted-foreground">colour</span>
							<select bind:value={colorBy} class="border-input bg-background rounded border px-1 py-1">
								<option value="genre">genre</option>
								<option value="artist">artist</option>
							</select>
						</label>

						<label class="flex items-center gap-1">
							<span class="text-muted-foreground">space</span>
							<select bind:value={space} class="border-input bg-background rounded border px-1 py-1">
								<option value="whitened">whitened</option>
								<option value="raw">raw</option>
							</select>
						</label>
					</div>

					<SimilarityGraph
						{graph}
						{minSim}
						{colorBy}
						onReseed={reseed}
						height={focus ? 700 : 520}
					/>
				</div>

				<div class="space-y-4">
					<EnrichmentPanel enrichment={graph.enrichment} k={graph.nodes.length - 1} />

					<div class="bg-card rounded-lg border">
						<div class="flex items-baseline justify-between border-b px-3 py-2">
							<h3 class="text-sm font-medium">Similar tracks</h3>
							<span class="text-muted-foreground text-xs">cosine</span>
						</div>
						<div class="max-h-[520px] overflow-y-auto">
							{#each neighbours as neighbour, index (neighbour.idx)}
								<TrackRow track={neighbour} rank={index + 1} compact />
							{/each}
						</div>
					</div>

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
