<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import {
		api,
		type ExternalSearch,
		type Facets,
		type SearchResponse,
		type Stats,
		type Track
	} from '$lib/api';
	import FacetFilter from '$lib/components/FacetFilter.svelte';
	import TrackRow from '$lib/components/TrackRow.svelte';
	import { Alert, AlertDescription } from '$lib/components/ui/alert/index.js';
	import { Button } from '$lib/components/ui/button/index.js';
	import { Card, CardContent } from '$lib/components/ui/card/index.js';
	import { Checkbox } from '$lib/components/ui/checkbox/index.js';
	import { Input } from '$lib/components/ui/input/index.js';
	import * as Select from '$lib/components/ui/select/index.js';
	import { Skeleton } from '$lib/components/ui/skeleton/index.js';

	const PAGE_SIZE = 50;

	const SORTS = [
		{ value: 'relevance', label: 'relevance' },
		{ value: 'artist', label: 'artist' },
		{ value: 'title', label: 'title' },
		{ value: 'released_desc', label: 'newest' },
		{ value: 'released_asc', label: 'oldest' },
		{ value: 'duration_asc', label: 'shortest' },
		{ value: 'idx', label: 'corpus order' }
	];

	// Initial state comes from the URL, so a filtered view is shareable and survives a
	// reload. It is written back on every change below.
	let q = $state(page.url.searchParams.get('q') ?? '');
	let genres = $state<string[]>(page.url.searchParams.getAll('genre'));
	let instruments = $state<string[]>(page.url.searchParams.getAll('instrument'));
	let moods = $state<string[]>(page.url.searchParams.getAll('mood'));
	let playableOnly = $state(page.url.searchParams.get('playable_only') === '1');
	let sort = $state(page.url.searchParams.get('sort') ?? 'relevance');

	let facets = $state<Facets>({ genre: [], instrument: [], mood: [] });
	let stats = $state<Stats | null>(null);
	let result = $state<SearchResponse | null>(null);
	let items = $state<Track[]>([]);
	let offset = $state(0);
	let loading = $state(false);
	let error = $state<string | null>(null);

	// Guards against out-of-order responses: typing produces overlapping requests and a
	// slow early one must not overwrite a fast later one.
	let sequence = 0;

	async function run(reset: boolean) {
		const mine = ++sequence;
		loading = true;
		error = null;
		const nextOffset = reset ? 0 : offset;
		try {
			const response = await api.search({
				q,
				genre: genres,
				instrument: instruments,
				mood: moods,
				playable_only: playableOnly,
				sort,
				limit: PAGE_SIZE,
				offset: nextOffset
			});
			if (mine !== sequence) return;
			result = response;
			items = reset ? response.items : [...items, ...response.items];
			offset = nextOffset + response.items.length;
		} catch (cause) {
			if (mine === sequence) error = cause instanceof Error ? cause.message : String(cause);
		} finally {
			if (mine === sequence) loading = false;
		}
	}

	function syncUrl() {
		const params = new URLSearchParams();
		if (q) params.set('q', q);
		for (const value of genres) params.append('genre', value);
		for (const value of instruments) params.append('instrument', value);
		for (const value of moods) params.append('mood', value);
		if (playableOnly) params.set('playable_only', '1');
		if (sort !== 'relevance') params.set('sort', sort);
		const query = params.toString();
		void goto(query ? `/?${query}` : '/', {
			replaceState: true,
			keepFocus: true,
			noScroll: true
		});
	}

	let timer: ReturnType<typeof setTimeout> | undefined;
	$effect(() => {
		// Reading these registers the dependencies; the timeout is the debounce.
		void [q, genres, instruments, moods, playableOnly, sort];
		clearTimeout(timer);
		timer = setTimeout(() => {
			void run(true);
			syncUrl();
		}, 200);
		return () => clearTimeout(timer);
	});

	$effect(() => {
		void api.facets().then((value) => (facets = value));
		void api.stats().then((value) => (stats = value));
	});

	// Songs the corpus does not have, searched on Deezer/iTunes. A separate, slower
	// debounce than the corpus search: these are third-party calls with rate limits, and
	// there is no reason to ask on every keystroke of a one-letter query.
	let external = $state<ExternalSearch | null>(null);
	let externalLoading = $state(false);
	let externalTimer: ReturnType<typeof setTimeout> | undefined;

	$effect(() => {
		const query = q.trim();
		clearTimeout(externalTimer);
		if (query.length < 3) {
			external = null;
			return;
		}
		externalTimer = setTimeout(() => {
			externalLoading = true;
			api
				.externalSearch(query, 6)
				.then((value) => (external = value))
				.catch(() => (external = null))
				.finally(() => (externalLoading = false));
		}, 600);
		return () => clearTimeout(externalTimer);
	});

	function externalHref(result: ExternalSearch['results'][number]) {
		const params = new URLSearchParams({
			provider: result.provider,
			id: result.id,
			title: result.title,
			artist: result.artist,
			preview_url: result.preview_url
		});
		return `/external?${params}`;
	}

	const activeFilters = $derived(
		genres.length + instruments.length + moods.length + (playableOnly ? 1 : 0)
	);

	function resetFilters() {
		genres = [];
		instruments = [];
		moods = [];
		playableOnly = false;
	}
</script>

<div class="grid gap-6 pt-4 lg:grid-cols-[260px_1fr]">
	<aside class="lg:sticky lg:top-4 lg:self-start">
		<Card class="gap-0 py-2">
			<CardContent class="px-3">
				<div class="flex items-center gap-2 py-1">
					<Checkbox id="playable-only" bind:checked={playableOnly} />
					<label for="playable-only" class="cursor-pointer text-xs font-medium">
						Only tracks with audio
					</label>
				</div>
				<p class="text-muted-foreground pb-1 text-[10px] leading-snug">
					{stats
						? `${stats.audio.playable.toLocaleString()} of ${stats.audio.total_tracks.toLocaleString()} playable so far`
						: 'audio coverage'}
					{#if stats && stats.audio.unresolved > 0}
						· {stats.audio.unresolved.toLocaleString()} not resolved yet
					{/if}
				</p>
			</CardContent>
		</Card>

		<Card class="mt-4 gap-0 py-1">
			<CardContent class="px-3">
				<FacetFilter
					title="genre"
					values={facets.genre}
					selected={genres}
					limit={14}
					onselect={(next) => (genres = next)}
				/>
				<FacetFilter
					title="instrument"
					values={facets.instrument}
					selected={instruments}
					limit={10}
					onselect={(next) => (instruments = next)}
				/>
				<FacetFilter
					title="mood"
					values={facets.mood}
					selected={moods}
					limit={10}
					onselect={(next) => (moods = next)}
				/>
			</CardContent>
		</Card>
	</aside>

	<section class="min-w-0">
		<div class="flex flex-wrap items-center gap-2">
			<Input
				type="search"
				bind:value={q}
				placeholder="Search title, artist or album — e.g. Podington Bear"
				class="min-w-[240px] flex-1"
			/>
			<Select.Root type="single" bind:value={sort}>
				<Select.Trigger size="sm" class="w-[150px]" aria-label="Sort results">
					<Select.Value placeholder="sort" />
				</Select.Trigger>
				<Select.Content>
					{#each SORTS as option (option.value)}
						<Select.Item value={option.value}>{option.label}</Select.Item>
					{/each}
				</Select.Content>
			</Select.Root>
		</div>

		<div class="text-muted-foreground mt-2 flex items-center gap-3 text-xs">
			<span>
				{#if result}
					<strong class="text-foreground">{result.total.toLocaleString()}</strong> tracks
					{#if q}match “{q}”{/if}
					{#if activeFilters}&nbsp;· {activeFilters} filter{activeFilters === 1 ? '' : 's'}{/if}
				{:else}
					loading…
				{/if}
			</span>
			{#if loading}<span>· working…</span>{/if}
			{#if activeFilters}
				<Button variant="link" size="xs" class="h-auto p-0 text-xs" onclick={resetFilters}>
					reset filters
				</Button>
			{/if}
		</div>

		{#if error}
			<Alert variant="destructive" class="mt-4">
				<AlertDescription>{error}</AlertDescription>
			</Alert>
		{/if}

		{#if !result && loading}
			<div class="mt-3 space-y-2">
				{#each Array(6) as _, index (index)}
					<Skeleton class="h-14 w-full rounded-lg" />
				{/each}
			</div>
		{/if}

		{#if result && !items.length && !loading}
			<Alert class="mt-6">
				<AlertDescription>
					No tracks match those filters.
					{#if playableOnly}
						Try turning off “only tracks with audio” — coverage is still filling in.
					{/if}
				</AlertDescription>
			</Alert>
		{/if}

		{#if external && external.results.length}
			<!-- Outside songs. Clicking one embeds its preview and shows the nearest tracks
			     the corpus does have, which is the whole point of the space. -->
			<Card class="mt-3 gap-0 py-3">
				<CardContent class="px-3">
					<div class="mb-2 flex flex-wrap items-baseline gap-x-2">
						<h3 class="text-sm font-medium">Not in the corpus — Deezer / iTunes</h3>
						<span class="text-muted-foreground text-xs">
							click one to find its nearest tracks here
						</span>
					</div>
					<div class="space-y-0.5">
						{#each external.results as result (result.provider + result.id)}
							<a
								href={externalHref(result)}
								class="hover:bg-muted flex items-center gap-2 rounded px-1.5 py-1"
							>
								<span class="text-muted-foreground w-12 shrink-0 text-[10px] uppercase">
									{result.provider}
								</span>
								<span class="min-w-0 flex-1 truncate text-xs">
									<span class="font-medium">{result.title}</span>
									— {result.artist}{#if result.album} · {result.album}{/if}
								</span>
								{#if result.seconds}
									<span class="text-muted-foreground shrink-0 text-[10px] tabular-nums">
										{Math.round(result.seconds)}s
									</span>
								{/if}
							</a>
						{/each}
					</div>
				</CardContent>
			</Card>
		{:else if externalLoading && q.trim().length >= 3}
			<Skeleton class="mt-3 h-16 w-full rounded-lg" />
		{/if}

		{#if items.length}
			<Card class="mt-3 gap-0 py-0">
				<CardContent class="px-0">
					{#each items as track (track.idx)}
						<TrackRow {track} />
					{/each}
				</CardContent>
			</Card>
		{/if}

		{#if result && items.length < result.total}
			<div class="mt-3 flex justify-center">
				<Button variant="outline" size="sm" disabled={loading} onclick={() => run(false)}>
					{loading ? 'loading…' : `load ${PAGE_SIZE} more`}
				</Button>
			</div>
		{/if}

		{#if stats}
			<p class="text-muted-foreground mt-6 text-[11px] leading-relaxed">
				Index: {stats.tracks.toLocaleString()} MTG-Jamendo tracks, {stats.dim}-dimensional Barlow Twins
				embeddings ({stats.spaces.join(' / ')}), built {stats.meta.built_at?.slice(0, 10) ?? '—'}.
				Similarity is exact cosine over the whole corpus — no approximate index. ·
				{#if stats.audio.unresolved > 0}
					Audio: {stats.audio.playable.toLocaleString()} playable
					({stats.audio.local} local full tracks, {stats.audio.deezer + stats.audio.itunes} previews),
					{stats.audio.unresolved.toLocaleString()} not yet looked up.
				{:else}
					Audio: {stats.audio.playable.toLocaleString()} playable,
					{stats.audio.attempted_misses.toLocaleString()} confirmed unavailable.
				{/if}
			</p>
		{/if}
	</section>
</div>
