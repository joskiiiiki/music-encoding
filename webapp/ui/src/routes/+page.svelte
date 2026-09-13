<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { api, type Facets, type SearchResponse, type Stats, type Track } from '$lib/api';
	import FacetFilter from '$lib/components/FacetFilter.svelte';
	import TrackRow from '$lib/components/TrackRow.svelte';

	const PAGE_SIZE = 50;

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

	const activeFilters = $derived(
		genres.length + instruments.length + moods.length + (playableOnly ? 1 : 0)
	);
</script>

<div class="grid gap-6 pt-4 lg:grid-cols-[260px_1fr]">
	<aside class="lg:sticky lg:top-4 lg:self-start">
		<div class="bg-card rounded-lg border px-3 py-2">
			<label class="flex cursor-pointer items-center gap-2 py-1 text-xs">
				<input
					type="checkbox"
					class="accent-primary size-3"
					bind:checked={playableOnly}
				/>
				<span class="font-medium">Only tracks with audio</span>
			</label>
			<p class="text-muted-foreground pb-1 text-[10px] leading-snug">
				{stats
					? `${stats.audio.playable.toLocaleString()} of ${stats.audio.total_tracks.toLocaleString()} playable so far`
					: 'audio coverage'}
				{#if stats && stats.audio.unresolved > 0}
					· {stats.audio.unresolved.toLocaleString()} not resolved yet
				{/if}
			</p>
		</div>

		<div class="bg-card mt-4 rounded-lg border px-3 py-1">
			<FacetFilter title="genre" values={facets.genre} selected={genres} limit={14} onselect={(next) => (genres = next)} />
			<FacetFilter
				title="instrument"
				values={facets.instrument}
				selected={instruments}
				limit={10}
				onselect={(next) => (instruments = next)}
			/>
			<FacetFilter title="mood" values={facets.mood} selected={moods} limit={10} onselect={(next) => (moods = next)} />
		</div>
	</aside>

	<section class="min-w-0">
		<div class="flex flex-wrap items-center gap-2">
			<input
				type="search"
				bind:value={q}
				placeholder="Search title, artist or album — e.g. Podington Bear"
				class="border-input bg-background focus:border-ring min-w-[240px] flex-1 rounded-lg border px-3 py-2 text-sm outline-none"
			/>
			<select
				bind:value={sort}
				class="border-input bg-background rounded-lg border px-2 py-2 text-xs"
			>
				<option value="relevance">relevance</option>
				<option value="artist">artist</option>
				<option value="title">title</option>
				<option value="released_desc">newest</option>
				<option value="released_asc">oldest</option>
				<option value="duration_asc">shortest</option>
				<option value="idx">corpus order</option>
			</select>
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
				<button
					type="button"
					class="hover:text-foreground underline"
					onclick={() => {
						genres = [];
						instruments = [];
						moods = [];
						playableOnly = false;
					}}
				>
					reset filters
				</button>
			{/if}
		</div>

		{#if error}
			<p class="text-destructive mt-4 rounded border border-dashed p-3 text-sm">
				{error}
			</p>
		{/if}

		{#if result && !items.length && !loading}
			<p class="text-muted-foreground mt-6 rounded border border-dashed p-6 text-center text-sm">
				No tracks match those filters.
				{#if playableOnly}
					Try turning off “only tracks with audio” — coverage is still filling in.
				{/if}
			</p>
		{/if}

		<div class="bg-card mt-3 rounded-lg border">
			{#each items as track (track.idx)}
				<TrackRow {track} />
			{/each}
		</div>

		{#if result && items.length < result.total}
			<div class="mt-3 flex justify-center">
				<button
					type="button"
					class="border-border hover:bg-accent rounded-lg border px-4 py-2 text-xs"
					disabled={loading}
					onclick={() => run(false)}
				>
					{loading ? 'loading…' : `load ${PAGE_SIZE} more`}
				</button>
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
					Audio: {stats.audio.playable.toLocaleString()} playable, {stats.audio.attempted_misses.toLocaleString()} confirmed unavailable.
				{/if}
			</p>
		{/if}
	</section>
</div>
