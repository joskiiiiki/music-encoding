<script lang="ts">
	import { page } from '$app/state';
	import { api, formatDuration, type ExternalSimilar } from '$lib/api';
	import TrackRow from '$lib/components/TrackRow.svelte';
	import { Alert, AlertDescription } from '$lib/components/ui/alert/index.js';
	import { Button } from '$lib/components/ui/button/index.js';
	import { Card, CardContent } from '$lib/components/ui/card/index.js';
	import { Skeleton } from '$lib/components/ui/skeleton/index.js';

	// An outside song, dropped into the corpus's embedding space. Everything about it is
	// carried in the URL -- the provider's identifiers plus the preview -- so the view is
	// reloadable and shareable like the corpus's own pages.
	const params = $derived(page.url.searchParams);
	const title = $derived(params.get('title') ?? '');
	const artist = $derived(params.get('artist') ?? '');
	const previewUrl = $derived(params.get('preview_url') ?? '');
	const provider = $derived(params.get('provider') ?? 'deezer');
	const externalId = $derived(params.get('id') ?? '');

	let result = $state<ExternalSimilar | null>(null);
	let error = $state<string | null>(null);
	let loading = $state(true);
	let sequence = 0;

	$effect(() => {
		const key = `${provider}|${externalId}|${title}|${artist}`;
		if (!previewUrl && !title) return;
		void key;
		void load();
	});

	async function load() {
		const mine = ++sequence;
		loading = true;
		error = null;
		try {
			const response = await api.externalSimilar({
				preview_url: previewUrl,
				provider,
				id: externalId || undefined,
				title,
				artist,
				k: 20
			});
			if (mine === sequence) result = response;
		} catch (cause) {
			if (mine === sequence) {
				result = null;
				error = cause instanceof Error ? cause.message : String(cause);
			}
		} finally {
			if (mine === sequence) loading = false;
		}
	}
</script>

<div class="pt-4">
	<div class="flex flex-wrap items-start gap-x-6 gap-y-3">
		<div class="min-w-0 flex-1">
			<h1 class="text-xl font-semibold tracking-tight">{title || 'external track'}</h1>
			<p class="text-muted-foreground text-sm">
				{artist || 'unknown artist'}
				{#if result?.query.audio_seconds}
					· {formatDuration(result.query.audio_seconds)} of audio analysed
				{/if}
				· <span class="font-mono text-[11px]">{provider}:{externalId}</span>
			</p>
			<p class="text-muted-foreground mt-1 text-xs">
				Not in the corpus — embedded from a provider preview so it can be compared with
				the {result ? result.items.length : 20} tracks below.
			</p>
		</div>
	</div>

	{#if previewUrl}
		<!-- A plain audio element, not the app's player bar: that bar is built around
		     corpus tracks (an idx, a badge, a link to a track page), and this is a
		     different recording entirely. -->
		<audio class="mt-3 w-full max-w-xl" controls preload="none" src={previewUrl}></audio>
	{/if}

	{#if error}
		<Alert variant="destructive" class="mt-4">
			<AlertDescription>{error}</AlertDescription>
		</Alert>
	{/if}

	{#if loading && !result}
		<div class="mt-4 space-y-2">
			<Skeleton class="h-5 w-64" />
			{#each Array(5) as _, index (index)}
				<Skeleton class="h-14 w-full rounded-lg" />
			{/each}
		</div>
	{:else if result}
		<Alert class="mt-4">
			<AlertDescription class="text-xs">{result.note}</AlertDescription>
		</Alert>

		<div class="mt-4">
			<div class="mb-2 flex items-baseline gap-3">
				<h2 class="text-sm font-medium">Nearest tracks in the corpus</h2>
				<span class="text-muted-foreground text-xs">
					cosine in the {result.space} space · embedded in {result.query.embed_ms} ms
				</span>
				<Button variant="link" size="xs" class="h-auto p-0 text-xs" onclick={() => load()}>
					re-embed
				</Button>
			</div>
			<Card class="gap-0 py-0">
				<CardContent class="px-0">
					{#each result.items as item, index (item.idx)}
						<TrackRow track={item} rank={index + 1} />
					{/each}
				</CardContent>
			</Card>
			<p class="text-muted-foreground mt-3 text-[11px] leading-relaxed">
				Click a track to open its own page, where the similarity graph is available.
				The query is ranked in the <strong>mean-centred</strong> space, not the whitened
				one the corpus views default to: the embeddings are ~97% a shared component, so
				raw cosines are compressed into 0.94–1.0 and whitening amplifies the remaining
				residual in numerically hopeless directions. Mean-centring exposes that residual
				without amplifying it — measured, it put 8 of 10 neighbours on the right artist
				where raw managed 6.
			</p>
		</div>
	{/if}
</div>
