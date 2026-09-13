<script lang="ts">
	import type { Enrichment } from '$lib/api';
	import { Card, CardContent } from '$lib/components/ui/card/index.js';
	import {
		Table,
		TableBody,
		TableCell,
		TableHead,
		TableHeader,
		TableRow
	} from '$lib/components/ui/table/index.js';

	let { enrichment, k }: { enrichment: Enrichment; k: number } = $props();

	// Chance is computed over the whole corpus, not by shuffling labels inside this
	// graph: the neighbour set was chosen *because* it is similar to the seed, so a
	// within-graph shuffle already contains the enrichment and reports a lift of ~1
	// even when every neighbour is by the same artist. See api/app/enrichment.py.
	function lift(value: number | null): string {
		if (value === null || !Number.isFinite(value)) return '—';
		return `${value < 10 ? value.toFixed(2) : value.toFixed(0)}×`;
	}

	const rows = $derived([
		{ name: 'same artist', data: enrichment.artist, observed: enrichment.artist.observed },
		{ name: 'same first genre', data: enrichment.genre, observed: enrichment.genre.observed },
		{
			name: 'shared genre tags',
			data: enrichment.tags,
			observed: enrichment.tags.mean_overlap
		}
	]);
</script>

<Card class="gap-0 py-3">
	<CardContent class="px-3">
		<div class="mb-2 flex items-baseline justify-between">
			<h3 class="text-sm font-medium">Is this neighbourhood organised?</h3>
			<span class="text-muted-foreground text-xs">{k} neighbours</span>
		</div>

		<Table>
			<TableHeader>
				<TableRow>
					<TableHead class="h-7 text-xs">signal</TableHead>
					<TableHead class="h-7 text-right text-xs">observed</TableHead>
					<TableHead class="h-7 text-right text-xs">chance</TableHead>
					<TableHead class="h-7 text-right text-xs">lift</TableHead>
				</TableRow>
			</TableHeader>
			<TableBody>
				{#each rows as row (row.name)}
					<TableRow>
						<TableCell class="py-1.5 text-xs">
							{row.name}
							{#if row.data.reason}
								<span class="text-muted-foreground">(n/a)</span>
							{/if}
						</TableCell>
						<TableCell class="py-1.5 text-right text-xs tabular-nums">
							{row.observed.toFixed(3)}
						</TableCell>
						<TableCell class="text-muted-foreground py-1.5 text-right text-xs tabular-nums">
							{row.data.chance.toFixed(4)}
						</TableCell>
						<TableCell class="py-1.5 text-right text-xs font-medium tabular-nums">
							{lift(row.data.lift)}
						</TableCell>
					</TableRow>
				{/each}
			</TableBody>
		</Table>

		<p class="text-muted-foreground mt-2 text-[11px] leading-snug">
			{#if enrichment.tags.reason}
				{enrichment.tags.reason}.
			{:else}
				Observed is the fraction of neighbours agreeing with the seed; chance is what a randomly
				picked corpus track would score, weighted by tag prevalence (genre tags are heavily skewed —
				<em>electronic</em> alone is 29% of the corpus). Measured corpus-wide on this checkpoint:
				same-artist ≈ 80–140×, genre tags ≈ 3.3–4.0×.
			{/if}
			{#if enrichment.artist.reason}
				Note: {enrichment.artist.reason}.
			{/if}
		</p>
	</CardContent>
</Card>
